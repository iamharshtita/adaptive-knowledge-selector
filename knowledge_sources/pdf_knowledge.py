"""
PDF Knowledge Source - Advanced Document Retrieval
Ingests PDF documents, applies sentence-aware chunking, and provides hybrid
semantic + keyword search via FAISS and BM25.

Key improvements over baseline:
  - Sentence-aware chunking that respects paragraph / sentence boundaries
  - Cosine-similarity scoring (normalised 0-1 relevance)
  - Optional BM25 keyword search with Reciprocal-Rank Fusion
  - Incremental indexing (add documents without rebuilding)
  - PDF metadata extraction (title, author, creation date)
  - Near-duplicate result deduplication
  - Proper Python logging instead of bare print()
"""

import os
import re
import math
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
import faiss
import pypdf
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
)


# ---------------------------------------------------------------------------
# Lightweight BM25 implementation (no extra dependency)
# ---------------------------------------------------------------------------
class _BM25:
    """Okapi BM25 scorer operating on pre-tokenised documents."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.n_docs = len(corpus)

        # Inverse document frequency
        self.idf: Dict[str, float] = {}
        df: Dict[str, int] = {}
        for doc in corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)
        for word, freq in df.items():
            self.idf[word] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for q in query_tokens:
            idf_val = self.idf.get(q, 0.0)
            for i, doc in enumerate(self.corpus):
                tf = doc.count(q)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf_val * (numerator / denominator)
        return scores


# ---------------------------------------------------------------------------
# PDF Knowledge Source
# ---------------------------------------------------------------------------
class PDFKnowledgeSource:
    """
    Retrieves information from ingested PDF documents using hybrid search.

    Workflow
    --------
    1. ``ingest_pdf`` / ``ingest_directory`` -> list of chunk dicts
    2. ``build_vector_store`` -> FAISS index  (can call repeatedly to add more)
    3. ``semantic_search`` / ``keyword_search`` / ``hybrid_search``
    4. ``save_vector_store`` / ``load_vector_store`` for persistence
    """

    # Default embedding model — compact and fast
    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 150,
        model_name: str = None,
    ):
        """
        Initialise the PDF knowledge source.

        Args:
            chunk_size:  Target size of each text chunk (characters).
            overlap:     Character overlap between consecutive chunks.
            model_name:  HuggingFace model id for the sentence encoder.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        model_name = model_name or self._DEFAULT_MODEL
        logger.info("Loading embedding model '%s' …", model_name)
        self.encoder = SentenceTransformer(model_name)
        self._embedding_dim: Optional[int] = None

        # Vector store state
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner-Product (cosine)
        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []

        # BM25 state (rebuilt lazily)
        self._bm25: Optional[_BM25] = None
        self._tokenised_docs: List[List[str]] = []

    # ------------------------------------------------------------------
    # Text extraction & chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf_metadata(reader: pypdf.PdfReader) -> Dict[str, str]:
        """Pull title / author / subject from the PDF info dict."""
        meta: Dict[str, str] = {}
        info = reader.metadata
        if info is None:
            return meta
        for key, attr in [("title", "/Title"), ("author", "/Author"),
                          ("subject", "/Subject"), ("creator", "/Creator")]:
            val = info.get(attr, None)
            if val:
                meta[key] = str(val).strip()
        return meta

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lower-case word tokenisation for BM25."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _sentence_aware_chunk(self, text: str) -> List[str]:
        """
        Split *text* into chunks that respect sentence boundaries wherever
        possible, falling back to whitespace / hard character limits.
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if current_len + len(sent) > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Keep overlap by retaining trailing sentences
                overlap_text = " ".join(current)
                keep: List[str] = []
                keep_len = 0
                for s in reversed(current):
                    if keep_len + len(s) > self.overlap:
                        break
                    keep.insert(0, s)
                    keep_len += len(s) + 1
                current = keep
                current_len = keep_len

            current.append(sent)
            current_len += len(sent) + 1

        if current:
            chunks.append(" ".join(current))

        # Safety: break any remaining oversized chunks at whitespace
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size * 1.3:
                final.append(chunk)
            else:
                start = 0
                while start < len(chunk):
                    end = min(start + self.chunk_size, len(chunk))
                    if end < len(chunk):
                        # Back-track to last space
                        space = chunk.rfind(" ", start, end)
                        if space > start:
                            end = space
                    final.append(chunk[start:end].strip())
                    start = end
        return [c for c in final if len(c) > 50]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read a PDF file and return a list of chunk dicts.

        Returns:
            List of ``{'text': str, 'metadata': dict}`` dictionaries.
        """
        if not os.path.exists(file_path):
            logger.warning("File not found: %s", file_path)
            return []

        try:
            reader = pypdf.PdfReader(file_path)
            file_name = os.path.basename(file_path)
            pdf_meta = self._extract_pdf_metadata(reader)

            # Concatenate all page text (preserving page boundaries for metadata)
            page_texts: List[Tuple[int, str]] = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    page_texts.append((page_num + 1, text))

            if not page_texts:
                logger.warning("No extractable text in %s", file_name)
                return []

            # Build chunks across the entire document
            full_text = "\n\n".join(t for _, t in page_texts)
            raw_chunks = self._sentence_aware_chunk(full_text)

            # Map each chunk back to its approximate page number
            extracted: List[Dict[str, Any]] = []
            char_cursor = 0
            page_boundaries = []
            running = 0
            for pnum, ptxt in page_texts:
                page_boundaries.append((running, running + len(ptxt), pnum))
                running += len(ptxt) + 2  # +2 for the "\n\n" join

            for chunk in raw_chunks:
                # Find where this chunk starts in the full text
                idx = full_text.find(chunk[:80], char_cursor)
                if idx == -1:
                    idx = full_text.find(chunk[:80])
                if idx != -1:
                    char_cursor = idx

                # Determine page
                page_num = 1
                for start_b, end_b, pnum in page_boundaries:
                    if start_b <= max(idx, 0) < end_b:
                        page_num = pnum
                        break

                meta = {
                    "source": file_name,
                    "page": page_num,
                    "chunk_chars": len(chunk),
                }
                meta.update(pdf_meta)
                extracted.append({"text": chunk, "metadata": meta})

            logger.info("Extracted %d chunks from %s (%d pages)",
                        len(extracted), file_name, len(reader.pages))
            return extracted

        except Exception as exc:
            logger.error("Error parsing PDF %s: %s", file_path, exc)
            return []

    def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Ingest every PDF found (recursively) under *directory_path*."""
        all_chunks: List[Dict[str, Any]] = []
        if not os.path.isdir(directory_path):
            logger.warning("Directory not found: %s", directory_path)
            return all_chunks

        pdf_files = []
        for root, _, files in os.walk(directory_path):
            for f in sorted(files):
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

        logger.info("Found %d PDF file(s) in %s", len(pdf_files), directory_path)
        for path in pdf_files:
            chunks = self.ingest_pdf(path)
            all_chunks.extend(chunks)

        logger.info("Total chunks after directory ingestion: %d", len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def _normalise(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise so Inner-Product == Cosine Similarity."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype("float32")

    def build_vector_store(self, documents: List[Dict[str, Any]]):
        """
        Build (or extend) the FAISS index from a list of document chunks.

        Can be called multiple times to add new documents incrementally.

        Args:
            documents: list of dicts with ``'text'`` and ``'metadata'`` keys.
        """
        if not documents:
            logger.warning("No documents provided — nothing to index.")
            return

        texts = [d["text"] for d in documents]
        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = self._normalise(embeddings)

        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        # Create or extend index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self._embedding_dim)

        self.index.add(embeddings)
        self.documents.extend(texts)
        self.document_metadata.extend([d.get("metadata", {}) for d in documents])

        # Rebuild BM25
        self._tokenised_docs = [self._tokenise(t) for t in self.documents]
        self._bm25 = _BM25(self._tokenised_docs)

        logger.info("Vector store now contains %d chunks", self.index.ntotal)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Dense retrieval via cosine similarity.

        Returns list of dicts with ``text``, ``metadata``, ``score`` (0-1).
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Call build_vector_store first.")

        q_emb = self._normalise(self.encoder.encode([query]))
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        seen_texts: set = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            text = self.documents[idx]
            # Deduplicate near-identical chunks
            sig = text[:120]
            if sig in seen_texts:
                continue
            seen_texts.add(sig)
            results.append({
                "text": text,
                "metadata": self.document_metadata[idx],
                "score": float(max(score, 0.0)),  # cosine ∈ [-1, 1], clamp to 0
            })
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sparse retrieval via BM25 keyword matching.

        Returns list of dicts with ``text``, ``metadata``, ``score``.
        """
        if self._bm25 is None:
            raise ValueError("BM25 index not built. Call build_vector_store first.")

        q_tokens = self._tokenise(query)
        raw_scores = self._bm25.score(q_tokens)

        top_indices = np.argsort(raw_scores)[::-1][:top_k]
        max_score = raw_scores[top_indices[0]] if len(top_indices) else 1.0
        if max_score == 0:
            max_score = 1.0

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            if raw_scores[idx] <= 0:
                break
            results.append({
                "text": self.documents[idx],
                "metadata": self.document_metadata[idx],
                "score": float(raw_scores[idx] / max_score),  # normalise to 0-1
            })
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal-Rank Fusion of semantic + BM25 results.

        Args:
            query:            Search query.
            top_k:            Number of results to return.
            semantic_weight:  Weight for the semantic component (0-1).

        Returns:
            Fused ranked list of dicts with ``text``, ``metadata``, ``score``.
        """
        k_rrf = 60  # RRF constant
        fetch_k = min(top_k * 3, self.index.ntotal) if self.index else top_k * 3

        sem_results = self.semantic_search(query, top_k=fetch_k)
        bm25_results = self.keyword_search(query, top_k=fetch_k)

        # Map text signature -> aggregated score
        fusion: Dict[str, float] = {}
        meta_map: Dict[str, Tuple[str, Dict]] = {}

        for rank, r in enumerate(sem_results):
            sig = r["text"][:120]
            rr = semantic_weight / (k_rrf + rank + 1)
            fusion[sig] = fusion.get(sig, 0.0) + rr
            meta_map[sig] = (r["text"], r["metadata"])

        bm25_weight = 1.0 - semantic_weight
        for rank, r in enumerate(bm25_results):
            sig = r["text"][:120]
            rr = bm25_weight / (k_rrf + rank + 1)
            fusion[sig] = fusion.get(sig, 0.0) + rr
            if sig not in meta_map:
                meta_map[sig] = (r["text"], r["metadata"])

        # Sort by fused score descending
        ranked = sorted(fusion.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Normalise scores to 0-1
        max_fused = ranked[0][1] if ranked else 1.0
        results: List[Dict[str, Any]] = []
        for sig, sc in ranked:
            text, meta = meta_map[sig]
            results.append({
                "text": text,
                "metadata": meta,
                "score": float(sc / max_fused),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vector_store(self, path: str):
        """Persist FAISS index, documents, metadata, and BM25 state to disk."""
        if self.index is None:
            logger.warning("Nothing to save — vector store is empty.")
            return

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "pdf_index.faiss"))

        with open(os.path.join(path, "pdf_documents.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.document_metadata,
                "embedding_dim": self._embedding_dim,
            }, f)

        logger.info("Vector store saved to %s (%d chunks)", path, len(self.documents))

    def load_vector_store(self, path: str):
        """Load a previously saved vector store from disk."""
        index_path = os.path.join(path, "pdf_index.faiss")
        docs_path = os.path.join(path, "pdf_documents.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"Vector store files not found in {path}")

        self.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.document_metadata = data["metadata"]
            self._embedding_dim = data.get("embedding_dim", self.index.d)

        # Rebuild BM25
        self._tokenised_docs = [self._tokenise(t) for t in self.documents]
        self._bm25 = _BM25(self._tokenised_docs)

        logger.info("Loaded vector store from %s — %d chunks", path, len(self.documents))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the current index."""
        sources: Dict[str, int] = {}
        for m in self.document_metadata:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        avg_len = (
            sum(len(d) for d in self.documents) / len(self.documents)
            if self.documents else 0
        )
        return {
            "total_chunks": len(self.documents),
            "index_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self._embedding_dim,
            "avg_chunk_chars": round(avg_len),
            "sources": sources,
        }

    def query(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Unified query interface — takes any text query, searches the
        PDF vector store, and returns a standardised result dict.

        Automatically loads the vector store from disk if not already loaded.

        Args:
            text: Free-text query
            top_k: Number of results to retrieve (default: 3)

        Returns:
            Dict with consolidated 'answer', best match details, and score
        """
        # Auto-load vector store if not already initialised
        if self.index is None:
            store_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "pdf_store",
            )
            if os.path.exists(os.path.join(store_dir, "pdf_index.faiss")):
                self.load_vector_store(store_dir)
            else:
                return {
                    "source": "PDFKnowledgeSource",
                    "answer": "No PDF vector store found. Ingest PDFs first.",
                    "score": 0.0,
                    "pdf_source": None,
                    "page": None,
                    "results": [],
                }

        results = self.semantic_search(text, top_k=top_k)

        if not results:
            return {
                "source": "PDFKnowledgeSource",
                "answer": f"No relevant PDF chunks found for: '{text}'",
                "score": 0.0,
                "pdf_source": None,
                "page": None,
                "results": [],
            }

        # Get the best match (highest score)
        best_match = results[0]
        pdf_source = best_match.get("metadata", {}).get("source", "Unknown")
        page_num = best_match.get("metadata", {}).get("page", "Unknown")
        score = best_match.get("score", 0.0)
        content = best_match.get("text", "")

        # Calculate approximate line numbers (assuming ~80 chars per line)
        lines_in_chunk = len(content.split('\n'))
        approx_start_line = (page_num - 1) * 50 if isinstance(page_num, int) else 0
        approx_end_line = approx_start_line + lines_in_chunk

        # Create consolidated answer
        consolidated_answer = (
            f"[Best Match - Score: {score:.4f}]\n"
            f"Source: {pdf_source}\n"
            f"Page: {page_num}\n"
            f"Approx. Lines: {approx_start_line}-{approx_end_line}\n\n"
            f"Content:\n{content[:500]}{'...' if len(content) > 500 else ''}"
        )

        return {
            "source": "PDFKnowledgeSource",
            "answer": consolidated_answer,
            "score": score,
            "pdf_source": pdf_source,
            "page": page_num,
            "line_range": f"{approx_start_line}-{approx_end_line}",
            "results": results,  # Keep all results for advanced users
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  PDF KNOWLEDGE SOURCE — Standalone Test")
    print("=" * 70)

    pdf = PDFKnowledgeSource()

    # Load store
    store_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdf_store")
    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdfs")

    if os.path.exists(os.path.join(store_dir, "pdf_index.faiss")):
        pdf.load_vector_store(store_dir)
    elif os.path.isdir(pdf_dir):
        print("Building vector store from PDFs …")
        chunks = pdf.ingest_directory(pdf_dir)
        if chunks:
            pdf.build_vector_store(chunks)
            pdf.save_vector_store(store_dir)
        else:
            print("❌ No chunks extracted"); sys.exit(1)
    else:
        print(f"❌ No PDFs found at {pdf_dir}"); sys.exit(1)

    print(f"Index: {pdf.index.ntotal} chunks\n")

    queries = [
        "What criteria does the WHO use for selecting essential drugs?",
        "What is the seventh list of essential drugs according to WHO Technical Report Series 825?",
        "What are the guidelines for establishing a national programme for essential drugs?",
        "What medication types are covered in BHMEDS-R3 for behavioral health?",
        "What categories of drugs does Médecins Sans Frontières classify in their essential drugs guide?",
        "What are the acute health risks associated with marijuana according to NIDA?"
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"🔎 Query: {q}")
        print(f"{'─' * 60}")
        out = pdf.query(q, top_k=3)
        print(out["answer"])

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")

