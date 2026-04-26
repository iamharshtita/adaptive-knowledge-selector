"""
Text Knowledge Source - Document Retrieval
Integrates PubMed and DailyMed for medical literature and drug information
"""

import os
import requests
from typing import List, Dict, Any
import time
from Bio import Entrez
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle


class TextKnowledgeSource:
    """
    Retrieves information from text-based medical knowledge sources
    """

    def __init__(self, email: str, api_key: str = None):
        """
        Initialize the text knowledge source

        Args:
            email: Email for NCBI API (required)
            api_key: NCBI API key (optional but recommended)
        """
        # Setup NCBI Entrez
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        # Initialize embedding model
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.document_metadata = []

    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for relevant articles

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of article dictionaries with title, abstract, pmid, etc.
        """
        try:
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record["IdList"]

            if not id_list:
                return []

            # Fetch article details
            handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="abstract",
                retmode="xml"
            )
            articles = Entrez.read(handle)
            handle.close()

            results = []
            for article in articles['PubmedArticle']:
                try:
                    medline = article['MedlineCitation']
                    article_data = medline['Article']

                    # Extract abstract
                    abstract = ""
                    if 'Abstract' in article_data:
                        abstract_parts = article_data['Abstract']['AbstractText']
                        abstract = " ".join([str(part) for part in abstract_parts])

                    dates = article_data.get('ArticleDate', [])
                    results.append({
                        'pmid': str(medline['PMID']),
                        'title': str(article_data['ArticleTitle']),
                        'abstract': abstract,
                        'journal': str(article_data.get('Journal', {}).get('Title', '')),
                        'pub_date': str(dates[0].get('Year', '')) if dates else ''
                    })
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue

            return results

        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def search_dailymed(self, drug_name: str) -> Dict[str, Any]:
        """
        Search DailyMed for drug label information

        Args:
            drug_name: Name of the drug

        Returns:
            Dictionary with drug label information
        """
        base_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

        try:
            # Search for drug
            params = {'drug_name': drug_name}
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get('data'):
                return None

            # Get the first result's setid
            setid = data['data'][0]['setid']

            # Fetch detailed information
            detail_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json"
            detail_response = requests.get(detail_url, timeout=10)
            detail_response.raise_for_status()

            return detail_response.json()

        except Exception as e:
            print(f"Error searching DailyMed: {e}")
            return None

    def build_vector_store(self, documents: List[Dict[str, str]]):
        """
        Build FAISS vector store from documents

        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        print(f"Building vector store with {len(documents)} documents...")

        # Extract text and create embeddings
        texts = [doc['text'] for doc in documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        # Store documents and metadata
        self.documents = texts
        self.document_metadata = [doc.get('metadata', {}) for doc in documents]

        print(f"Vector store built with {self.index.ntotal} documents")

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the vector store

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        if self.index is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        # Encode query
        query_embedding = self.encoder.encode([query])

        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'text': self.documents[idx],
                'metadata': self.document_metadata[idx],
                'score': float(dist)
            })

        return results

    def save_vector_store(self, path: str):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save documents and metadata
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)

        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        # Load documents and metadata
        with open(os.path.join(path, "documents.pkl"), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['metadata']

        print(f"Vector store loaded from {path}")


    def query(self, text: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Unified query interface — takes any text query, searches PubMed,
        and returns a standardised result dict.

        Args:
            text: Free-text medical query

        Returns:
            Dict with 'answer' (str summary) and 'results' (raw list)
        """
        results = self.search_pubmed(text, max_results=max_results)

        if not results:
            return {
                "source": "TextKnowledgeSource",
                "answer": f"No PubMed articles found for: '{text}'",
                "results": [],
            }

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            pmid = r.get("pmid", "?")
            journal = r.get("journal", "")
            abstract = r.get("abstract", "")[:200]
            lines.append(f"{i}. {title}\n   PMID: {pmid}  Journal: {journal}\n   {abstract}…")

        return {
            "source": "TextKnowledgeSource",
            "answer": "\n".join(lines),
            "results": results,
        }


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  TEXT KNOWLEDGE SOURCE — Standalone Test")
    print("=" * 70)

    ts = TextKnowledgeSource(
        email="test@example.com",
        api_key=os.getenv("NCBI_API_KEY"),
    )

    queries = [
        "Latest research on statin therapy",
        "aspirin drug interactions",
        "CRISPR gene editing clinical trials",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"🔎 Query: {q}")
        print(f"{'─' * 60}")
        out = ts.query(q, max_results=3)
        print(out["answer"])

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")

