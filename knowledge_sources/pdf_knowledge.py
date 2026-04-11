"""
PDF Knowledge Source - Document Retrieval
Ingests PDF documents, chunks text, and provides semantic search via FAISS.
"""

import os
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import pypdf

class PDFKnowledgeSource:
    """
    Retrieves information from ingested PDF documents using semantic search.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the PDF knowledge source
        
        Args:
            chunk_size: Approximate size of each text chunk (in characters)
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.index = None
        self.documents = []
        self.document_metadata = []

    def ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read a PDF file and split its content into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document dictionaries with 'text' and 'metadata' keys.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
            
        try:
            reader = pypdf.PdfReader(file_path)
            file_name = os.path.basename(file_path)
            
            extracted_chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                    
                # Basic chunking by character count with overlap
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    
                    # Try to break at a space if we're not at the end
                    if end < len(text):
                        while end > start and text[end] not in (' ', '\n', '.', ','):
                            end -= 1
                        if end == start: # Fallback if no natural break found
                            end = start + self.chunk_size
                            
                    chunk = text[start:end].strip()
                    if len(chunk) > 50: # Ignore very small fragments
                        extracted_chunks.append({
                            'text': chunk,
                            'metadata': {
                                'source': file_name,
                                'page': page_num + 1
                            }
                        })
                    
                    if end >= len(text):
                        break
                    start = end - self.overlap
                    if start < 0:
                        start = 0
                        
            print(f"Extracted {len(extracted_chunks)} chunks from {file_name}")
            return extracted_chunks
            
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return []
            
    def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Ingest all PDF files from a directory.
        """
        all_chunks = []
        if not os.path.isdir(directory_path):
            print(f"Directory not found: {directory_path}")
            return all_chunks
            
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    chunks = self.ingest_pdf(file_path)
                    all_chunks.extend(chunks)
                    
        return all_chunks

    def build_vector_store(self, documents: List[Dict[str, Any]]):
        """
        Build FAISS vector store from documents
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        if not documents:
            print("No documents to index.")
            return
            
        print(f"Building vector store with {len(documents)} document chunks...")
        
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
        
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)
            
        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "documents.pkl"), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['metadata']
            
        print(f"Vector store loaded from {path} with {len(self.documents)} doc chunks.")

if __name__ == "__main__":
    print("PDFKnowledgeSource initialized.")
