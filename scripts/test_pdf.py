import sys
import os

# Add parent directory to path to allow importing knowledge_sources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.pdf_knowledge import PDFKnowledgeSource

def main():
    print("Initializing PDFKnowledgeSource...")
    pdf_source = PDFKnowledgeSource(chunk_size=500, overlap=50)
    
    # Check if a PDF file was provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use a dummy pdf from root if not provided
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "krrprojoverview.pdf")
        
    abs_pdf_path = os.path.abspath(pdf_path)
        
    print(f"Ingesting PDF: {abs_pdf_path}")
    chunks = pdf_source.ingest_pdf(abs_pdf_path)
    
    if chunks:
        print(f"First chunk preview: {chunks[0]['text'][:100]}...")
        
        # Build vector store
        pdf_source.build_vector_store(chunks)
        
        # Search for a term, e.g., 'knowledge'
        query = "knowledge extraction"
        print(f"\nSearching for: '{query}'")
        results = pdf_source.semantic_search(query, top_k=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f})")
            print(f"Page: {result['metadata']['page']}")
            print(f"Text: {result['text'][:200]}...")
    else:
        print("Failed to extract chunks or file not found.")

if __name__ == "__main__":
    main()
