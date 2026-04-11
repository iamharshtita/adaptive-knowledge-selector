import os
import sys

# Add parent directory to path to allow importing knowledge_sources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.pdf_knowledge import PDFKnowledgeSource

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "pdf")
    vector_store_dir = os.path.join(project_root, "data", "pdf_knowledge")
    
    # Initialize the source
    print("Initializing PDF Knowledge Source...")
    pdf_source = PDFKnowledgeSource(chunk_size=1000, overlap=150)
    
    # Ingest the directory
    print(f"\nIngesting PDFs from: {pdf_dir}")
    chunks = pdf_source.ingest_directory(pdf_dir)
    
    if chunks:
        # Build vector store
        print(f"\nBuilding vector store from {len(chunks)} chunks...")
        pdf_source.build_vector_store(chunks)
        
        # Save vector store for future use
        print(f"\nSaving vector store to: {vector_store_dir}")
        pdf_source.save_vector_store(vector_store_dir)
        print("Done! You can now load this vector store quickly in other scripts.")
    else:
        print("No valid PDF chunks were extracted. Please ensure there are well-formatted PDFs in the folder.")

if __name__ == "__main__":
    main()
