"""
Quick Setup Script for Hetionet Knowledge Graph
Downloads and prepares Hetionet - No account needed, 100% free!

Run this script to get your knowledge graph ready in minutes.
"""

import os
import sys
import requests
import bz2
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def download_hetionet(data_dir: str = "data/hetionet"):
    """
    Download Hetionet knowledge graph

    Args:
        data_dir: Directory to save the data
    """
    # Create directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    compressed_file = os.path.join(data_dir, "hetionet-v1.0.json.bz2")
    json_file = os.path.join(data_dir, "hetionet-v1.0.json")

    # Check if already downloaded
    if os.path.exists(json_file):
        print(f"✓ Hetionet already exists: {json_file}")
        return json_file

    print("=" * 80)
    print("DOWNLOADING HETIONET")
    print("=" * 80)

    # Download
    url = "https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2"
    print(f"\nDownloading from: {url}")
    print("File size: ~260MB (compressed)")
    print("This may take a few minutes depending on your connection...\n")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(compressed_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 50
                        filled = int(bar_length * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        print(f"\r[{bar}] {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='', flush=True)

        print("\n\n✓ Download complete!")

    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        print("\nAlternative: Manual download")
        print("1. Visit: https://github.com/hetio/hetionet/tree/master/hetnet/json")
        print("2. Download: hetionet-v1.0.json.bz2")
        print(f"3. Place it in: {compressed_file}")
        print("4. Run this script again")
        return None

    # Decompress
    print("\nDecompressing file...")
    try:
        with bz2.open(compressed_file, 'rb') as f_in:
            with open(json_file, 'wb') as f_out:
                # Decompress in chunks
                while True:
                    chunk = f_in.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)

        print(f"✓ Decompressed to: {json_file}")

        # Remove compressed file to save space
        os.remove(compressed_file)
        print("✓ Removed compressed file to save space")

        return json_file

    except Exception as e:
        print(f"❌ Decompression failed: {e}")
        return None


def load_and_test_hetionet(json_file: str):
    """Load Hetionet into knowledge graph and run tests"""

    print("\n" + "=" * 80)
    print("LOADING INTO KNOWLEDGE GRAPH")
    print("=" * 80)

    try:
        kg = KnowledgeGraphSource()
        print("\nLoading Hetionet (this takes 1-2 minutes)...")
        kg.load_hetionet(json_file)
        print("✓ Knowledge graph loaded successfully!")

        # Save as pickle for faster future loading
        pickle_path = "data/hetionet/hetionet_graph.pkl"
        print(f"\nSaving to pickle for faster loading: {pickle_path}")
        kg.save_graph(pickle_path)
        print("✓ Graph saved!")

        # Run tests
        print("\n" + "=" * 80)
        print("TESTING KNOWLEDGE GRAPH")
        print("=" * 80)

        # Test 1: Statistics
        print("\n### Graph Statistics ###")
        stats = kg.get_statistics()
        print(f"  Nodes: {stats['num_nodes']:,}")
        print(f"  Edges: {stats['num_edges']:,}")
        print(f"  Density: {stats['density']:.6f}")

        # Test 2: Node types
        print("\n### Node Types (sample) ###")
        node_types = {}
        for _, data in list(kg.graph.nodes(data=True))[:5000]:
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {node_type}: {count}")

        # Test 3: Relationship types
        print("\n### Relationship Types (sample) ###")
        rel_types = {}
        for _, _, data in list(kg.graph.edges(data=True))[:5000]:
            rel_type = data.get('relation', 'unknown')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

        for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rel_type}: {count}")

        # Test 4: Drug search
        print("\n### Drug Search Test ###")
        test_drugs = ["Metformin", "Aspirin", "Ibuprofen"]
        for drug_name in test_drugs:
            matches = kg.search_drug_by_name(drug_name)
            if matches:
                print(f"  '{drug_name}' found: {matches[0]}")
                neighbors = kg.get_neighbors(matches[0])
                print(f"    → Connected to {len(neighbors)} entities")
                break
        else:
            print("  Note: Drug names in Hetionet use specific identifiers")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function"""
    print("=" * 80)
    print("HETIONET KNOWLEDGE GRAPH SETUP")
    print("=" * 80)
    print("\nHetionet is a comprehensive biomedical knowledge graph:")
    print("  • 47,031 nodes (11 types)")
    print("  • 2,250,197 relationships (24 types)")
    print("  • Includes: Drugs, Diseases, Genes, Proteins, Side Effects, etc.")
    print("  • 100% FREE - No account or registration needed!")
    print("  • Published research-grade data")
    print()

    # Step 1: Download
    json_file = download_hetionet()
    if not json_file:
        print("\n❌ Setup failed at download stage")
        sys.exit(1)

    # Step 2: Load and test
    success = load_and_test_hetionet(json_file)

    if success:
        # Success message
        print("\n" + "=" * 80)
        print("✓ SETUP COMPLETE!")
        print("=" * 80)
        print("\nYou can now use Hetionet in your code:")
        print()
        print("  Option 1: Load from pickle (FAST - recommended):")
        print("    from knowledge_sources.knowledge_graph import KnowledgeGraphSource")
        print("    kg = KnowledgeGraphSource()")
        print("    kg.load_graph('data/hetionet/hetionet_graph.pkl')")
        print()
        print("  Option 2: Load from JSON:")
        print("    kg.load_hetionet('data/hetionet/hetionet-v1.0.json')")
        print()
        print("Next steps:")
        print("  - Run: python knowledge_sources/knowledge_graph.py")
        print("  - Or: python examples/integrate_all_sources.py")
        print("=" * 80)
    else:
        print("\n❌ Setup failed at loading stage")
        sys.exit(1)


if __name__ == "__main__":
    main()
