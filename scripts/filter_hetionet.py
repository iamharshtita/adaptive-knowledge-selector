"""
Filter Hetionet to keep only specific relationships.
This creates a smaller, focused knowledge graph.

Keeps only:
- treats: Compound → Disease
- causes: Compound → Side Effect
- presents: Disease → Symptom
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def main():
    print("=" * 80)
    print("HETIONET RELATIONSHIP FILTER")
    print("=" * 80)

    # Define relationships to keep
    ALLOWED_RELATIONSHIPS = ['treats', 'causes', 'presents']

    print("\nThis script will filter Hetionet to keep only:")
    for rel in ALLOWED_RELATIONSHIPS:
        print(f"  ✓ {rel}")

    print("\nAll other relationships will be removed.")
    print("Isolated nodes (nodes with no edges) will also be removed.")

    # Paths
    pickle_path = "data/hetionet/hetionet_graph.pkl"
    json_path = "data/hetionet/hetionet-v1.0.json"
    output_path = "data/hetionet/hetionet_filtered.pkl"

    # Check if source exists
    if not os.path.exists(pickle_path) and not os.path.exists(json_path):
        print("\n❌ Error: Hetionet not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    # Load graph
    print("\n" + "=" * 80)
    print("STEP 1: Loading Hetionet")
    print("=" * 80)

    kg = KnowledgeGraphSource()

    if os.path.exists(pickle_path):
        print(f"Loading from pickle: {pickle_path}")
        kg.load_graph(pickle_path)
    else:
        print(f"Loading from JSON: {json_path}")
        kg.load_hetionet(json_path)

    # Show initial stats
    stats = kg.get_statistics()
    print(f"\nInitial Graph:")
    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges: {stats['num_edges']:,}")

    # Filter
    print("\n" + "=" * 80)
    print("STEP 2: Filtering Relationships")
    print("=" * 80)

    filter_stats = kg.filter_by_relationships(
        allowed_relations=ALLOWED_RELATIONSHIPS,
        remove_isolated_nodes=True
    )

    # Save filtered graph
    print("\n" + "=" * 80)
    print("STEP 3: Saving Filtered Graph")
    print("=" * 80)

    print(f"\nSaving to: {output_path}")
    kg.save_graph(output_path)

    # Summary
    print("\n" + "=" * 80)
    print("✓ FILTERING COMPLETE!")
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"  Original nodes: {filter_stats['initial_nodes']:,}")
    print(f"  Filtered nodes: {filter_stats['final_nodes']:,}")
    print(f"  Nodes removed: {filter_stats['nodes_removed']:,}")
    print()
    print(f"  Original edges: {filter_stats['initial_edges']:,}")
    print(f"  Filtered edges: {filter_stats['final_edges']:,}")
    print(f"  Edges removed: {filter_stats['edges_removed']:,}")

    print("\n📂 Files:")
    print(f"  Original: {pickle_path}")
    print(f"  Filtered: {output_path}")

    print("\n🚀 Next Steps:")
    print("\n  1. Use the filtered graph in your code:")
    print("     from knowledge_sources.knowledge_graph import KnowledgeGraphSource")
    print("     kg = KnowledgeGraphSource()")
    print(f"     kg.load_graph('{output_path}')")

    print("\n  2. Update examples to use filtered graph:")
    print("     # In examples/integrate_all_sources.py, change:")
    print(f"     # pickle_path = '{output_path}'")

    print("\n  3. Create new visualizations:")
    print("     python scripts/create_viz.py metformin")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
