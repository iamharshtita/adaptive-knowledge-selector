"""
Quick Knowledge Graph Inspector
Usage: python scripts/check_kg.py [drug_name]
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def show_overview(kg):
    """Show graph overview"""
    print("=" * 70)
    print("📊 KNOWLEDGE GRAPH OVERVIEW")
    print("=" * 70)

    stats = kg.get_statistics()
    print(f"\n✓ Total Nodes: {stats['num_nodes']:,}")
    print(f"✓ Total Edges: {stats['num_edges']:,}")
    print(f"✓ Graph Density: {stats['density']:.6f}")

    # Node types
    print("\n📦 NODE TYPES:")
    node_types = {}
    for _, data in kg.graph.nodes(data=True):
        nt = data.get('type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1

    for nt, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nt:20s}: {count:6,}")

    # Relationship types
    print("\n🔗 RELATIONSHIPS:")
    rel_types = {}
    for _, _, data in kg.graph.edges(data=True):
        rt = data.get('relation', 'unknown')
        rel_types[rt] = rel_types.get(rt, 0) + 1

    for rt, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rt:20s}: {count:8,}")


def explore_drug(kg, drug_name):
    """Explore a specific drug"""
    print("=" * 70)
    print(f"🔍 EXPLORING: {drug_name.upper()}")
    print("=" * 70)

    matches = kg.search_drug_by_name(drug_name)

    if not matches:
        print(f"\n❌ Drug '{drug_name}' not found.")
        print("\nTry searching for similar names:")
        # Show some suggestions
        all_drugs = []
        for node_id, data in kg.graph.nodes(data=True):
            if data.get('type') == 'Compound':
                all_drugs.append(data.get('name', ''))

        suggestions = [d for d in all_drugs if drug_name.lower()[0] == d.lower()[0]][:5]
        for sugg in suggestions:
            print(f"  - {sugg}")
        return

    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]

    print(f"\n✓ Found: {drug_data['name']}")
    print(f"  ID: {drug_id}")
    print(f"  Type: {drug_data['type']}")

    # Count relationships
    rel_count = {}
    for _, _, data in kg.graph.out_edges(drug_id, data=True):
        rel = data['relation']
        rel_count[rel] = rel_count.get(rel, 0) + 1

    print(f"\n📊 Relationship Summary:")
    for rel, count in rel_count.items():
        print(f"  {rel}: {count}")

    # Show treats
    if 'treats' in rel_count:
        print(f"\n💊 Treats ({rel_count['treats']} diseases):")
        for _, target, data in kg.graph.out_edges(drug_id, data=True):
            if data['relation'] == 'treats':
                disease = kg.graph.nodes[target]['name']
                print(f"  ✓ {disease}")

    # Show side effects (first 15)
    if 'causes' in rel_count:
        print(f"\n⚠️  Side Effects (showing 15 of {rel_count['causes']}):")
        count = 0
        for _, target, data in kg.graph.out_edges(drug_id, data=True):
            if data['relation'] == 'causes' and count < 15:
                side_effect = kg.graph.nodes[target]['name']
                print(f"  • {side_effect}")
                count += 1


def search_drugs(kg, search_term):
    """Search for drugs by name"""
    print("=" * 70)
    print(f"🔍 SEARCHING FOR: '{search_term}'")
    print("=" * 70)

    matches = kg.search_drug_by_name(search_term)

    if not matches:
        print(f"\n❌ No drugs found matching '{search_term}'")
    else:
        print(f"\n✓ Found {len(matches)} match(es):\n")
        for i, drug_id in enumerate(matches[:20], 1):  # Show max 20
            drug_data = kg.graph.nodes[drug_id]
            print(f"  {i}. {drug_data['name']}")


def main():
    print("\n" + "=" * 70)
    print("🔬 KNOWLEDGE GRAPH INSPECTOR")
    print("=" * 70 + "\n")

    # Load graph
    kg = KnowledgeGraphSource()
    kg.load_graph('data/hetionet/hetionet_filtered.pkl')

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command in ['overview', 'stats', 'summary']:
            show_overview(kg)
        elif command == 'search' and len(sys.argv) > 2:
            search_drugs(kg, sys.argv[2])
        else:
            # Assume it's a drug name
            explore_drug(kg, command)
    else:
        # No arguments - show overview
        show_overview(kg)
        print("\n" + "=" * 70)
        print("💡 USAGE TIPS")
        print("=" * 70)
        print("\nExplore a drug:")
        print("  python scripts/check_kg.py metformin")
        print("  python scripts/check_kg.py aspirin")
        print("\nSearch for drugs:")
        print("  python scripts/check_kg.py search statin")
        print("\nShow overview:")
        print("  python scripts/check_kg.py overview")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
