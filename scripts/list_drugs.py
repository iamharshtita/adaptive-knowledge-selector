"""
List all drugs (compounds) in the knowledge graph
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource


def list_all_drugs(kg: KnowledgeGraphSource, limit: int = None, search: str = None):
    """
    List all drugs in the knowledge graph

    Args:
        kg: Knowledge graph source
        limit: Maximum number of drugs to show (None for all)
        search: Filter drugs by name (case-insensitive)
    """
    drugs = []

    # Collect all compounds (drugs)
    for node_id, data in kg.graph.nodes(data=True):
        if data.get('type') == 'Compound':
            drug_name = data.get('name', '')
            drug_identifier = data.get('identifier', '')

            # Apply search filter
            if search and search.lower() not in drug_name.lower():
                continue

            drugs.append({
                'id': node_id,
                'name': drug_name,
                'identifier': drug_identifier
            })

    # Sort alphabetically
    drugs.sort(key=lambda x: x['name'])

    # Print results
    print("\n" + "=" * 80)
    if search:
        print(f"DRUGS MATCHING: '{search}'")
    else:
        print("ALL DRUGS IN KNOWLEDGE GRAPH")
    print("=" * 80)

    total = len(drugs)
    display_count = min(limit, total) if limit else total

    print(f"\nShowing {display_count:,} of {total:,} drugs\n")

    for i, drug in enumerate(drugs[:display_count], 1):
        print(f"{i:4d}. {drug['name']:50s} | ID: {drug['identifier']}")

    if limit and total > limit:
        print(f"\n... and {total - limit:,} more drugs")
        print(f"\nTo see all drugs, run: python scripts/list_drugs.py --all")


def list_common_drugs(kg: KnowledgeGraphSource):
    """List commonly known drugs"""
    common_drug_names = [
        "Aspirin", "Metformin", "Ibuprofen", "Acetaminophen", "Warfarin",
        "Atorvastatin", "Simvastatin", "Lisinopril", "Levothyroxine", "Omeprazole",
        "Amlodipine", "Metoprolol", "Losartan", "Albuterol", "Gabapentin",
        "Hydrochlorothiazide", "Sertraline", "Fluoxetine", "Prednisone", "Amoxicillin",
        "Ciprofloxacin", "Azithromycin", "Doxycycline", "Cephalexin", "Morphine",
        "Codeine", "Fentanyl", "Oxycodone", "Hydrocodone", "Tramadol",
        "Insulin", "Glipizide", "Glyburide", "Pioglitazone", "Sitagliptin"
    ]

    print("\n" + "=" * 80)
    print("COMMON DRUGS AVAILABLE IN KNOWLEDGE GRAPH")
    print("=" * 80)
    print()

    found = []
    not_found = []

    for drug_name in common_drug_names:
        matches = kg.search_drug_by_name(drug_name)
        if matches:
            drug_data = kg.graph.nodes[matches[0]]
            found.append({
                'name': drug_data.get('name', drug_name),
                'id': matches[0],
                'identifier': drug_data.get('identifier', '')
            })
        else:
            not_found.append(drug_name)

    # Print found drugs
    print(f"✓ Found {len(found)} common drugs:\n")
    for i, drug in enumerate(found, 1):
        print(f"  {i:2d}. {drug['name']:30s} → {drug['identifier']}")

    # Print not found
    if not_found:
        print(f"\n✗ Not found ({len(not_found)}):")
        print(f"  {', '.join(not_found)}")

    print(f"\n\nTo search for a specific drug:")
    print(f"  python scripts/list_drugs.py --search <drug_name>")


def list_by_category(kg: KnowledgeGraphSource):
    """List drugs grouped by pharmacologic class"""
    print("\n" + "=" * 80)
    print("DRUGS BY PHARMACOLOGIC CLASS")
    print("=" * 80)

    # Find all pharmacologic classes
    classes = {}

    for node_id, data in kg.graph.nodes(data=True):
        if data.get('type') == 'Pharmacologic Class':
            class_name = data.get('name', '')
            classes[node_id] = {
                'name': class_name,
                'drugs': []
            }

    # Find drugs in each class
    for source, target, edge_data in kg.graph.edges(data=True):
        if edge_data.get('relation') == 'includes':
            if source in classes:
                drug_data = kg.graph.nodes.get(target)
                if drug_data and drug_data.get('type') == 'Compound':
                    classes[source]['drugs'].append(drug_data.get('name', ''))

    # Sort and display
    sorted_classes = sorted(classes.items(), key=lambda x: len(x[1]['drugs']), reverse=True)

    print(f"\nFound {len(sorted_classes)} drug classes\n")

    for i, (class_id, class_data) in enumerate(sorted_classes[:20], 1):  # Show top 20
        drug_count = len(class_data['drugs'])
        if drug_count > 0:
            print(f"\n{i:2d}. {class_data['name']} ({drug_count} drugs)")
            for drug in sorted(class_data['drugs'])[:5]:  # Show first 5 drugs
                print(f"     - {drug}")
            if drug_count > 5:
                print(f"     ... and {drug_count - 5} more")


def main():
    """Main function"""
    print("=" * 80)
    print("DRUG NAME FINDER")
    print("=" * 80)

    # Load knowledge graph
    pickle_path = "data/hetionet/hetionet_graph.pkl"

    if not os.path.exists(pickle_path):
        print("\n❌ Knowledge graph not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print("\nLoading knowledge graph...")
    kg = KnowledgeGraphSource()
    kg.load_graph(pickle_path)

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            list_all_drugs(kg)
        elif sys.argv[1] == '--search' and len(sys.argv) > 2:
            search_term = ' '.join(sys.argv[2:])
            list_all_drugs(kg, search=search_term)
        elif sys.argv[1] == '--common':
            list_common_drugs(kg)
        elif sys.argv[1] == '--categories':
            list_by_category(kg)
        elif sys.argv[1] == '--limit' and len(sys.argv) > 2:
            try:
                limit = int(sys.argv[2])
                list_all_drugs(kg, limit=limit)
            except ValueError:
                print("Error: limit must be a number")
        else:
            print(f"\nUnknown option: {sys.argv[1]}")
            print("\nUsage:")
            print("  python scripts/list_drugs.py                    # Show first 50 drugs")
            print("  python scripts/list_drugs.py --all              # Show all drugs")
            print("  python scripts/list_drugs.py --limit 100        # Show first 100 drugs")
            print("  python scripts/list_drugs.py --search aspirin   # Search for specific drug")
            print("  python scripts/list_drugs.py --common           # Show common drugs")
            print("  python scripts/list_drugs.py --categories       # Group by drug class")
    else:
        # Default: show first 50 and common drugs
        list_all_drugs(kg, limit=50)
        list_common_drugs(kg)


if __name__ == "__main__":
    main()
