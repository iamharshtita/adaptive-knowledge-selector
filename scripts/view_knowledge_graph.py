"""
Knowledge Graph Viewer
Interactive viewer and visualizer for Hetionet knowledge graph
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource
import networkx as nx


def print_statistics(kg: KnowledgeGraphSource):
    """Print comprehensive graph statistics"""
    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("=" * 80)

    stats = kg.get_statistics()
    print(f"\nTotal Nodes: {stats['num_nodes']:,}")
    print(f"Total Edges: {stats['num_edges']:,}")
    print(f"Graph Density: {stats['density']:.6f}")
    print(f"Connected: {stats['is_connected']}")


def print_node_types(kg: KnowledgeGraphSource):
    """Print all node types and their counts"""
    print("\n" + "=" * 80)
    print("NODE TYPES")
    print("=" * 80)

    node_types = {}
    for node_id, data in kg.graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print(f"\nFound {len(node_types)} different node types:\n")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type:30s} : {count:6,} nodes")


def print_edge_types(kg: KnowledgeGraphSource):
    """Print all relationship types and their counts"""
    print("\n" + "=" * 80)
    print("RELATIONSHIP TYPES")
    print("=" * 80)

    edge_types = {}
    for _, _, data in kg.graph.edges(data=True):
        edge_type = data.get('relation', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\nFound {len(edge_types)} different relationship types:\n")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type:30s} : {count:8,} relationships")


def search_and_explore(kg: KnowledgeGraphSource, search_term: str):
    """Search for a drug and explore its connections"""
    print("\n" + "=" * 80)
    print(f"SEARCHING FOR: '{search_term}'")
    print("=" * 80)

    # Search drugs
    matches = kg.search_drug_by_name(search_term)

    if not matches:
        print(f"\n❌ No drugs found matching '{search_term}'")
        return

    print(f"\n✓ Found {len(matches)} match(es):\n")

    for i, drug_id in enumerate(matches[:5], 1):  # Show first 5
        data = kg.graph.nodes[drug_id]
        print(f"{i}. {drug_id}")
        print(f"   Name: {data.get('name', 'N/A')}")
        print(f"   Type: {data.get('type', 'N/A')}")

        # Get connections
        neighbors = kg.get_neighbors(drug_id)
        print(f"   Connected to: {len(neighbors)} entities")

        # Get relationship types
        rel_types = {}
        for _, target, data in kg.graph.out_edges(drug_id, data=True):
            rel = data.get('relation', 'unknown')
            rel_types[rel] = rel_types.get(rel, 0) + 1

        if rel_types:
            print(f"   Relationships:")
            for rel, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     - {rel}: {count}")

        print()


def explore_node(kg: KnowledgeGraphSource, node_id: str, depth: int = 1):
    """Explore a specific node and its immediate neighbors"""
    print("\n" + "=" * 80)
    print(f"EXPLORING NODE: {node_id}")
    print("=" * 80)

    if node_id not in kg.graph:
        print(f"\n❌ Node '{node_id}' not found in graph")
        return

    # Node info
    data = kg.graph.nodes[node_id]
    print(f"\nName: {data.get('name', 'N/A')}")
    print(f"Type: {data.get('type', 'N/A')}")
    print(f"Identifier: {data.get('identifier', 'N/A')}")

    # Outgoing edges
    print(f"\n### Outgoing Relationships ###")
    out_edges = list(kg.graph.out_edges(node_id, data=True))
    print(f"Total: {len(out_edges)}")

    # Group by relationship type
    by_relation = {}
    for _, target, edge_data in out_edges:
        relation = edge_data.get('relation', 'unknown')
        if relation not in by_relation:
            by_relation[relation] = []
        by_relation[relation].append((target, kg.graph.nodes[target]))

    for relation, targets in sorted(by_relation.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"\n  {relation} ({len(targets)} connections):")
        for target_id, target_data in targets[:3]:  # Show first 3
            print(f"    → {target_data.get('name', target_id)[:60]}")

    # Incoming edges
    print(f"\n### Incoming Relationships ###")
    in_edges = list(kg.graph.in_edges(node_id, data=True))
    print(f"Total: {len(in_edges)}")

    by_relation = {}
    for source, _, edge_data in in_edges:
        relation = edge_data.get('relation', 'unknown')
        if relation not in by_relation:
            by_relation[relation] = []
        by_relation[relation].append((source, kg.graph.nodes[source]))

    for relation, sources in sorted(by_relation.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"\n  {relation} ({len(sources)} connections):")
        for source_id, source_data in sources[:3]:  # Show first 3
            print(f"    ← {source_data.get('name', source_id)[:60]}")


def find_paths_between(kg: KnowledgeGraphSource, source: str, target: str):
    """Find paths between two nodes"""
    print("\n" + "=" * 80)
    print(f"FINDING PATHS: {source} → {target}")
    print("=" * 80)

    paths = kg.find_path(source, target, max_length=3)

    if not paths:
        print("\n❌ No paths found within 3 hops")
        return

    print(f"\n✓ Found {len(paths)} path(s):\n")

    for i, path in enumerate(paths, 1):
        print(f"Path {i}:")
        for j in range(len(path)):
            node_data = kg.graph.nodes[path[j]]
            print(f"  {j+1}. {node_data.get('name', path[j])[:60]}")
            if j < len(path) - 1:
                # Get edge relationship
                edge_data = kg.graph.get_edge_data(path[j], path[j+1])
                if edge_data:
                    # MultiDiGraph returns dict of dicts
                    relations = [d.get('relation', 'unknown') for d in edge_data.values()]
                    print(f"      --{relations[0]}-->")
        print()


def interactive_mode(kg: KnowledgeGraphSource):
    """Interactive exploration mode"""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  search <drug name>       - Search for drugs")
    print("  explore <node_id>        - Explore a specific node")
    print("  path <source> <target>   - Find paths between nodes")
    print("  stats                    - Show statistics")
    print("  nodes                    - Show node types")
    print("  edges                    - Show edge types")
    print("  quit                     - Exit")

    while True:
        try:
            cmd = input("\n> ").strip()

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()

            if command == 'quit':
                print("Goodbye!")
                break

            elif command == 'search':
                if len(parts) < 2:
                    print("Usage: search <drug name>")
                else:
                    search_and_explore(kg, parts[1])

            elif command == 'explore':
                if len(parts) < 2:
                    print("Usage: explore <node_id>")
                else:
                    explore_node(kg, parts[1])

            elif command == 'path':
                if len(parts) < 2:
                    print("Usage: path <source_id> <target_id>")
                else:
                    node_ids = parts[1].split()
                    if len(node_ids) != 2:
                        print("Usage: path <source_id> <target_id>")
                    else:
                        find_paths_between(kg, node_ids[0], node_ids[1])

            elif command == 'stats':
                print_statistics(kg)

            elif command == 'nodes':
                print_node_types(kg)

            elif command == 'edges':
                print_edge_types(kg)

            else:
                print(f"Unknown command: {command}")
                print("Type 'quit' to exit or try another command")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main viewer function"""
    print("=" * 80)
    print("HETIONET KNOWLEDGE GRAPH VIEWER")
    print("=" * 80)

    # Load graph
    pickle_path = "data/hetionet/hetionet_graph.pkl"
    json_path = "data/hetionet/hetionet-v1.0.json"

    if not os.path.exists(pickle_path) and not os.path.exists(json_path):
        print("\n❌ Hetionet not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    kg = KnowledgeGraphSource()

    if os.path.exists(pickle_path):
        print(f"\nLoading from pickle: {pickle_path}")
        kg.load_graph(pickle_path)
    else:
        print(f"\nLoading from JSON: {json_path}")
        kg.load_hetionet(json_path)

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'stats':
            print_statistics(kg)
            print_node_types(kg)
            print_edge_types(kg)

        elif command == 'search':
            if len(sys.argv) < 3:
                print("Usage: python view_knowledge_graph.py search <drug_name>")
            else:
                search_and_explore(kg, sys.argv[2])

        elif command == 'explore':
            if len(sys.argv) < 3:
                print("Usage: python view_knowledge_graph.py explore <node_id>")
            else:
                explore_node(kg, sys.argv[2])

        elif command == 'path':
            if len(sys.argv) < 4:
                print("Usage: python view_knowledge_graph.py path <source_id> <target_id>")
            else:
                find_paths_between(kg, sys.argv[2], sys.argv[3])

        elif command == 'interactive' or command == 'i':
            interactive_mode(kg)

        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  stats              - Show full statistics")
            print("  search <name>      - Search for drugs")
            print("  explore <id>       - Explore a node")
            print("  path <id1> <id2>   - Find paths")
            print("  interactive        - Interactive mode")

    else:
        # Default: show overview and enter interactive mode
        print_statistics(kg)
        interactive_mode(kg)


if __name__ == "__main__":
    main()
