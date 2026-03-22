"""
Graph Visualization Tools
Create visual representations of the knowledge graph
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_drug_neighborhood(kg: KnowledgeGraphSource, drug_name: str,
                                depth: int = 1, max_neighbors: int = 20,
                                output_file: str = None):
    """
    Visualize a drug and its immediate neighborhood

    Args:
        kg: Knowledge graph
        drug_name: Drug name to visualize
        depth: How many hops from the drug (1 or 2)
        max_neighbors: Maximum neighbors to show
        output_file: Where to save the image (None = display only)
    """
    print(f"\nVisualizing neighborhood of: {drug_name}")

    # Find drug
    matches = kg.search_drug_by_name(drug_name)
    if not matches:
        print(f"❌ Drug '{drug_name}' not found")
        return

    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]
    print(f"✓ Found: {drug_data['name']} ({drug_id})")

    # Get neighbors
    neighbors = list(kg.get_neighbors(drug_id))[:max_neighbors]
    print(f"✓ Found {len(neighbors)} neighbors (showing {min(len(neighbors), max_neighbors)})")

    # Create subgraph
    nodes_to_include = [drug_id] + neighbors
    subgraph = kg.graph.subgraph(nodes_to_include)

    print(f"✓ Creating subgraph with {len(nodes_to_include)} nodes")

    # Create visualization
    plt.figure(figsize=(20, 20))

    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Node colors by type
    node_colors = []
    node_types = set()
    type_colors = {
        'Compound': '#FF6B6B',      # Red
        'Disease': '#4ECDC4',        # Teal
        'Gene': '#95E1D3',           # Light green
        'Side Effect': '#FFD93D',    # Yellow
        'Pathway': '#A8E6CF',        # Mint
        'Symptom': '#FFB6B9',        # Pink
        'Anatomy': '#C7CEEA',        # Lavender
        'Biological Process': '#B4E7CE',  # Aqua
        'Molecular Function': '#FFE5B4',  # Peach
        'Cellular Component': '#E0BBE4',  # Purple
        'Pharmacologic Class': '#FEC8D8'  # Light pink
    }

    for node in subgraph.nodes():
        node_type = kg.graph.nodes[node].get('type', 'unknown')
        node_types.add(node_type)
        node_colors.append(type_colors.get(node_type, '#CCCCCC'))

    # Node sizes - drug is bigger
    node_sizes = [3000 if node == drug_id else 800 for node in subgraph.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )

    # Draw edges with colors by relationship
    edge_colors = []
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'unknown')
        if relation == 'treats':
            edge_colors.append('#2E8B57')  # Green
        elif relation == 'causes':
            edge_colors.append('#DC143C')  # Red
        elif relation == 'binds':
            edge_colors.append('#4169E1')  # Blue
        elif 'regulates' in relation:
            edge_colors.append('#FF8C00')  # Orange
        else:
            edge_colors.append('#808080')  # Gray

    nx.draw_networkx_edges(
        subgraph, pos,
        edge_color=edge_colors,
        width=2,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )

    # Labels - show node names
    labels = {}
    for node in subgraph.nodes():
        name = kg.graph.nodes[node].get('name', node)
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + '...'
        labels[node] = name

    nx.draw_networkx_labels(
        subgraph, pos,
        labels,
        font_size=8,
        font_weight='bold'
    )

    # Create legend for node types
    legend_elements = []
    for node_type in sorted(node_types):
        color = type_colors.get(node_type, '#CCCCCC')
        legend_elements.append(mpatches.Patch(color=color, label=node_type))

    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title
    plt.title(f"Knowledge Graph: {drug_data['name']} and Neighbors\n"
             f"Showing {len(neighbors)} connections",
             fontsize=16, fontweight='bold', pad=20)

    plt.axis('off')
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    else:
        print("✓ Displaying graph (close window to continue)")
        plt.show()

    plt.close()


def visualize_relationship_network(kg: KnowledgeGraphSource, relation_type: str,
                                   max_nodes: int = 50, output_file: str = None):
    """
    Visualize a specific type of relationship

    Args:
        kg: Knowledge graph
        relation_type: Type of relationship (e.g., 'treats', 'causes')
        max_nodes: Maximum nodes to include
        output_file: Where to save
    """
    print(f"\nVisualizing '{relation_type}' relationships")

    # Collect edges of this type
    edges_to_include = []
    nodes_to_include = set()

    for u, v, data in kg.graph.edges(data=True):
        if data.get('relation') == relation_type:
            edges_to_include.append((u, v))
            nodes_to_include.add(u)
            nodes_to_include.add(v)

            if len(nodes_to_include) >= max_nodes:
                break

    print(f"✓ Found {len(edges_to_include)} '{relation_type}' edges")
    print(f"✓ Involving {len(nodes_to_include)} nodes")

    if not edges_to_include:
        print(f"❌ No '{relation_type}' relationships found")
        return

    # Create subgraph
    subgraph = kg.graph.edge_subgraph(edges_to_include[:max_nodes])

    # Visualize
    plt.figure(figsize=(16, 16))

    pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)

    # Colors by node type
    node_colors = []
    for node in subgraph.nodes():
        node_type = kg.graph.nodes[node].get('type', 'unknown')
        if node_type == 'Compound':
            node_colors.append('#FF6B6B')
        elif node_type == 'Disease':
            node_colors.append('#4ECDC4')
        elif node_type == 'Side Effect':
            node_colors.append('#FFD93D')
        else:
            node_colors.append('#95E1D3')

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=600,
        alpha=0.8
    )

    nx.draw_networkx_edges(
        subgraph, pos,
        edge_color='#666666',
        width=1.5,
        alpha=0.5,
        arrows=True,
        arrowsize=15
    )

    # Labels
    labels = {}
    for node in subgraph.nodes():
        name = kg.graph.nodes[node].get('name', node)
        if len(name) > 20:
            name = name[:17] + '...'
        labels[node] = name

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=7)

    plt.title(f"'{relation_type}' Relationships\n{len(edges_to_include)} connections",
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def create_interactive_html(kg: KnowledgeGraphSource, drug_name: str,
                            max_neighbors: int = 30, output_file: str = "graph.html"):
    """
    Create interactive HTML visualization using pyvis

    Args:
        kg: Knowledge graph
        drug_name: Drug to visualize
        max_neighbors: Max neighbors
        output_file: HTML file to create
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("❌ pyvis not installed. Install with: pip install pyvis")
        return

    print(f"\nCreating interactive HTML visualization for: {drug_name}")

    # Find drug
    matches = kg.search_drug_by_name(drug_name)
    if not matches:
        print(f"❌ Drug '{drug_name}' not found")
        return

    drug_id = matches[0]
    drug_data = kg.graph.nodes[drug_id]

    # Get neighborhood
    neighbors = list(kg.get_neighbors(drug_id))[:max_neighbors]
    nodes_to_include = [drug_id] + neighbors
    subgraph = kg.graph.subgraph(nodes_to_include)

    # Create pyvis network
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    # Add nodes
    for node in subgraph.nodes():
        node_data = kg.graph.nodes[node]
        name = node_data.get('name', node)
        node_type = node_data.get('type', 'unknown')

        # Color by type
        color_map = {
            'Compound': '#FF6B6B',
            'Disease': '#4ECDC4',
            'Gene': '#95E1D3',
            'Side Effect': '#FFD93D'
        }
        color = color_map.get(node_type, '#CCCCCC')

        # Size - drug is bigger
        size = 40 if node == drug_id else 20

        net.add_node(
            node,
            label=name,
            title=f"{node_type}: {name}",
            color=color,
            size=size
        )

    # Add edges
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'unknown')
        net.add_edge(u, v, title=relation, arrows='to')

    # Save
    net.show(output_file)
    print(f"✓ Interactive visualization saved to: {output_file}")
    print(f"✓ Open in browser: file://{os.path.abspath(output_file)}")


def main():
    """Main visualization function"""
    print("=" * 80)
    print("KNOWLEDGE GRAPH VISUALIZER")
    print("=" * 80)

    # Load graph
    pickle_path = "data/hetionet/hetionet_graph.pkl"
    if not os.path.exists(pickle_path):
        print("\n❌ Knowledge graph not found!")
        print("Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print("\nLoading knowledge graph...")
    kg = KnowledgeGraphSource()
    kg.load_graph(pickle_path)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python scripts/visualize_graph.py drug <drug_name>")
        print("  python scripts/visualize_graph.py relation <relation_type>")
        print("  python scripts/visualize_graph.py interactive <drug_name>")
        print("\nExamples:")
        print("  python scripts/visualize_graph.py drug metformin")
        print("  python scripts/visualize_graph.py relation treats")
        print("  python scripts/visualize_graph.py interactive metformin")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'drug' and len(sys.argv) >= 3:
        drug_name = ' '.join(sys.argv[2:])
        output = f"visualizations/{drug_name.replace(' ', '_')}_graph.png"
        os.makedirs('visualizations', exist_ok=True)
        visualize_drug_neighborhood(kg, drug_name, output_file=output)

    elif command == 'relation' and len(sys.argv) >= 3:
        relation = sys.argv[2]
        output = f"visualizations/{relation}_network.png"
        os.makedirs('visualizations', exist_ok=True)
        visualize_relationship_network(kg, relation, output_file=output)

    elif command == 'interactive' and len(sys.argv) >= 3:
        drug_name = ' '.join(sys.argv[2:])
        output = f"visualizations/{drug_name.replace(' ', '_')}_interactive.html"
        os.makedirs('visualizations', exist_ok=True)
        create_interactive_html(kg, drug_name, output_file=output)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
