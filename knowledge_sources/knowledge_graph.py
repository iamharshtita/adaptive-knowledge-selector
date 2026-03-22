"""
Knowledge Graph Source - Structured Relationships
Uses Hetionet for drug-drug interactions, drug-disease relationships, and more

Hetionet is a comprehensive biomedical knowledge graph with:
- 47,031 nodes (drugs, diseases, genes, proteins, etc.)
- 2,250,197 relationships
- 100% FREE - No account needed!
"""

import networkx as nx
import requests
import json
import os
from typing import List, Dict, Any, Tuple
import pickle


class KnowledgeGraphSource:
    """
    Manages biomedical knowledge graph for structured queries
    """

    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.MultiDiGraph()
        self.node_types = {}  # Store node types (drug, disease, protein, etc.)

    def load_hetionet(self, json_path: str = None):
        """
        Load Hetionet knowledge graph

        Hetionet is a comprehensive biomedical knowledge graph
        Download from: https://github.com/hetio/hetionet

        Args:
            json_path: Path to hetionet JSON file
        """
        if json_path and os.path.exists(json_path):
            print(f"Loading Hetionet from {json_path}...")
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Add nodes
            # Hetionet structure: node has 'identifier' and 'kind'
            for node in data.get('nodes', []):
                node_kind = node.get('kind', 'unknown')
                node_identifier = node.get('identifier', '')
                # Create unique ID as "Kind::Identifier"
                node_id = f"{node_kind}::{node_identifier}"

                self.graph.add_node(
                    node_id,
                    name=node.get('name', node_identifier),
                    type=node_kind,
                    identifier=node_identifier
                )
                self.node_types[node_id] = node_kind

            # Add edges
            # Hetionet structure: edge has 'source_id' and 'target_id' as [kind, identifier] tuples
            for edge in data.get('edges', []):
                source_kind, source_identifier = edge['source_id']
                target_kind, target_identifier = edge['target_id']

                source_id = f"{source_kind}::{source_identifier}"
                target_id = f"{target_kind}::{target_identifier}"

                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation=edge.get('kind', 'relates_to'),
                    direction=edge.get('direction', 'forward')
                )

            print(f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

        else:
            print("Hetionet file not found. You can download it from:")
            print("https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2")

    def search_drug_by_name(self, drug_name: str) -> List[str]:
        """
        Search for drugs by name (case-insensitive partial match)

        Args:
            drug_name: Drug name to search for

        Returns:
            List of matching drug node IDs
        """
        matches = []
        drug_name_lower = drug_name.lower()

        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type == 'Compound':  # Hetionet uses 'Compound' for drugs
                node_name = data.get('name', '').lower()
                # Check if search term appears in the name
                if drug_name_lower in node_name:
                    matches.append(node_id)

        return matches

    def search_by_identifier(self, node_type: str, identifier: str) -> str:
        """
        Get node ID from type and identifier

        Args:
            node_type: Node type (e.g., 'Compound', 'Disease')
            identifier: Node identifier (e.g., 'DB00331')

        Returns:
            Full node ID or None
        """
        node_id = f"{node_type}::{identifier}"
        if node_id in self.graph:
            return node_id
        return None

    def query_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        Find all drugs that interact with the given drug

        Args:
            drug_name: Name of the drug

        Returns:
            List of interacting drugs with relationship details
        """
        if drug_name not in self.graph:
            return []

        interactions = []

        # Get all edges from this drug
        for source, target, key, data in self.graph.out_edges(drug_name, data=True, keys=True):
            if data.get('relation') == 'interacts_with':
                interactions.append({
                    'drug': target,
                    'severity': data.get('severity', 'unknown'),
                    'description': data.get('description', 'No description available')
                })

        # Get all edges to this drug
        for source, target, key, data in self.graph.in_edges(drug_name, data=True, keys=True):
            if data.get('relation') == 'interacts_with':
                interactions.append({
                    'drug': source,
                    'severity': data.get('severity', 'unknown'),
                    'description': data.get('description', 'No description available')
                })

        return interactions

    def query_drug_targets(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        Find proteins/targets that the drug acts on

        Args:
            drug_name: Name of the drug

        Returns:
            List of targets with mechanism details
        """
        if drug_name not in self.graph:
            return []

        targets = []

        for source, target, key, data in self.graph.out_edges(drug_name, data=True, keys=True):
            if data.get('relation') == 'targets':
                targets.append({
                    'target': target,
                    'mechanism': data.get('mechanism', 'unknown'),
                    'type': self.graph.nodes[target].get('type', 'protein')
                })

        return targets

    def query_disease_treatments(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        Find drugs that treat a specific disease

        Args:
            disease_name: Name of the disease

        Returns:
            List of drugs that treat this disease
        """
        treatments = []

        # Look for incoming edges to the disease
        if disease_name in self.graph:
            for source, target, key, data in self.graph.in_edges(disease_name, data=True, keys=True):
                if data.get('relation') in ['treats', 'prevents']:
                    treatments.append({
                        'drug': source,
                        'relation': data.get('relation'),
                        'indication': data.get('indication', 'No indication available')
                    })

        return treatments

    def find_path(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """
        Find paths between two entities in the graph

        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of nodes)
        """
        if source not in self.graph or target not in self.graph:
            return []

        try:
            paths = []
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=max_length):
                paths.append(path)
                if len(paths) >= 5:  # Limit to 5 paths
                    break
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_neighbors(self, node: str, relation_type: str = None) -> List[str]:
        """
        Get neighboring nodes

        Args:
            node: Node name
            relation_type: Filter by relation type (optional)

        Returns:
            List of neighbor nodes
        """
        if node not in self.graph:
            return []

        neighbors = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(node, data=True):
            if relation_type is None or data.get('relation') == relation_type:
                neighbors.append(target)

        # Incoming edges
        for source, _, data in self.graph.in_edges(node, data=True):
            if relation_type is None or data.get('relation') == relation_type:
                neighbors.append(source)

        return list(set(neighbors))

    def add_custom_relationship(self, source: str, target: str, relation: str, **properties):
        """
        Add a custom relationship to the graph

        Args:
            source: Source entity
            target: Target entity
            relation: Type of relationship
            **properties: Additional properties
        """
        # Add nodes if they don't exist
        if source not in self.graph:
            self.graph.add_node(source, name=source)
        if target not in self.graph:
            self.graph.add_node(target, name=target)

        # Add edge
        self.graph.add_edge(source, target, relation=relation, **properties)

    def save_graph(self, path: str):
        """Save graph to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'node_types': self.node_types
            }, f)
        print(f"Graph saved to {path}")

    def load_graph(self, path: str):
        """Load graph from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.node_types = data.get('node_types', {})
        print(f"Graph loaded from {path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("HETIONET KNOWLEDGE GRAPH DEMO")
    print("=" * 80)

    # Initialize
    kg = KnowledgeGraphSource()

    # Check if Hetionet is available
    hetionet_path = "data/hetionet/hetionet-v1.0.json"
    pickle_path = "data/hetionet/hetionet_graph.pkl"

    if os.path.exists(pickle_path):
        print(f"\nLoading from pickle (fast): {pickle_path}")
        kg.load_graph(pickle_path)
    elif os.path.exists(hetionet_path):
        print(f"\nLoading from JSON: {hetionet_path}")
        kg.load_hetionet(hetionet_path)
    else:
        print("\n❌ Hetionet not found!")
        print("\nTo download Hetionet, run:")
        print("  python scripts/setup_hetionet.py")
        print("\nOr manually download:")
        print("  wget https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2")
        print("  bunzip2 hetionet-v1.0.json.bz2")
        print("  mv hetionet-v1.0.json data/hetionet/")
        sys.exit(1)

    # Statistics
    print("\n=== Graph Statistics ===")
    stats = kg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # Sample node types
    print("\n=== Sample Node Types ===")
    node_types = {}
    for _, data in list(kg.graph.nodes(data=True))[:1000]:
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type}: {count}")

    # Sample edge types
    print("\n=== Sample Relationship Types ===")
    edge_types = {}
    for _, _, data in list(kg.graph.edges(data=True))[:1000]:
        edge_type = data.get('relation', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {edge_type}: {count}")

    # Example: Search for a drug
    print("\n=== Drug Search Example ===")
    search_results = kg.search_drug_by_name("Metformin")
    if search_results:
        print(f"Found {len(search_results)} matches for 'Metformin':")
        for drug_id in search_results[:3]:
            print(f"  - {drug_id}")
            # Get neighbors
            neighbors = kg.get_neighbors(drug_id)
            print(f"    Connected to {len(neighbors)} entities")
    else:
        print("No matches found")

    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("=" * 80)
