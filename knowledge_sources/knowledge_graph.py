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

    # Common drug name aliases → Hetionet formal names
    _DRUG_ALIASES = {
        "aspirin": "acetylsalicylic acid",
        "tylenol": "acetaminophen",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "lipitor": "atorvastatin",
        "zocor": "simvastatin",
        "glucophage": "metformin",
        "coumadin": "warfarin",
        "plavix": "clopidogrel",
        "prilosec": "omeprazole",
        "nexium": "esomeprazole",
        "zoloft": "sertraline",
        "prozac": "fluoxetine",
        "lasix": "furosemide",
        "synthroid": "levothyroxine",
        "xanax": "alprazolam",
        "ambien": "zolpidem",
        "viagra": "sildenafil",
        "cialis": "tadalafil",
        "celebrex": "celecoxib",
        "humira": "adalimumab",
        "eliquis": "apixaban",
        "jardiance": "empagliflozin",
        "ozempic": "semaglutide",
        "trulicity": "dulaglutide",
    }

    def search_drug_by_name(self, drug_name: str) -> List[str]:
        """
        Search for drugs by name (case-insensitive partial match).
        Resolves common/brand names via alias map.

        Args:
            drug_name: Drug name to search for (common or formal)

        Returns:
            List of matching drug node IDs
        """
        matches = []
        drug_name_lower = drug_name.lower().strip()

        # 1. Try alias resolution first
        search_terms = [drug_name_lower]
        if drug_name_lower in self._DRUG_ALIASES:
            search_terms.append(self._DRUG_ALIASES[drug_name_lower])

        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type == 'Compound':
                node_name = data.get('name', '').lower()
                for term in search_terms:
                    if term in node_name or node_name in term:
                        matches.append(node_id)
                        break

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
            matches = self.search_drug_by_name(drug_name)
            if not matches:
                return []
            drug_name = matches[0]

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
            matches = self.search_drug_by_name(drug_name)
            if not matches:
                return []
            drug_name = matches[0]

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

        # Resolve plain disease name to node ID if needed
        if disease_name not in self.graph:
            disease_lower = disease_name.lower()
            for node_id, data in self.graph.nodes(data=True):
                if data.get('type') == 'Disease' and disease_lower in data.get('name', '').lower():
                    disease_name = node_id
                    break
            else:
                return []

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
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
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

    def query(self, text: str) -> Dict[str, Any]:
        """
        Unified query interface — takes free-text, extracts a drug name,
        finds it in the graph, and returns its relationships.

        Args:
            text: Free-text query (should mention a drug)

        Returns:
            Dict with 'answer' (str summary) and 'results' (raw neighbor list)
        """
        import re

        # Try to extract drug name from text
        drug_name = None
        # Pattern: look for known drug-like words
        for word in re.findall(r'\b[A-Za-z]{3,}\b', text):
            if word.lower() in self._DRUG_ALIASES or self.search_drug_by_name(word):
                drug_name = word
                break

        if not drug_name:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Could not identify a drug name in: '{text}'",
                "results": [],
            }

        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]
        neighbors = self.get_neighbors(node_id)

        # Format neighbors into readable lines
        lines = [f"Drug: {drug_name} (node: {node_id})", f"Connected entities: {len(neighbors)}", ""]
        for i, n in enumerate(neighbors[:15], 1):
            data = self.graph.nodes.get(n, {})
            name = data.get("name", n)
            ntype = data.get("type", "Unknown")
            lines.append(f"  {i}. [{ntype}] {name}")
        if len(neighbors) > 15:
            lines.append(f"  ... and {len(neighbors) - 15} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": neighbors,
        }


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  KNOWLEDGE GRAPH SOURCE — Standalone Test")
    print("=" * 70)

    kg = KnowledgeGraphSource()

    # Load graph
    pickle_path = "data/hetionet/hetionet_graph.pkl"
    json_path = "data/hetionet/hetionet-v1.0.json"

    if os.path.exists(pickle_path):
        kg.load_graph(pickle_path)
    elif os.path.exists(json_path):
        kg.load_hetionet(json_path)
    else:
        print("❌ Hetionet data not found. Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print(f"Graph: {kg.graph.number_of_nodes():,} nodes, {kg.graph.number_of_edges():,} edges\n")

    queries = [
        "What drugs interact with aspirin?",
        "Show relationships for metformin",
        "Tell me about ibuprofen side effects",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"🔎 Query: {q}")
        print(f"{'─' * 60}")
        out = kg.query(q)
        print(out["answer"])

    print(f"\n{'=' * 70}")
    print("✅ Done")
    print(f"{'=' * 70}")

