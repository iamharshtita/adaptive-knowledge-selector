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
        Intelligent unified query interface — detects intent from the query text
        and routes to the appropriate method for relevant results.

        Args:
            text: Free-text query

        Returns:
            Dict with 'answer' (str summary) and 'results' (raw data)
        """
        import re

        text_lower = text.lower()

        # 1. Detect query intent
        intent = "general"
        if any(keyword in text_lower for keyword in ["interact", "interaction", "combine", "together with"]):
            intent = "interaction"
        elif any(keyword in text_lower for keyword in ["treat", "cure", "therapy", "medication for", "drug for"]):
            intent = "treatment"
        elif any(keyword in text_lower for keyword in ["target", "bind", "mechanism", "acts on"]):
            intent = "target"
        elif any(keyword in text_lower for keyword in ["side effect", "adverse", "reaction", "causes"]):
            intent = "side_effects"

        # 2. Extract drug or disease name from text
        entity_name = None
        for word in re.findall(r'\b[A-Za-z]{3,}\b', text):
            if word.lower() in self._DRUG_ALIASES or self.search_drug_by_name(word):
                entity_name = word
                break

        if not entity_name:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Could not identify a drug name in: '{text}'",
                "results": [],
            }

        # 3. Route to appropriate method based on intent
        try:
            if intent == "interaction":
                return self._query_interactions(entity_name)
            elif intent == "treatment":
                return self._query_treatments(entity_name)
            elif intent == "target":
                return self._query_targets(entity_name)
            elif intent == "side_effects":
                return self._query_side_effects(entity_name)
            else:
                return self._query_general(entity_name)
        except Exception as e:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Error processing query: {e}",
                "results": [],
            }

    def _query_interactions(self, drug_name: str) -> Dict[str, Any]:
        """Query for drug-drug interactions."""
        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]

        # Get only drug-drug interactions
        interactions = []
        for source, target, key, data in self.graph.out_edges(node_id, data=True, keys=True):
            relation = data.get('relation', '')
            target_type = self.graph.nodes.get(target, {}).get('type', '')
            if target_type == 'Compound':
                interactions.append({
                    'drug': target,
                    'name': self.graph.nodes.get(target, {}).get('name', target),
                    'relation': relation
                })

        for source, target, key, data in self.graph.in_edges(node_id, data=True, keys=True):
            relation = data.get('relation', '')
            source_type = self.graph.nodes.get(source, {}).get('type', '')
            if source_type == 'Compound':
                interactions.append({
                    'drug': source,
                    'name': self.graph.nodes.get(source, {}).get('name', source),
                    'relation': relation
                })

        if not interactions:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"No drug interactions found for '{drug_name}' in the knowledge graph.",
                "results": [],
            }

        # Format results
        lines = [
            f"Drug Interactions for: {drug_name}",
            f"Found {len(interactions)} interacting drugs:",
            ""
        ]
        for i, inter in enumerate(interactions[:20], 1):
            lines.append(f"  {i}. {inter['name']}")

        if len(interactions) > 20:
            lines.append(f"  ... and {len(interactions) - 20} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": interactions,
        }

    def _query_treatments(self, drug_name: str) -> Dict[str, Any]:
        """Query for diseases that a drug treats."""
        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]

        # Get diseases this drug treats
        treatments = []
        for _, target, _, data in self.graph.out_edges(node_id, data=True, keys=True):
            relation = data.get('relation', '')
            if relation in ['treats', 'palliates']:
                target_type = self.graph.nodes.get(target, {}).get('type', '')
                if target_type == 'Disease':
                    treatments.append({
                        'disease': target,
                        'name': self.graph.nodes.get(target, {}).get('name', target),
                        'relation': relation
                    })

        if not treatments:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"No disease treatments found for '{drug_name}' in the knowledge graph.",
                "results": [],
            }

        # Format results
        lines = [
            f"Diseases treated by: {drug_name}",
            f"Found {len(treatments)} diseases:",
            ""
        ]
        for i, treat in enumerate(treatments[:15], 1):
            lines.append(f"  {i}. {treat['name']} ({treat['relation']})")

        if len(treatments) > 15:
            lines.append(f"  ... and {len(treatments) - 15} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": treatments,
        }

    def _query_targets(self, drug_name: str) -> Dict[str, Any]:
        """Query for drug targets (genes/proteins)."""
        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]

        # Get genes/proteins this drug targets
        targets = []
        for _, target, _, data in self.graph.out_edges(node_id, data=True, keys=True):
            relation = data.get('relation', '')
            if relation in ['binds', 'downregulates', 'upregulates']:
                target_type = self.graph.nodes.get(target, {}).get('type', '')
                if target_type == 'Gene':
                    targets.append({
                        'target': target,
                        'name': self.graph.nodes.get(target, {}).get('name', target),
                        'relation': relation
                    })

        if not targets:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"No gene/protein targets found for '{drug_name}' in the knowledge graph.",
                "results": [],
            }

        # Format results
        lines = [
            f"Gene/Protein targets for: {drug_name}",
            f"Found {len(targets)} targets:",
            ""
        ]
        for i, tgt in enumerate(targets[:15], 1):
            lines.append(f"  {i}. {tgt['name']} ({tgt['relation']})")

        if len(targets) > 15:
            lines.append(f"  ... and {len(targets) - 15} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": targets,
        }

    def _query_side_effects(self, drug_name: str) -> Dict[str, Any]:
        """Query for side effects of a drug."""
        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]

        # Get side effects
        side_effects = []
        for _, target, _, data in self.graph.out_edges(node_id, data=True, keys=True):
            relation = data.get('relation', '')
            if relation == 'causes':
                target_type = self.graph.nodes.get(target, {}).get('type', '')
                if target_type == 'Side Effect':
                    side_effects.append({
                        'effect': target,
                        'name': self.graph.nodes.get(target, {}).get('name', target)
                    })

        if not side_effects:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"No side effects found for '{drug_name}' in the knowledge graph.",
                "results": [],
            }

        # Format results
        lines = [
            f"Side effects for: {drug_name}",
            f"Found {len(side_effects)} side effects:",
            ""
        ]
        for i, eff in enumerate(side_effects[:20], 1):
            lines.append(f"  {i}. {eff['name']}")

        if len(side_effects) > 20:
            lines.append(f"  ... and {len(side_effects) - 20} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": side_effects,
        }

    def _query_general(self, drug_name: str) -> Dict[str, Any]:
        """General query - return summary of all relationships."""
        matches = self.search_drug_by_name(drug_name)
        if not matches:
            return {
                "source": "KnowledgeGraphSource",
                "answer": f"Drug '{drug_name}' not found in Hetionet graph.",
                "results": [],
            }

        node_id = matches[0]

        # Categorize neighbors by type
        by_type = {}
        for neighbor in self.get_neighbors(node_id):
            ntype = self.graph.nodes.get(neighbor, {}).get('type', 'Unknown')
            if ntype not in by_type:
                by_type[ntype] = []
            by_type[ntype].append({
                'id': neighbor,
                'name': self.graph.nodes.get(neighbor, {}).get('name', neighbor)
            })

        # Format results
        lines = [
            f"Overview for: {drug_name}",
            f"Total connected entities: {sum(len(v) for v in by_type.values())}",
            ""
        ]

        for entity_type, entities in sorted(by_type.items()):
            lines.append(f"  {entity_type}: {len(entities)}")
            for entity in entities[:3]:
                lines.append(f"    - {entity['name']}")
            if len(entities) > 3:
                lines.append(f"    ... and {len(entities) - 3} more")

        return {
            "source": "KnowledgeGraphSource",
            "answer": "\n".join(lines),
            "results": by_type,
        }


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  KNOWLEDGE GRAPH SOURCE - Standalone Test")
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
        print("Error: Hetionet data not found. Run: python scripts/setup_hetionet.py")
        sys.exit(1)

    print(f"Graph: {kg.graph.number_of_nodes():,} nodes, {kg.graph.number_of_edges():,} edges\n")

    queries = [
        "What drugs interact with aspirin?",
        "Show relationships for metformin",
        "Tell me about ibuprofen side effects",
    ]

    for q in queries:
        print(f"\n{'-' * 60}")
        print(f"Query: {q}")
        print(f"{'-' * 60}")
        out = kg.query(q)
        print(out["answer"])

    print(f"\n{'=' * 70}")
    print("Done")
    print(f"{'=' * 70}")

