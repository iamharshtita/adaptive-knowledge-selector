"""
Unified Integration Example
Demonstrates how to use all knowledge sources together
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.text_knowledge import TextKnowledgeSource
from knowledge_sources.knowledge_graph import KnowledgeGraphSource
from knowledge_sources.tool_api_source import ToolAPISource
from knowledge_sources.llm_source import LLMSource


class UnifiedKnowledgeSystem:
    """
    Unified system that integrates all knowledge sources
    """

    def __init__(self, email: str, ncbi_api_key: str = None, llm_api_key: str = None, use_filtered_kg: bool = True):
        """
        Initialize all knowledge sources

        Args:
            email: Email for NCBI/PubMed
            ncbi_api_key: NCBI API key (optional)
            llm_api_key: API key for LLM (if using commercial model)
            use_filtered_kg: If True, use filtered KG (treats, causes, presents only). Default: True
        """
        print("Initializing Unified Knowledge System...")

        # Initialize all sources
        print("1. Loading Text Knowledge Source...")
        self.text_source = TextKnowledgeSource(email=email, api_key=ncbi_api_key)

        print("2. Loading Knowledge Graph Source...")
        self.kg_source = KnowledgeGraphSource()

        # Choose between filtered or full graph
        if use_filtered_kg:
            pickle_path = "data/hetionet/hetionet_filtered.pkl"
            graph_type = "Filtered (treats, causes, presents)"
        else:
            pickle_path = "data/hetionet/hetionet_graph.pkl"
            graph_type = "Full"

        json_path = "data/hetionet/hetionet-v1.0.json"

        if os.path.exists(pickle_path):
            print(f"   Loading {graph_type} Hetionet from pickle (fast)...")
            self.kg_source.load_graph(pickle_path)
        elif os.path.exists(json_path) and not use_filtered_kg:
            print("   Loading Hetionet from JSON (slower)...")
            self.kg_source.load_hetionet(json_path)
        elif use_filtered_kg:
            print("   ⚠️  Filtered graph not found. Run: python scripts/filter_hetionet.py")
            print("   Continuing without knowledge graph...")
        else:
            print("   ⚠️  Hetionet not found. Run: python scripts/setup_hetionet.py")
            print("   Continuing without knowledge graph...")

        print("3. Loading Tool/API Source...")
        self.tool_source = ToolAPISource()

        print("4. Loading LLM Source...")
        # Change model_type to 'biogpt' for local, 'gpt4' for commercial
        self.llm_source = LLMSource(model_type="gpt4", api_key=llm_api_key)

        print("\nAll sources initialized successfully!\n")

    def query_all_sources(self, query: str, drug_name: str = None):
        """
        Query all knowledge sources for comparison

        Args:
            query: The user query
            drug_name: Drug name to search (extracted from query)
        """
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        # 1. Text Knowledge Source
        print("\n### SOURCE 1: Text Knowledge (PubMed) ###")
        try:
            pubmed_results = self.text_source.search_pubmed(query, max_results=3)
            if pubmed_results:
                print(f"Found {len(pubmed_results)} articles:")
                for i, article in enumerate(pubmed_results, 1):
                    print(f"\n{i}. {article['title']}")
                    print(f"   PMID: {article['pmid']}")
                    print(f"   Abstract: {article['abstract'][:150]}...")
            else:
                print("No PubMed results found")
        except Exception as e:
            print(f"Error querying PubMed: {e}")

        # 2. Knowledge Graph Source
        if drug_name:
            print(f"\n\n### SOURCE 2: Knowledge Graph (Drug Interactions for {drug_name}) ###")
            try:
                interactions = self.kg_source.query_drug_interactions(drug_name)
                if interactions:
                    print(f"Found {len(interactions)} interactions:")
                    for interaction in interactions:
                        print(f"- {interaction['drug']}: {interaction['severity']}")
                        print(f"  {interaction['description']}")
                else:
                    print(f"No interactions found for {drug_name}")
            except Exception as e:
                print(f"Error querying knowledge graph: {e}")

        # 3. Tool/API Source
        if drug_name:
            print(f"\n\n### SOURCE 3: Tool/API (RxNorm Interactions for {drug_name}) ###")
            try:
                api_interactions = self.tool_source.get_drug_interactions(drug_name)
                if api_interactions:
                    print(f"Found {len(api_interactions)} interactions from RxNorm:")
                    for i, interaction in enumerate(api_interactions[:5], 1):
                        print(f"{i}. {interaction['drug']}: {interaction['severity']}")
                        print(f"   {interaction['description'][:100]}...")
                else:
                    print(f"No RxNorm interactions found for {drug_name}")
            except Exception as e:
                print(f"Error querying RxNorm: {e}")

        # 4. LLM Source
        print(f"\n\n### SOURCE 4: LLM (Parametric Knowledge) ###")
        try:
            llm_response = self.llm_source.answer_question(query)
            print(f"LLM Response:\n{llm_response}")
        except Exception as e:
            print(f"Error querying LLM: {e}")

        print("\n" + "=" * 80)

    def demonstrate_source_selection(self):
        """
        Demonstrate which source is best for different query types
        """
        print("\n" + "=" * 80)
        print("DEMONSTRATION: Source Selection Strategy")
        print("=" * 80)

        examples = [
            {
                'query': 'What drugs interact with aspirin?',
                'best_source': 'Knowledge Graph',
                'reason': 'Structured relationships are perfect for drug-drug interactions',
                'drug': 'aspirin'
            },
            {
                'query': 'What is the adult dosage of ibuprofen?',
                'best_source': 'Tool/API',
                'reason': 'Deterministic calculation/lookup from drug database',
                'drug': 'ibuprofen'
            },
            {
                'query': 'Explain how insulin works in the body',
                'best_source': 'LLM',
                'reason': 'Conceptual explanation requiring reasoning',
                'drug': None
            },
            {
                'query': 'Latest research on aspirin benefits',
                'best_source': 'Text Knowledge',
                'reason': 'Recent research papers from PubMed',
                'drug': 'aspirin'
            }
        ]

        for example in examples:
            print(f"\n{'-' * 80}")
            print(f"Query: {example['query']}")
            print(f"Best Source: {example['best_source']}")
            print(f"Reason: {example['reason']}")
            print(f"{'-' * 80}")


# Example usage
if __name__ == "__main__":
    # Setup
    EMAIL = "your.email@example.com"  # Required for PubMed
    NCBI_KEY = os.getenv("NCBI_API_KEY")  # Optional but recommended
    LLM_KEY = os.getenv("OPENAI_API_KEY")  # For GPT-4, or use None for local models

    try:
        # Initialize unified system
        system = UnifiedKnowledgeSystem(
            email=EMAIL,
            ncbi_api_key=NCBI_KEY,
            llm_api_key=LLM_KEY
        )

        # Example 1: Query about drug interactions
        print("\n" + "=" * 80)
        print("EXAMPLE 1: Drug Interaction Query")
        print("=" * 80)
        system.query_all_sources(
            query="What drugs interact with aspirin?",
            drug_name="aspirin"
        )

        input("\nPress Enter to continue to Example 2...")

        # Example 2: Conceptual question
        print("\n" + "=" * 80)
        print("EXAMPLE 2: Conceptual Question")
        print("=" * 80)
        system.query_all_sources(
            query="How does insulin regulate blood sugar?",
            drug_name=None
        )

        input("\nPress Enter to see source selection strategy...")

        # Demonstrate source selection
        system.demonstrate_source_selection()

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Build the selector model to learn which source to use")
        print("2. Implement reward mechanism for feedback")
        print("3. Train the system with reinforcement learning")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set NCBI_API_KEY in your .env file")
        print("2. Set OPENAI_API_KEY if using GPT-4")
        print("3. Installed all requirements: pip install -r requirements.txt")
