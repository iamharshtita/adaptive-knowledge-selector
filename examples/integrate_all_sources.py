"""
Unified Integration Example
Demonstrates how to use all knowledge sources together
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_sources.knowledge_graph import KnowledgeGraphSource
from knowledge_sources.tool_api_source import ToolAPISource
from knowledge_sources.llm_source import LLMSource


class UnifiedKnowledgeSystem:
    """
    Unified system that integrates all knowledge sources
    """

    def __init__(self, llm_api_key: str = None):
        """
        Initialize all knowledge sources

        Args:
            llm_api_key: API key for LLM (if using commercial model)
        """
        print("Initializing Unified Knowledge System...")

        # Initialize all sources
        print("1. Loading Knowledge Graph Source...")
        self.kg_source = KnowledgeGraphSource()
        # Try to load Hetionet from pickle (fast) or JSON
        pickle_path = "data/hetionet/hetionet_graph.pkl"
        json_path = "data/hetionet/hetionet-v1.0.json"

        if os.path.exists(pickle_path):
            print("   Loading Hetionet from pickle (fast)...")
            self.kg_source.load_graph(pickle_path)
        elif os.path.exists(json_path):
            print("   Loading Hetionet from JSON (slower)...")
            self.kg_source.load_hetionet(json_path)
        else:
            print("   Warning: Hetionet not found. Run: python scripts/setup_hetionet.py")
            print("   Continuing without knowledge graph...")

        print("2. Loading Tool/API Source...")
        self.tool_source = ToolAPISource()

        print("3. Loading LLM Source...")
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

        # 1. Knowledge Graph Source
        if drug_name:
            print(f"\n\n### SOURCE 1: Knowledge Graph (Drug Interactions for {drug_name}) ###")
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

        # 2. Tool/API Source
        print(f"\n\n### SOURCE 2: Tool/API ###")
        try:
            result = self.tool_source.query(query)
            print(f"Function: {result.get('function', 'N/A')}")
            print(f"Answer: {result.get('answer', 'N/A')}")
        except Exception as e:
            print(f"Error querying Tool/API: {e}")

        # 3. LLM Source
        print(f"\n\n### SOURCE 3: LLM (Parametric Knowledge) ###")
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
                'query': 'What do PDFs say about knowledge representation?',
                'best_source': 'PDF Knowledge',
                'reason': 'Document-based search in research papers',
                'drug': None
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
    LLM_KEY = os.getenv("GROQ_API_KEY")  # For Groq, or use OPENAI_API_KEY for GPT-4

    try:
        # Initialize unified system
        system = UnifiedKnowledgeSystem(llm_api_key=LLM_KEY)

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
        print("1. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Run python scripts/setup_hetionet.py to set up the knowledge graph")
