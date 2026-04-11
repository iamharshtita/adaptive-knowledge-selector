"""
Script to simulate and train the Reinforcement Learning Adaptive Selector Model
using synthetic queries.
"""

import sys
import os
import random
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_selector import AdaptiveSelector
from models.reward_evaluator import RewardEvaluator
from models.rl_environment import RLEnvironment

class MockUnifiedSystem:
    """A minimal mock of the UnifiedKnowledgeSystem to train the RL loop initially"""
    
    class MockSource:
        def __init__(self, name):
            self.name = name
            
        def search_pubmed(self, query, max_results):
            if "research" in query or "study" in query or "papers" in query: return [{"title": "Research"}] * 4
            return []
            
        def query_drug_interactions(self, drug):
            if drug and ("interact" in drug or "aspirin" in drug): return [{"drug": "X", "severity": "High"}]
            return []
            
        def get_drug_interactions(self, drug):
            if drug and ("dosage" in drug or "protocol" in drug): return [{"drug": "X"}]
            return []
            
        def answer_question(self, query):
            if "explain" in query or "how" in query: return "This is a detailed conceptual explanation of the mechanism... " * 10
            return "Error or short answer"
            
        def semantic_search(self, query, top_k):
            if "manual" in query or "protocol" in query: return [{"score": 0.5, "text": "manual excerpt"}]
            return []

    def __init__(self):
        self.text_source = self.MockSource("Text")
        self.kg_source = self.MockSource("KG")
        self.tool_source = self.MockSource("Tool")
        self.llm_source = self.MockSource("LLM")
        self.pdf_source = self.MockSource("PDF")

def main():
    print("Initializing RL Architecture...")
    
    # 1. Initialize RL components
    agent = AdaptiveSelector(input_dim=384, num_sources=5, lr=0.01)
    evaluator = RewardEvaluator()
    mock_system = MockUnifiedSystem()
    env = RLEnvironment(agent, mock_system, evaluator)
    
    print("Loading Sentence Transformer...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Simple synthetic dataset (Query, expected semantic word)
    queries = [
        ("What is the latest research on aspirin?", "research"), 
        ("What drugs interact with aspirin?", "interact"),
        ("Explain how insulin works conceptually.", "how"),
        ("What is the local protocol manual?", "protocol"),
        ("What is the exact dosage for statins?", "dosage")
    ]
    
    print("\nStarting Online RL Training Loop (300 randomly chosen steps)...\n")
    
    for epoch in range(300):
        query, word_hint = random.choice(queries)
        print(f"Step {epoch+1}/300 | User Query: '{query}'")
        
        # 1. State extraction
        embedded_state = encoder.encode([query])[0]
        
        # 2. RL Step
        source, reward, loss = env.step(query, embedded_state, drug_name=word_hint)
        print("-" * 50)
        
    print("\nTraining Complete! Saving model...")
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rl_selector", "adaptive_dqn.pth")
    agent.save(save_path)
    print(f"Saved weights to {save_path}")
    
if __name__ == "__main__":
    main()
