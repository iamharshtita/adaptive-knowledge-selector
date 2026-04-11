import sys
import os
from sentence_transformers import SentenceTransformer

# 0. Important: Add the project root to sys.path so Python can find the 'models' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_selector import AdaptiveSelector

def main():
    print("Loading Sentence Transformer...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 1. Load the smart brain
    print("Loading Trained Brain...")
    agent = AdaptiveSelector()
    
    # Path to the saved weights from your training session
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rl_selector", "adaptive_dqn.pth")
    success = agent.load(weights_path)
    if not success:
        print(f"Could not load weights at {weights_path}")
        return

    # 2. Interactive Loop
    print("\n" + "=" * 50)
    print("🧠 Adaptive Knowledge Selector Online!")
    print("=" * 50)
    
    while True:
        try:
            new_query = input("\nEnter your medical query (or type 'exit' to quit): ")
            if new_query.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not new_query.strip():
                continue
                
            embedded_query = encoder.encode([new_query])[0]

            # 3. Have the neural network tell you which source is best to choose!
            best_source_id = agent.select_action(embedded_query, epsilon=0.0) 
            best_source_name = agent.sources[best_source_id]
            
            print(f"🤖 The Neural Network routes this to -> '{best_source_name}'")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
