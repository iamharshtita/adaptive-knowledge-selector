"""
Train the Adaptive Knowledge Selector via Reinforcement Learning.

This script:
  1. Initialises all 5 knowledge sources
  2. Runs curated medical queries through the RL loop
  3. The DQN learns which source to pick for each query type
  4. Saves trained weights to data/rl_selector/adaptive_dqn.pth

Usage:
    python scripts/train_rl_agent.py
    python scripts/train_rl_agent.py --episodes 200 --fresh
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from models.reward_evaluator import RewardEvaluator, classify_query
from models.rl_environment import ReplayBuffer


# ===========================================================================
# Training Queries - categorised by expected best source
# ===========================================================================
TRAINING_QUERIES = [
    # Drug interactions - KnowledgeGraph / ToolAPI
    {"q": "What drugs interact with aspirin?", "drug": "aspirin"},
    {"q": "Can I take warfarin and ibuprofen together?", "drug": "warfarin"},
    {"q": "Drug interactions of metformin", "drug": "metformin"},
    {"q": "What are the contraindications of combining ACE inhibitors with potassium supplements?", "drug": None},
    {"q": "Does clopidogrel interact with omeprazole?", "drug": "clopidogrel"},
    {"q": "What happens if you combine SSRIs with MAOIs?", "drug": None},
    {"q": "Interactions between statins and grapefruit juice", "drug": None},
    {"q": "Can methotrexate be taken with NSAIDs?", "drug": "methotrexate"},
    {"q": "Drug-drug interaction between digoxin and amiodarone", "drug": "digoxin"},
    {"q": "Is it safe to take aspirin with blood thinners?", "drug": "aspirin"},

    # Dosage/Calculations - ToolAPI
    {"q": "Calculate BMI for a patient weighing 70 kg and 1.75 m tall", "drug": None},
    {"q": "What is the pediatric dose for a 20 kg child if adult dose is 500 mg?", "drug": None},
    {"q": "Creatinine clearance calculation for 65 year old male 80 kg creatinine 1.2", "drug": None},
    {"q": "Ideal body weight for a 175 cm male", "drug": None},
    {"q": "Calculate BMI for 90 kg and 1.80 m", "drug": None},
    {"q": "What is the adult dosage of amoxicillin?", "drug": "amoxicillin"},
    {"q": "Calculate creatinine clearance age 50 female weight 60 kg Cr 0.9", "drug": None},
    {"q": "Pediatric dose calculation for 15 kg child adult dose 250 mg", "drug": None},
    {"q": "Body mass index for 100 kg 1.65 m patient", "drug": None},
    {"q": "Clark's rule for dose: child weight 28 kg adult dose 800 mg", "drug": None},

    # Conceptual questions - LLM
    {"q": "How does insulin regulate blood sugar?", "drug": None},
    {"q": "Explain the mechanism of action of beta blockers", "drug": None},
    {"q": "What is the difference between Type 1 and Type 2 diabetes?", "drug": None},
    {"q": "Describe the pathophysiology of hypertension", "drug": None},
    {"q": "Why do antibiotics not work against viruses?", "drug": None},
    {"q": "How do ACE inhibitors lower blood pressure?", "drug": None},
    {"q": "What role does the liver play in drug metabolism?", "drug": None},
    {"q": "Explain the concept of drug bioavailability", "drug": None},
    {"q": "How does chemotherapy target cancer cells?", "drug": None},
    {"q": "What is pharmacokinetics vs pharmacodynamics?", "drug": None},

    # Research / Literature - TextKnowledge (PubMed)
    {"q": "Latest research on aspirin and cardiovascular prevention", "drug": "aspirin"},
    {"q": "Recent studies on metformin for diabetes treatment", "drug": "metformin"},
    {"q": "Clinical trial results for new SGLT2 inhibitors", "drug": None},
    {"q": "Systematic review of statin therapy benefits", "drug": None},
    {"q": "Evidence for aspirin use in cancer prevention", "drug": "aspirin"},
    {"q": "PubMed studies on drug-resistant tuberculosis treatment", "drug": None},
    {"q": "Meta-analysis of ACE inhibitors vs ARBs", "drug": None},
    {"q": "Research findings on GLP-1 receptor agonists for obesity", "drug": None},
    {"q": "Journal articles on antibiotic resistance mechanisms", "drug": None},
    {"q": "Recent evidence on COVID-19 antivirals efficacy", "drug": None},

    # Document / PDF - PDFKnowledgeSource
    {"q": "What do the KRR papers say about ontologies in healthcare?", "drug": None},
    {"q": "Knowledge representation and reasoning in clinical decision support", "drug": None},
    {"q": "How are medical ontologies used in the documents provided?", "drug": None},
    {"q": "Clinical reasoning approaches described in the PDF papers", "drug": None},
    {"q": "What representation formalisms are discussed in the KRR documents?", "drug": None},
    {"q": "OWL and description logic in medical knowledge representation", "drug": None},
    {"q": "Decision support systems mentioned in the papers", "drug": None},
    {"q": "How does the paper discuss SNOMED CT and medical terminologies?", "drug": None},
    {"q": "What are the challenges of knowledge representation in biomedicine?", "drug": None},
    {"q": "Semantic web technologies for healthcare in the documents", "drug": None},

    # Drug Labels - ToolAPI
    {"q": "What are the FDA label indications for lisinopril?", "drug": "lisinopril"},
    {"q": "Show me the prescribing information for atorvastatin", "drug": "atorvastatin"},
    {"q": "What are the warnings on the drug label for warfarin?", "drug": "warfarin"},
    {"q": "FDA approved uses of omeprazole", "drug": "omeprazole"},
    {"q": "Boxed warning for methotrexate", "drug": "methotrexate"},
]


# ===========================================================================
# Lightweight system that wires up all 4 sources
# ===========================================================================
class TrainingSystem:
    """Wraps all knowledge sources for the RL loop."""

    def __init__(self):
        print("=" * 70)
        print("  INITIALISING KNOWLEDGE SOURCES")
        print("=" * 70)

        # 1. Knowledge Graph
        print("\n[1/4] Knowledge Graph (Hetionet) …")
        try:
            from knowledge_sources.knowledge_graph import KnowledgeGraphSource
            self.kg_source = KnowledgeGraphSource()
            pkl = "data/hetionet/hetionet_graph.pkl"
            if os.path.exists(pkl):
                self.kg_source.load_graph(pkl)
                print(f"  Loaded ({self.kg_source.graph.number_of_nodes():,} nodes)")
            else:
                print("  Warning: No Hetionet data found")
                self.kg_source = None
        except Exception as e:
            print(f"  Warning: Failed - {e}")
            self.kg_source = None

        # 2. Tool/API
        print("\n[2/4] Tool/API (OpenFDA/RxNorm)...")
        try:
            from knowledge_sources.tool_api_source import ToolAPISource
            self.tool_source = ToolAPISource()
            print("  Ready")
        except Exception as e:
            print(f"  Warning: Failed - {e}")
            self.tool_source = None

        # 3. LLM
        print("\n[3/4] LLM (AWS Bedrock - Claude 3.5 Haiku)...")
        try:
            from knowledge_sources.llm_source import LLMSource
            self.llm_source = LLMSource()
            print("  Ready")
        except Exception as e:
            print(f"  Warning: Failed - {e}")
            self.llm_source = None

        # 4. PDF
        print("\n[4/4] PDF Knowledge...")
        try:
            from knowledge_sources.pdf_knowledge import PDFKnowledgeSource
            self.pdf_source = PDFKnowledgeSource()
            # Try loading existing store, else build from PDFs
            store_dir = "data/pdf_store"
            pdf_dir = "pdfs"
            if os.path.exists(os.path.join(store_dir, "pdf_index.faiss")):
                self.pdf_source.load_vector_store(store_dir)
            elif os.path.isdir(pdf_dir):
                chunks = self.pdf_source.ingest_directory(pdf_dir)
                if chunks:
                    self.pdf_source.build_vector_store(chunks)
                    self.pdf_source.save_vector_store(store_dir)
            print(f"  Ready ({self.pdf_source.index.ntotal if self.pdf_source.index else 0} chunks)")
        except Exception as e:
            print(f"  Warning: Failed - {e}")
            self.pdf_source = None

        print("\n" + "=" * 70)

    def query_source(self, source_name: str, query: str):
        """
        Execute a query against the named source using the unified query() interface.
        Returns the full result dict from each source's query() method.
        """
        try:
            if source_name == "KnowledgeGraphSource" and self.kg_source:
                return self.kg_source.query(query)

            elif source_name == "ToolAPISource" and self.tool_source:
                return self.tool_source.query(query)

            elif source_name == "LLMSource" and self.llm_source:
                return self.llm_source.query(query)

            elif source_name == "PDFKnowledgeSource" and self.pdf_source:
                return self.pdf_source.query(query, top_k=3)

        except Exception as e:
            print(f"    Warning: {source_name} error - {e}")
        return None


# ===========================================================================
# Training Loop
# ===========================================================================
def train(args):
    # Setup
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=args.lr)
    system = TrainingSystem()
    memory = ReplayBuffer(capacity=2000)

    model_path = args.model_path
    if not args.fresh and os.path.exists(model_path):
        if agent.load(model_path):
            print(f"\nLoaded existing weights from {model_path}")

    # Pre-embed all training queries
    print("\nEncoding training queries...")
    query_texts = [q["q"] for q in TRAINING_QUERIES]
    embeddings = encoder.encode(query_texts, show_progress_bar=True)

    # Training hyperparameters
    epsilon = args.epsilon_start
    epsilon_min = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    batch_size = 16

    # Tracking
    rewards_log = []
    loss_log = []
    episode_log = []  # Per-episode details for visualization
    source_picks = dict.fromkeys(agent.sources.values(), 0)
    total_reward = 0.0
    total_steps = 0

    print("\n" + "=" * 70)
    print(f"  TRAINING START - {args.episodes} episodes, epsilon={epsilon:.2f} to {epsilon_min:.2f}")
    print("=" * 70 + "\n")

    t_start = time.time()

    for ep in range(1, args.episodes + 1):
        # Cycle through queries
        idx = (ep - 1) % len(TRAINING_QUERIES)
        qdata = TRAINING_QUERIES[idx]
        query = qdata["q"]
        emb = embeddings[idx]

        # 1. Select action
        action_idx = agent.select_action(emb, epsilon)
        source_name = agent.sources[action_idx]
        source_picks[source_name] += 1

        # 2. Execute query
        results = system.query_source(source_name, query)

        # 3. Compute reward (3-tier)
        reward = RewardEvaluator.compute_reward(query, source_name, results)

        # 4. Store experience
        memory.push(emb, action_idx, reward)
        total_reward += reward
        total_steps += 1

        # 5. Train on batch
        loss = 0.0
        if len(memory) >= batch_size:
            states, actions, rewards_batch = memory.sample(batch_size)
            loss = agent.train_step(states, actions, rewards_batch)
            agent.update_target_network()
            loss_log.append(loss)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards_log.append(reward)

        # Track per-episode detail for visualization
        qtype = classify_query(query)
        episode_log.append({
            "ep": ep,
            "query": query[:60],
            "query_type": qtype,
            "source_chosen": source_name,
            "reward": round(reward, 3),
            "loss": round(loss, 6),
            "epsilon": round(epsilon, 4),
        })

        # Logging
        if ep % 5 == 0 or ep <= 3:
            avg_r = np.mean(rewards_log[-20:]) if rewards_log else 0
            print(
                f"  Ep {ep:>4d}  |  eps={epsilon:.3f}  |  "
                f"R={reward:.2f}  avg={avg_r:.3f}  |  "
                f"loss={loss:.4f}  |  "
                f"{source_name:<24s}  |  [{qtype}] {query[:45]}..."
            )

        # Periodic save
        if ep % 50 == 0:
            agent.save(model_path)

    elapsed = time.time() - t_start

    # Final save
    agent.save(model_path)

    # Summary
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE - {total_steps} steps in {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  Average reward: {total_reward / max(total_steps, 1):.3f}")
    print(f"  Final epsilon:  {epsilon:.4f}")
    print(f"  Model saved to: {model_path}")

    print("\n  Source selection distribution:")
    for src, count in sorted(source_picks.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(count / max(total_steps, 1) * 40)
        print(f"    {src:<24s}  {count:>4d}  ({100*count/max(total_steps,1):.1f}%)  {bar}")

    # Q-value snapshot for sample queries
    print("\n  Q-value snapshot (learned preferences):")
    test_queries = [
        ("What drugs interact with aspirin?", "interaction"),
        ("Calculate BMI for 70 kg 1.75 m", "dosage"),
        ("How does insulin work?", "concept"),
        ("Latest research on statins", "research"),
        ("KRR paper discussion on ontologies", "document"),
    ]
    for tq, qtype in test_queries:
        emb = encoder.encode([tq])[0]
        qtable = agent.get_q_table(emb)
        best = max(qtable, key=qtable.get)
        print(f"    [{qtype:>12s}]  {tq[:40]:<42s}  ->  {best}")
        for src, val in sorted(qtable.items(), key=lambda x: x[1], reverse=True):
            mark = " <--" if src == best else ""
            print(f"                    {src:<24s}  Q={val:+.4f}{mark}")

    # Save training log
    log_path = os.path.join(os.path.dirname(model_path), "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "rewards": rewards_log,
            "losses": loss_log,
            "episodes": episode_log,
            "source_picks": source_picks,
            "total_episodes": total_steps,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\n  Training log saved to: {log_path}")
    print("=" * 70 + "\n")


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Adaptive Knowledge Selector")
    parser.add_argument("--episodes", type=int, default=150,
                        help="Number of training episodes (default: 150)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Starting epsilon (default: 1.0)")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Minimum epsilon (default: 0.05)")
    parser.add_argument("--epsilon-decay", type=float, default=0.98,
                        help="Epsilon decay per episode (default: 0.98)")
    parser.add_argument("--model-path", type=str,
                        default="data/rl_selector/adaptive_dqn.pth",
                        help="Path to save/load model weights")
    parser.add_argument("--fresh", action="store_true",
                        help="Start training from scratch (ignore existing weights)")
    args = parser.parse_args()
    train(args)
