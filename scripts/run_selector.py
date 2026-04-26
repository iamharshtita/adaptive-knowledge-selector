"""
Interactive Adaptive Knowledge Selector — Inference + Online Learning

This script:
  1. Loads the trained DQN model
  2. Takes user queries interactively
  3. Routes to the best knowledge source
  4. Evaluates the response → reward → updates weights (self-learning)

Usage:
    python scripts/run_selector.py
    python scripts/run_selector.py --no-learn   # inference only, no weight updates
"""

import os
import sys
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
from knowledge_sources.tool_api_source import fetch_drug_name


MODEL_PATH = "data/rl_selector/adaptive_dqn.pth"


def truncate(text, max_len=500):
    """Truncate text for display."""
    text = str(text).strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def format_results(source_name, results, kg_source=None):
    """Pretty-print results from any source."""
    if results is None:
        return "  (no results — source may not be available)"

    # Handle empty lists/strings
    if isinstance(results, (list, str)) and len(results) == 0:
        return "  (no results found for this query)"

    lines = []

    if source_name == "PDFKnowledgeSource" and isinstance(results, list):
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            src = r.get("metadata", {}).get("source", "?")
            page = r.get("metadata", {}).get("page", "?")
            lines.append(f"  {i}. [score={score:.3f}] {src} p.{page}")
            lines.append(f"     {truncate(r['text'], 200)}")
    elif source_name == "TextKnowledgeSource" and isinstance(results, list):
        if not results:
            lines.append("  (no PubMed articles found)")
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.get('title', 'No title')}")
            lines.append(f"     PMID: {r.get('pmid', '?')}  Journal: {r.get('journal', '?')}")
            abstract = r.get("abstract", "")
            if abstract:
                lines.append(f"     {truncate(abstract, 250)}")
    elif source_name == "LLMSource" and isinstance(results, str):
        lines.append(f"  {truncate(results, 400)}")
    elif source_name == "ToolAPISource" and isinstance(results, dict):
        fn = results.get("function", "?")
        answer = results.get("answer", "")
        lines.append(f"  🔧 Function: {fn}")
        lines.append(f"  {answer}")
    elif source_name == "KnowledgeGraphSource" and isinstance(results, list):
        if not results:
            lines.append("  (no graph relationships found)")
        else:
            lines.append(f"  Found {len(results)} connected entities:")
            for i, node_id in enumerate(results[:10], 1):
                # Node IDs like "Side Effect::C0009421" — extract kind + readable name
                parts = str(node_id).split("::", 1)
                kind = parts[0] if len(parts) > 1 else "Entity"
                raw_id = parts[1] if len(parts) > 1 else str(node_id)

                # Try to resolve from KG graph if available
                display_name = raw_id
                if kg_source and hasattr(kg_source, 'graph'):
                    node_data = kg_source.graph.nodes.get(str(node_id), {})
                    if node_data.get('name'):
                        display_name = node_data['name']

                lines.append(f"  {i}. [{kind}] {display_name}")
            if len(results) > 10:
                lines.append(f"  ... and {len(results) - 10} more")
    elif isinstance(results, list):
        for i, r in enumerate(results[:5], 1):
            lines.append(f"  {i}. {truncate(str(r), 150)}")
        if len(results) > 5:
            lines.append(f"  ... and {len(results) - 5} more")
    elif isinstance(results, dict):
        for k, v in results.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append(f"  {truncate(str(results), 300)}")

    return "\n".join(lines) if lines else "  (empty response)"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Adaptive Knowledge Selector")
    parser.add_argument("--no-learn", action="store_true",
                        help="Disable online learning (inference only)")
    args = parser.parse_args()

    online_learning = not args.no_learn

    # ── Load encoder + agent ──
    print("\n🔧 Loading encoder …")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    agent = AdaptiveSelector(input_dim=384, num_sources=5, lr=0.0005)
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"📂 Loaded trained model from {MODEL_PATH}")
    else:
        print("⚠️  No trained model found — using random weights")
        print("   Run `python scripts/train_rl_agent.py` first for better results")

    # ── Load knowledge sources ──
    # Lazy import to avoid loading everything if not needed
    from scripts.train_rl_agent import TrainingSystem
    system = TrainingSystem()

    memory = ReplayBuffer(capacity=500)
    epsilon = 0.05  # Low exploration during inference
    queries_processed = 0

    # ── Interactive loop ──
    print(f"\n{'═' * 70}")
    print(f"  ADAPTIVE KNOWLEDGE SELECTOR — Interactive Mode")
    print(f"  Online learning: {'ON ✅' if online_learning else 'OFF'}")
    print(f"{'═' * 70}")
    print("  Type your medical question below. Type 'quit' to exit.")
    print("  Type 'qtable' to see learned Q-values for the last query.")
    print(f"{'═' * 70}\n")

    last_embedding = None

    while True:
        try:
            query = input("🔎 Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "qtable" and last_embedding is not None:
            qtable = agent.get_q_table(last_embedding)
            print("\n  Q-values (learned preferences):")
            for src, val in sorted(qtable.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * max(int((val + 1) * 10), 0)
                print(f"    {src:<24s}  Q={val:+.4f}  {bar}")
            print()
            continue

        # 1. Classify & embed
        qtype = classify_query(query)
        emb = encoder.encode([query])[0]
        last_embedding = emb

        # 2. Select source
        action_idx = agent.select_action(emb, epsilon)
        source_name = agent.sources[action_idx]

        # 3. Show Q-values
        qtable = agent.get_q_table(emb)
        best_q = max(qtable.values())

        print(f"\n  Query type: [{qtype}]")
        print(f"  Selected:   {source_name}  (Q={qtable[source_name]:+.4f})")

        # 4. Execute query
        # Use regex-based drug extraction instead of hardcoded list
        drug = fetch_drug_name(query)

        print(f"  Querying …")
        results = system.query_source(source_name, query, drug)

        # 5. Compute reward
        reward = RewardEvaluator.compute_reward(query, source_name, results)

        # 6. Display results
        print(f"\n  {'─' * 60}")
        print(f"  📋 Results from {source_name}  (reward: {reward:.2f})")
        print(f"  {'─' * 60}")
        print(format_results(source_name, results, kg_source=getattr(system, 'kg_source', None)))
        print(f"  {'─' * 60}\n")

        # 7. Online learning
        if online_learning:
            memory.push(emb, action_idx, reward)
            queries_processed += 1

            if len(memory) >= 8:
                states, actions, rewards = memory.sample(8)
                loss = agent.train_step(states, actions, rewards)
                agent.update_target_network()
                print(f"  🧠 Online learning: loss={loss:.4f} (buffer={len(memory)})")

            # Save periodically
            if queries_processed % 5 == 0:
                agent.save(MODEL_PATH)
                print(f"  💾 Weights saved ({queries_processed} queries processed)")

        print()

    # ── Save on exit ──
    if online_learning and queries_processed > 0:
        agent.save(MODEL_PATH)
        print(f"\n💾 Final weights saved after {queries_processed} queries.")

    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
