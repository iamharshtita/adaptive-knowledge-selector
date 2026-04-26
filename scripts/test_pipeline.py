"""
Test the Complete Adaptive Knowledge Selector Pipeline

Tests the trained model with diverse queries to demonstrate:
1. Query routing across all 4 sources
2. Q-value predictions
3. Result quality
4. End-to-end flow

Usage:
    python scripts/test_pipeline.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from models.reward_evaluator import RewardEvaluator, classify_query
from scripts.train_rl_agent import TrainingSystem


def format_results_summary(source_name, results):
    """Format results for display."""
    if results is None:
        return "  (no results)"

    if isinstance(results, dict):
        if source_name == "KnowledgeGraphSource":
            answer = results.get('answer', '')
            return f"  {answer[:200]}"
        elif source_name == "ToolAPISource":
            answer = results.get('answer', '')
            return f"  {answer[:200]}"

    if isinstance(results, str):
        return f"  {results[:200]}..."

    if isinstance(results, list):
        if len(results) == 0:
            return "  (empty list)"
        count = len(results)
        return f"  Found {count} results"

    return f"  {str(results)[:200]}"


def test_pipeline():
    """Test the complete pipeline with diverse queries."""

    print("\n" + "=" * 80)
    print("  ADAPTIVE KNOWLEDGE SELECTOR - PIPELINE TEST")
    print("=" * 80)

    # Load model
    print("\nLoading model and encoder...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=0.001)

    model_path = "data/rl_selector/adaptive_dqn.pth"
    if not agent.load(model_path):
        print(f"ERROR: Could not load model from {model_path}")
        return

    print(f"✓ Model loaded from {model_path}")

    # Initialize sources
    print("\nInitializing knowledge sources...")
    system = TrainingSystem()
    print("✓ All sources ready\n")

    # Test queries
    test_queries = [
        {
            "query": "What drugs interact with aspirin?",
            "expected_type": "interaction",
            "expected_source": "KnowledgeGraphSource"
        },
        {
            "query": "Calculate BMI for a patient weighing 70 kg and 1.75 m tall",
            "expected_type": "dosage",
            "expected_source": "ToolAPISource"
        },
        {
            "query": "How does insulin regulate blood sugar?",
            "expected_type": "concept",
            "expected_source": "LLMSource or PDFKnowledgeSource"
        },
        {
            "query": "What are the FDA approved indications for metformin?",
            "expected_type": "label",
            "expected_source": "ToolAPISource"
        },
        {
            "query": "What do the KRR papers say about ontologies in healthcare?",
            "expected_type": "document",
            "expected_source": "PDFKnowledgeSource"
        },
        {
            "query": "Explain the mechanism of action of beta blockers",
            "expected_type": "concept",
            "expected_source": "LLMSource"
        },
        {
            "query": "Can I take warfarin and ibuprofen together?",
            "expected_type": "interaction",
            "expected_source": "KnowledgeGraphSource"
        },
        {
            "query": "What is the creatinine clearance for a 65 year old male, 80 kg, creatinine 1.2?",
            "expected_type": "dosage",
            "expected_source": "ToolAPISource"
        }
    ]

    results_summary = []

    print("=" * 80)
    print("TESTING PIPELINE")
    print("=" * 80 + "\n")

    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected_type = test["expected_type"]
        expected_source = test["expected_source"]

        print(f"\n[{i}/{len(test_queries)}] Query: \"{query}\"")
        print("-" * 80)

        # Classify query
        qtype = classify_query(query)
        print(f"  Query Type: [{qtype}]")

        # Encode query
        emb = encoder.encode([query])[0]

        # Get Q-values
        qtable = agent.get_q_table(emb)
        sorted_q = sorted(qtable.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Q-Values (Predicted Preferences):")
        for src, val in sorted_q:
            marker = " ← SELECTED" if src == sorted_q[0][0] else ""
            print(f"    {src:<30s}  Q={val:+.4f}{marker}")

        # Select source
        action_idx = agent.select_action(emb, epsilon=0.0)  # No exploration
        source_name = agent.sources[action_idx]

        print(f"\n  Selected Source: {source_name}")
        print(f"  Expected: {expected_source}")

        correct = "✓" if expected_source in source_name or "or" in expected_source else "?"
        print(f"  Match: {correct}")

        # Query the source
        print(f"\n  Querying {source_name}...")
        try:
            results = system.query_source(source_name, query)

            # Compute reward
            reward = RewardEvaluator.compute_reward(query, source_name, results)

            print(f"\n  Results:")
            print(format_results_summary(source_name, results))
            print(f"\n  Reward: {reward:.2f} {'✓' if reward >= 0.5 else '✗'}")

            results_summary.append({
                'query': query[:50],
                'selected': source_name,
                'expected': expected_source,
                'reward': reward,
                'correct': correct
            })

        except Exception as e:
            print(f"\n  ERROR: {e}")
            results_summary.append({
                'query': query[:50],
                'selected': source_name,
                'expected': expected_source,
                'reward': 0.0,
                'correct': '✗'
            })

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    correct_count = sum(1 for r in results_summary if r['correct'] == '✓')
    avg_reward = sum(r['reward'] for r in results_summary) / len(results_summary)

    print(f"\nTotal Queries: {len(results_summary)}")
    print(f"Correct Routing: {correct_count}/{len(results_summary)} ({100*correct_count/len(results_summary):.1f}%)")
    print(f"Average Reward: {avg_reward:.3f}")

    print("\n  Per-Query Results:")
    print("  " + "-" * 76)
    print(f"  {'Query':<40s}  {'Selected':<20s}  {'Reward':<8s}  {'OK'}")
    print("  " + "-" * 76)

    for r in results_summary:
        query_short = r['query'][:38] + '..' if len(r['query']) > 40 else r['query']
        source_short = r['selected'].replace('Source', '')[:18]
        print(f"  {query_short:<40s}  {source_short:<20s}  {r['reward']:>6.2f}    {r['correct']}")

    print("  " + "-" * 76)

    print("\n  Source Distribution:")
    from collections import Counter
    source_counts = Counter([r['selected'] for r in results_summary])
    for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        src_short = src.replace('Source', '')
        pct = 100 * count / len(results_summary)
        bar = '█' * int(pct / 5)
        print(f"    {src_short:<28s}  {count:>2d} ({pct:>5.1f}%)  {bar}")

    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_pipeline()