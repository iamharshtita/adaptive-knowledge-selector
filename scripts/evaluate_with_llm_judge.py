# Evaluate test pipeline results using LLM-as-Judge

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from scripts.train_rl_agent import TrainingSystem
from utils.llm_judge import LLMJudge


def evaluate_model_with_judge():
    """
    Run test queries through the model and evaluate answers with LLM judge.
    """

    print("\n" + "=" * 80)
    print("  OFFLINE EVALUATION WITH LLM-AS-JUDGE")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=0.001)

    model_path = "data/rl_selector/adaptive_dqn.pth"
    if not agent.load(model_path):
        print(f"ERROR: Could not load model from {model_path}")
        return

    print(f"✓ Model loaded")

    # Initialize sources
    print("\nInitializing sources...")
    system = TrainingSystem()
    print("✓ Sources ready")

    # Initialize LLM judge
    print("\nInitializing LLM judge...")
    judge = LLMJudge()

    # Test queries (smaller set for offline evaluation - LLM calls are slow)
    test_queries = [
        "What drugs interact with aspirin?",
        "Calculate BMI for 70 kg and 1.75 m tall",
        "How does insulin regulate blood sugar?",
        "What are the FDA approved indications for metformin?",
        "What do the KRR papers say about ontologies in healthcare?",
        "Explain the mechanism of action of beta blockers",
        "Can I take warfarin and ibuprofen together?",
        "What is the creatinine clearance for 65 year old male, 80 kg, creatinine 1.2?"
    ]

    print(f"\nRunning {len(test_queries)} queries through the model...")
    print("=" * 80)

    results_for_judge = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: \"{query}\"")
        print("-" * 80)

        # Encode query
        emb = encoder.encode([query])[0]

        # Get Q-values
        qtable = agent.get_q_table(emb)
        sorted_q = sorted(qtable.items(), key=lambda x: x[1], reverse=True)

        print(f"  Q-Values:")
        for src, val in sorted_q[:3]:  # show top 3
            marker = " ← SELECTED" if src == sorted_q[0][0] else ""
            print(f"    {src:<30s}  Q={val:+.4f}{marker}")

        # Select source
        action_idx = agent.select_action(emb, epsilon=0.0)
        source_name = agent.sources[action_idx]

        print(f"\n  Selected: {source_name}")

        # Query the source
        try:
            results = system.query_source(source_name, query)

            # Extract answer
            if isinstance(results, dict):
                answer = results.get('answer', str(results))
                confidence = results.get('confidence', 0.5)
            else:
                answer = str(results)
                confidence = 0.5

            print(f"  Answer: {answer[:100]}...")
            print(f"  Confidence: {confidence:.2f}")

            # Collect for LLM judge
            results_for_judge.append({
                'query': query,
                'source': source_name,
                'answer': answer,
                'confidence': confidence
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results_for_judge.append({
                'query': query,
                'source': source_name,
                'answer': f"Error: {e}",
                'confidence': 0.0
            })

    # Now evaluate with LLM judge
    print("\n\n" + "=" * 80)
    print("  EVALUATING ANSWERS WITH LLM JUDGE")
    print("=" * 80)

    evaluations = judge.evaluate_batch(
        results_for_judge,
        save_path='data/rl_selector/llm_judge_evaluation.json'
    )

    # Detailed per-query results
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)

    print(f"\n  {'Query':<45s}  {'Source':<20s}  {'Quality':<8s}  {'Conf':<6s}")
    print("  " + "-" * 80)

    for ev in evaluations:
        query_short = ev['query'][:43] + '..' if len(ev['query']) > 45 else ev['query']
        source_short = ev['source'].replace('Source', '')[:18]
        quality = ev['quality']
        confidence = ev.get('confidence', 0.5)

        # Color code by quality
        quality_marker = '✓' if quality >= 0.7 else '⚠' if quality >= 0.4 else '✗'

        print(f"  {query_short:<45s}  {source_short:<20s}  "
              f"{quality:>5.2f} {quality_marker}  {confidence:>5.2f}")

    # Compare model confidence vs LLM-judged quality
    print("\n" + "=" * 80)
    print("  CONFIDENCE VS QUALITY CORRELATION")
    print("=" * 80)

    confidences = [e.get('confidence', 0.5) for e in evaluations]
    qualities = [e['quality'] for e in evaluations]

    # Simple correlation
    import numpy as np
    if len(confidences) > 1:
        correlation = np.corrcoef(confidences, qualities)[0, 1]
        print(f"\n  Correlation: {correlation:.3f}")
        if correlation > 0.5:
            print(f"  ✓ Good! Model confidence aligns with actual quality")
        elif correlation > 0.2:
            print(f"  ⚠ Moderate. Model confidence somewhat predicts quality")
        else:
            print(f"  ✗ Poor. Model confidence doesn't match actual quality")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: data/rl_selector/llm_judge_evaluation.json")


if __name__ == "__main__":
    evaluate_model_with_judge()