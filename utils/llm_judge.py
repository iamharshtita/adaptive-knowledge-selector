# LLM-as-Judge Quality Evaluator
# Uses AWS Nova Lite to evaluate answer quality offline

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from knowledge_sources.llm_source import LLMSource
import json
import re


class LLMJudge:
    """
    Uses LLM to evaluate answer quality.
    Run this OFFLINE on saved results, not during training.
    """

    def __init__(self):
        # Use Nova Lite - cheap and fast
        self.llm = LLMSource(model_id="us.amazon.nova-lite-v1:0")
        print(f"✓ LLM Judge initialized (Model: {self.llm.model_id})")

    def evaluate_quality(self, query, answer, source_name):
        """
        Evaluate answer quality on three dimensions:
        - Correctness (0-1): Is it factually accurate?
        - Relevance (0-1): Does it answer the question?
        - Completeness (0-1): Is enough detail provided?

        Returns dict with scores and overall quality.
        """
        if not answer or answer == "None":
            return {
                'correctness': 0.0,
                'relevance': 0.0,
                'completeness': 0.0,
                'quality': 0.0,
                'explanation': 'No answer provided'
            }

        # Build prompt for LLM judge
        prompt = f"""You are evaluating a medical answer for quality.

Query: {query}
Source: {source_name}
Answer: {answer}

Rate the answer on three dimensions (0.0 to 1.0):

1. Correctness: Is the information factually accurate?
   - 1.0 = completely accurate
   - 0.5 = partially accurate or uncertain
   - 0.0 = incorrect or error message

2. Relevance: Does it answer the question asked?
   - 1.0 = directly answers the question
   - 0.5 = partially relevant
   - 0.0 = off-topic or doesn't address query

3. Completeness: Is enough detail provided?
   - 1.0 = comprehensive answer
   - 0.5 = basic answer, missing some details
   - 0.0 = incomplete or too vague

Respond ONLY with JSON in this exact format:
{{
  "correctness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "explanation": "brief reason for scores"
}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=200, temperature=0.1)

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                # Fallback: try to parse the whole response
                scores = json.loads(response)

            # Calculate overall quality (average of three scores)
            quality = (
                scores.get('correctness', 0.5) +
                scores.get('relevance', 0.5) +
                scores.get('completeness', 0.5)
            ) / 3.0

            return {
                'correctness': scores.get('correctness', 0.5),
                'relevance': scores.get('relevance', 0.5),
                'completeness': scores.get('completeness', 0.5),
                'quality': quality,
                'explanation': scores.get('explanation', '')
            }

        except Exception as e:
            print(f"Warning: LLM judge failed for query '{query[:50]}...': {e}")
            # Fallback to heuristic scoring
            return self._fallback_heuristic_score(query, answer)

    def _fallback_heuristic_score(self, query, answer):
        """Simple heuristic if LLM judge fails"""
        answer_text = str(answer).lower()
        query_words = set(query.lower().split())
        answer_words = set(answer_text.split())

        # Relevance: query word overlap
        relevance = len(query_words & answer_words) / len(query_words) if query_words else 0.0

        # Completeness: length-based
        completeness = min(len(answer_text) / 200.0, 1.0)

        # Correctness: check for error signals
        correctness = 0.5
        if 'error' in answer_text or 'not found' in answer_text:
            correctness = 0.0
        elif len(answer_text) > 100:
            correctness = 0.7

        quality = (correctness + relevance + completeness) / 3.0

        return {
            'correctness': correctness,
            'relevance': relevance,
            'completeness': completeness,
            'quality': quality,
            'explanation': 'Heuristic fallback (LLM judge failed)'
        }

    def evaluate_batch(self, results_list, save_path=None):
        """
        Evaluate a batch of query results.

        Args:
            results_list: List of dicts with 'query', 'source', 'answer'
            save_path: Optional path to save evaluation results

        Returns:
            List of evaluation results with scores
        """
        print(f"\nEvaluating {len(results_list)} results...")
        evaluations = []

        for i, item in enumerate(results_list, 1):
            query = item['query']
            source = item['source']
            answer = item['answer']

            print(f"  [{i}/{len(results_list)}] Evaluating: {query[:50]}...")

            eval_result = self.evaluate_quality(query, answer, source)
            eval_result['query'] = query
            eval_result['source'] = source
            eval_result['answer'] = answer[:200]  # truncate for storage

            evaluations.append(eval_result)

        # Summary statistics
        avg_quality = sum(e['quality'] for e in evaluations) / len(evaluations)
        avg_correctness = sum(e['correctness'] for e in evaluations) / len(evaluations)
        avg_relevance = sum(e['relevance'] for e in evaluations) / len(evaluations)
        avg_completeness = sum(e['completeness'] for e in evaluations) / len(evaluations)

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Evaluated: {len(evaluations)}")
        print(f"Average Quality:      {avg_quality:.3f}")
        print(f"Average Correctness:  {avg_correctness:.3f}")
        print(f"Average Relevance:    {avg_relevance:.3f}")
        print(f"Average Completeness: {avg_completeness:.3f}")
        print("=" * 70)

        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump({
                    'evaluations': evaluations,
                    'summary': {
                        'count': len(evaluations),
                        'avg_quality': avg_quality,
                        'avg_correctness': avg_correctness,
                        'avg_relevance': avg_relevance,
                        'avg_completeness': avg_completeness
                    }
                }, f, indent=2)
            print(f"\n✓ Saved evaluation results to: {save_path}")

        return evaluations


if __name__ == "__main__":
    # Example usage
    judge = LLMJudge()

    # Test with a few sample queries
    test_results = [
        {
            'query': 'What drugs interact with aspirin?',
            'source': 'KnowledgeGraphSource',
            'answer': 'Drug Interactions for: aspirin\nFound 5 interacting drugs:\n  1. Warfarin\n  2. Ibuprofen\n  3. Clopidogrel'
        },
        {
            'query': 'Calculate BMI for 70 kg and 1.75 m',
            'source': 'ToolAPISource',
            'answer': 'BMI = 22.86 (Normal) for 70 kg, 1.75 m.'
        },
        {
            'query': 'How does insulin work?',
            'source': 'LLMSource',
            'answer': 'Insulin is a hormone that regulates blood glucose. It allows cells to take up glucose from the bloodstream.'
        }
    ]

    evaluations = judge.evaluate_batch(
        test_results,
        save_path='data/rl_selector/llm_judge_test.json'
    )

    print("\nDetailed Results:")
    for ev in evaluations:
        print(f"\nQuery: {ev['query']}")
        print(f"Quality: {ev['quality']:.2f} | "
              f"Correct: {ev['correctness']:.2f} | "
              f"Relevant: {ev['relevance']:.2f} | "
              f"Complete: {ev['completeness']:.2f}")
        print(f"Reason: {ev['explanation']}")