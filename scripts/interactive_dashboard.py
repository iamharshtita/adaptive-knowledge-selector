# Interactive Query Dashboard - Collect experiences for RL training

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector
from models.reward_evaluator import RewardEvaluator, classify_query
from scripts.train_rl_agent import TrainingSystem
from utils.llm_judge import LLMJudge
from utils.s3_sync import S3TeamSync
from utils.reward_blending import extract_rewards_from_experiences


class InteractiveDashboard:
    """Interactive dashboard showing routing decisions with LLM evaluation"""

    def __init__(self):
        print("\n" + "=" * 80)
        print("  ADAPTIVE KNOWLEDGE SELECTOR - INTERACTIVE DASHBOARD")
        print("=" * 80)
        print("\n  Loading components...")

        # Load model
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=0.001)

        model_path = "data/rl_selector/adaptive_dqn.pth"

        # S3 sync setup (initialize early for download)
        self.s3_sync = None
        try:
            self.s3_sync = S3TeamSync()
            print("  ✓ S3 connection established")

            # Try to download latest model from S3 (team collaboration)
            try:
                print("  📥 Checking S3 for latest model...")
                self.s3_sync.download('model')
                print("  ✓ Downloaded latest model from S3")
            except Exception as e:
                print(f"  ℹ Using local model (S3 download skipped: {str(e)[:50]})")

        except Exception as e:
            print(f"  ⚠ S3 sync disabled: {str(e)[:50]}")

        # Load model (either fresh from S3 or local)
        if not self.agent.load(model_path):
            print(f"ERROR: Could not load model from {model_path}")
            sys.exit(1)

        print("  ✓ Model loaded")

        # Initialize sources
        self.system = TrainingSystem()
        print("  ✓ Sources ready")

        # Initialize LLM judge
        self.judge = LLMJudge()
        print("  ✓ LLM Judge ready")

        # Experience log
        self.experience_log = "data/rl_selector/online_experiences.jsonl"
        self.query_count = 0
        self.batch_size = 16  # Trigger training after 16 experiences
        self.model_path = model_path

        print("\n" + "=" * 80)

    def process_query(self, query):
        """Process query and show full evaluation"""

        print("\n" + "=" * 80)
        print(f"QUERY #{self.query_count + 1}: \"{query}\"")
        print("=" * 80)

        # Step 1: Query Classification
        qtype = classify_query(query)
        print(f"\n[1] QUERY CLASSIFICATION:")
        print(f"    Type: {qtype}")

        # Step 2: Encode query
        emb = self.encoder.encode([query])[0]

        # Step 3: Router Decision (Q-values)
        print(f"\n[2] ROUTER DECISION (Q-Values):")
        qtable = self.agent.get_q_table(emb)
        sorted_q = sorted(qtable.items(), key=lambda x: x[1], reverse=True)

        for i, (src, val) in enumerate(sorted_q):
            marker = " ← SELECTED" if i == 0 else ""
            src_short = src.replace('Source', '')
            bar = "█" * int((val + 1) * 10) if val > -1 else ""
            print(f"    {src_short:<25s}  Q={val:+.4f}  {bar}{marker}")

        # Step 4: Query selected source
        action_idx = self.agent.select_action(emb, epsilon=0.0)  # No exploration
        source_name = self.agent.sources[action_idx]

        print(f"\n[3] QUERYING SOURCE: {source_name}")

        try:
            results = self.system.query_source(source_name, query)

            # Extract answer and confidence
            if isinstance(results, dict):
                answer = results.get('answer', str(results))
                confidence = results.get('confidence', 0.5)
            else:
                answer = str(results)
                confidence = 0.5

            # Step 5: Show output
            print(f"\n[4] OUTPUT:")
            answer_preview = answer[:400] + "..." if len(answer) > 400 else answer
            print(f"    {answer_preview}")
            print(f"\n    Confidence Score: {confidence:.2f}")

            # Step 6: Compute automatic reward
            print(f"\n[5] AUTOMATIC REWARD:")
            reward = RewardEvaluator.compute_reward(query, source_name, results)

            if reward > 0.8:
                reward_status = "EXCELLENT ✓✓"
            elif reward > 0.3:
                reward_status = "GOOD ✓"
            elif reward > -0.1:
                reward_status = "NEUTRAL ~"
            else:
                reward_status = "POOR ✗"

            print(f"    Reward: {reward:+.3f} ({reward_status})")

            # Step 7: LLM Evaluation
            print(f"\n[6] LLM EVALUATION:")
            print(f"    Evaluating answer quality...")

            evaluation = self.judge.evaluate_quality(query, answer, source_name)

            correctness = evaluation['correctness']
            relevance = evaluation['relevance']
            completeness = evaluation['completeness']
            quality = evaluation['quality']
            explanation = evaluation['explanation']

            print(f"\n    Scores (0.0 - 1.0):")
            print(f"      Correctness:  {correctness:.2f}  {'█' * int(correctness * 20)}")
            print(f"      Relevance:    {relevance:.2f}  {'█' * int(relevance * 20)}")
            print(f"      Completeness: {completeness:.2f}  {'█' * int(completeness * 20)}")
            print(f"      ─────────────────────────")
            print(f"      Overall Quality: {quality:.2f}  {'█' * int(quality * 20)}")

            if quality >= 0.7:
                quality_status = "✓ HIGH QUALITY"
            elif quality >= 0.4:
                quality_status = "~ ACCEPTABLE"
            else:
                quality_status = "✗ LOW QUALITY"

            print(f"\n    Assessment: {quality_status}")
            print(f"    Reason: {explanation}")

            # Step 8: Validity Check (Router vs LLM Judge)
            print(f"\n[7] VALIDITY CHECK:")
            print(f"    Router Confidence:  {confidence:.2f}")
            print(f"    LLM Quality Score:  {quality:.2f}")
            print(f"    Automatic Reward:   {reward:+.3f}")

            # Check if router decision was valid
            if quality >= 0.7 and reward > 0.3:
                validity = "✓ VALID - Good routing decision"
            elif quality < 0.4 and reward < 0:
                validity = "✓ VALID - Poor routing correctly penalized"
            elif quality >= 0.7 and reward < 0:
                validity = "⚠ MISMATCH - Good answer but low reward"
            elif quality < 0.4 and reward > 0.3:
                validity = "✗ INVALID - Poor answer but high reward"
            else:
                validity = "~ UNCERTAIN - Mixed signals"

            print(f"    Validity: {validity}")

            # Step 9: Store experience for RL training
            experience = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "query_type": qtype,
                "embedding": emb.tolist(),
                "source_selected": source_name,
                "action_idx": action_idx,
                "q_values": qtable,
                "confidence": confidence,
                "reward": reward,
                "llm_evaluation": {
                    "correctness": correctness,
                    "relevance": relevance,
                    "completeness": completeness,
                    "quality": quality,
                    "explanation": explanation
                },
                "validity": validity,
                "answer_preview": answer[:200]
            }

            # Append to log file
            with open(self.experience_log, 'a') as f:
                f.write(json.dumps(experience) + '\n')

            print(f"\n[8] EXPERIENCE STORED:")
            print(f"    ✓ Saved to {self.experience_log}")

            self.query_count += 1

            # Check if we have enough experiences to trigger training
            num_experiences = self.count_experiences()
            print(f"    Total experiences: {num_experiences}/{self.batch_size}")

            if num_experiences >= self.batch_size:
                print(f"\n[9] 🎓 TRIGGERING RETRAINING...")
                print(f"    {self.batch_size} experiences collected!")
                self.retrain_model()
            else:
                print(f"    Experiences until retraining: {self.batch_size - num_experiences}")

        except Exception as e:
            print(f"    ERROR: {e}")

        print("\n" + "=" * 80)

    def count_experiences(self):
        """Count number of experiences in log file"""
        if not os.path.exists(self.experience_log):
            return 0

        with open(self.experience_log, 'r') as f:
            return sum(1 for _ in f)

    def retrain_model(self):
        """Retrain model using logged experiences"""
        print(f"    Loading {self.batch_size} experiences from log...")

        experiences = []
        with open(self.experience_log, 'r') as f:
            for line in f:
                experiences.append(json.loads(line))

        # Take last batch_size experiences
        batch = experiences[-self.batch_size:]

        # Extract states, actions, rewards
        states = [e['embedding'] for e in batch]
        actions = [e['action_idx'] for e in batch]

        # Use blended rewards (30% automatic + 70% LLM quality)
        rewards = extract_rewards_from_experiences(batch, alpha=0.3, beta=0.7)

        # Convert to tensors
        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions)
        reward_batch = torch.FloatTensor(rewards)

        # Train
        print(f"    Training model on batch...")
        loss = self.agent.train_step(state_batch, action_batch, reward_batch)

        print(f"    ✓ Model updated (loss: {loss:.4f})")

        # Save model
        print(f"    💾 Saving model...")
        self.agent.save(self.model_path)
        print(f"    ✓ Model saved to {self.model_path}")

        # Reload model with updated weights
        print(f"    🔄 Reloading updated model...")
        self.agent.load(self.model_path)
        print(f"    ✓ Model reloaded with new weights")

        # Upload to S3
        if self.s3_sync:
            try:
                print(f"    📤 Uploading to S3...")
                self.s3_sync.upload('model', message=f"Dashboard retraining after {self.count_experiences()} experiences")
                print(f"    ✓ Model uploaded to S3")
            except Exception as e:
                print(f"    ⚠ S3 upload failed: {e}")

        # Clear the log file after successful training
        print(f"    🗑️  Clearing experience log...")
        with open(self.experience_log, 'w') as f:
            pass  # Empty the file
        print(f"    ✓ Log cleared, ready for next batch")

    def show_stats(self):
        """Show statistics from logged experiences"""
        if not os.path.exists(self.experience_log):
            print("\n⚠ No experiences logged yet")
            return

        print("\n" + "=" * 80)
        print("  EXPERIENCE LOG STATISTICS")
        print("=" * 80)

        experiences = []
        with open(self.experience_log, 'r') as f:
            for line in f:
                experiences.append(json.loads(line))

        total = len(experiences)
        if total == 0:
            print("\n  No experiences recorded yet")
            return

        # Calculate stats
        avg_reward = sum(e['reward'] for e in experiences) / total
        avg_quality = sum(e['llm_evaluation']['quality'] for e in experiences) / total
        avg_confidence = sum(e['confidence'] for e in experiences) / total

        source_counts = {}
        for e in experiences:
            src = e['source_selected']
            source_counts[src] = source_counts.get(src, 0) + 1

        print(f"\n  Total Experiences: {total}")
        print(f"  Average Reward: {avg_reward:+.3f}")
        print(f"  Average Quality: {avg_quality:.3f}")
        print(f"  Average Confidence: {avg_confidence:.3f}")

        print(f"\n  Source Distribution:")
        for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            src_short = src.replace('Source', '')
            bar = "█" * int(pct / 5)
            print(f"    {src_short:<25s}  {count:2d}  ({pct:5.1f}%)  {bar}")

        print(f"\n  Log file: {self.experience_log}")
        print("=" * 80)

    def run(self):
        """Run interactive loop"""

        print("\n" + "=" * 80)
        print("  INSTRUCTIONS")
        print("=" * 80)
        print("\n  Enter queries to see:")
        print("    • Router decision (source selection + Q-values)")
        print("    • Answer output + confidence score")
        print("    • Automatic reward computation")
        print("    • LLM quality evaluation")
        print("    • Validity check (router vs LLM)")
        print("    • Experience logged for batch RL training")
        print("\n  Commands:")
        print("    'quit' or 'exit' - Exit dashboard")
        print("    'stats' - Show experience statistics")
        print("    'examples' - Show sample queries")
        print("    'train' - Manually trigger model retraining (any number of experiences)")
        print("\n" + "=" * 80)

        while True:
            try:
                query = input("\n🔍 Your query: ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'q']:
                    self.show_stats()
                    print("\n👋 Goodbye!\n")
                    break

                if query.lower() == 'stats':
                    self.show_stats()
                    continue

                if query.lower() == 'examples':
                    self.show_examples()
                    continue

                if query.lower() == 'train':
                    # Manual training trigger
                    num_experiences = self.count_experiences()
                    if num_experiences == 0:
                        print("\n⚠ No experiences logged yet. Run some queries first.")
                    else:
                        print(f"\n🎓 Manual training triggered with {num_experiences} experiences...")
                        self.retrain_model()
                    continue

                self.process_query(query)

            except KeyboardInterrupt:
                self.show_stats()
                print("\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def show_examples(self):
        """Show example queries"""
        print("\n" + "=" * 80)
        print("  EXAMPLE QUERIES")
        print("=" * 80)

        examples = {
            "Drug Interactions": [
                "What drugs interact with warfarin?",
                "Can I take aspirin with ibuprofen?",
                "Does metformin interact with alcohol?"
            ],
            "Calculations": [
                "Calculate BMI for 85 kg and 1.80 m",
                "What is creatinine clearance for 60 year old, 70 kg, Cr 1.0?",
                "Ideal body weight for 175 cm male"
            ],
            "Concepts": [
                "How does insulin work?",
                "Explain the mechanism of ACE inhibitors",
                "What is the difference between Type 1 and Type 2 diabetes?"
            ],
            "Documents": [
                "What do the KRR papers say about ontologies?",
                "Knowledge representation in clinical decision support",
                "Medical reasoning approaches in the documents"
            ]
        }

        for category, queries in examples.items():
            print(f"\n  {category}:")
            for q in queries:
                print(f"    • {q}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    dashboard = InteractiveDashboard()
    dashboard.run()