#!/usr/bin/env python3
"""
Manual RL Training Script
Trains the model on all logged experiences regardless of batch size.
"""

import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from models.adaptive_selector import AdaptiveSelector
from utils.s3_sync import S3TeamSync
from utils.reward_blending import extract_rewards_from_experiences


def manual_train():
    """Train model on all logged experiences"""

    experience_log = "data/rl_selector/online_experiences.jsonl"
    model_path = "data/rl_selector/adaptive_dqn.pth"

    if not os.path.exists(experience_log):
        print("❌ No experience log found at", experience_log)
        return

    # Count experiences
    with open(experience_log, 'r') as f:
        experiences = [json.loads(line) for line in f]

    num_exp = len(experiences)

    if num_exp == 0:
        print("❌ Experience log is empty. Run some queries first.")
        return

    print("="*80)
    print(f"  MANUAL RL TRAINING")
    print("="*80)
    print(f"\n  Found {num_exp} experiences in log")

    # Load model
    agent = AdaptiveSelector(input_dim=384, num_sources=4, lr=0.001)
    if not agent.load(model_path):
        print(f"❌ Could not load model from {model_path}")
        return

    print(f"  ✓ Model loaded from {model_path}")

    # Extract training data
    states = [e['embedding'] for e in experiences]
    actions = [e['action_idx'] for e in experiences]

    # Use blended rewards (30% automatic + 70% LLM quality)
    rewards = extract_rewards_from_experiences(experiences, alpha=0.3, beta=0.7)

    # Count how many experiences used LLM evaluation
    llm_count = sum(1 for e in experiences if e.get('llm_evaluation', {}).get('quality', 0) > 0)
    print(f"  Using blended rewards: {llm_count}/{num_exp} with LLM evaluation")

    state_batch = torch.FloatTensor(states)
    action_batch = torch.LongTensor(actions)
    reward_batch = torch.FloatTensor(rewards)

    # Train
    print(f"\n  🎓 Training on {num_exp} experiences...")
    loss = agent.train_step(state_batch, action_batch, reward_batch)
    print(f"  ✓ Training complete (loss: {loss:.4f})")

    # Save
    print(f"\n  💾 Saving model...")
    agent.save(model_path)
    print(f"  ✓ Model saved to {model_path}")

    # Reload to verify
    print(f"\n  🔄 Reloading model to verify...")
    agent.load(model_path)
    print(f"  ✓ Model reloaded successfully")

    # Upload to S3
    try:
        s3_sync = S3TeamSync()
        print(f"\n  📤 Uploading to S3...")
        s3_sync.upload('model', message=f"Manual training on {num_exp} experiences")
        print(f"  ✓ Model uploaded to S3")
    except Exception as e:
        print(f"  ⚠ S3 upload failed: {e}")

    # Clear log
    print(f"\n  🗑️  Clearing experience log...")
    with open(experience_log, 'w') as f:
        pass
    print(f"  ✓ Log cleared")

    print("\n" + "="*80)
    print("  ✅ TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    manual_train()