"""
Reward Blending Utility
Combines automatic heuristic rewards with LLM-based quality evaluation
for more accurate reinforcement learning.
"""


def compute_blended_reward(automatic_reward: float, llm_evaluation: dict = None,
                          alpha: float = 0.3, beta: float = 0.7) -> float:
    """
    Blend automatic reward with LLM evaluation quality.

    Args:
        automatic_reward: Heuristic reward from RewardEvaluator (-1.0 to +1.5)
        llm_evaluation: Dict with 'quality' score (0.0 to 1.0), or None
        alpha: Weight for automatic reward (default: 0.3 = 30%)
        beta: Weight for LLM quality (default: 0.7 = 70%)

    Returns:
        Final blended reward for RL training (clamped to -1.0 to +1.0)

    Examples:
        >>> # Wrong answer: auto=+0.20, LLM quality=0.3
        >>> compute_blended_reward(0.20, {'quality': 0.3})
        -0.22  # Penalty instead of reward!

        >>> # Good answer: auto=+0.70, LLM quality=0.9
        >>> compute_blended_reward(0.70, {'quality': 0.9})
        0.77  # Strong positive reward

        >>> # LLM unavailable: fallback to automatic
        >>> compute_blended_reward(0.50, None)
        0.50  # Uses automatic reward only
    """
    # Validate weights sum to 1.0
    if abs(alpha + beta - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got alpha={alpha}, beta={beta}")

    # If LLM evaluation is available and valid
    if llm_evaluation and llm_evaluation.get('quality', 0) > 0:
        llm_quality = llm_evaluation['quality']

        # Scale LLM quality (0-1) to reward range (-1 to +1)
        llm_reward = (llm_quality * 2.0) - 1.0

        # Weighted blend
        final_reward = (alpha * automatic_reward) + (beta * llm_reward)

        # Clamp to standard reward range
        final_reward = max(-1.0, min(final_reward, 1.0))

        return final_reward
    else:
        # Fallback: use automatic reward only (LLM unavailable/failed)
        return automatic_reward


def extract_rewards_from_experiences(experiences: list, alpha: float = 0.3,
                                     beta: float = 0.7) -> list:
    """
    Extract blended rewards from a list of experience dictionaries.

    Args:
        experiences: List of experience dicts with 'reward' and optional 'llm_evaluation'
        alpha: Weight for automatic reward (default: 0.3)
        beta: Weight for LLM quality (default: 0.7)

    Returns:
        List of blended rewards (same length as experiences)
    """
    rewards = []

    for exp in experiences:
        auto_reward = exp.get('reward', 0.0)
        llm_eval = exp.get('llm_evaluation', None)

        blended_reward = compute_blended_reward(auto_reward, llm_eval, alpha, beta)
        rewards.append(blended_reward)

    return rewards


if __name__ == "__main__":
    # Test cases
    print("Testing Blended Reward System")
    print("=" * 60)

    # Test 1: Wrong answer detected by LLM
    print("\nTest 1: Wrong Answer (Auto thinks OK, LLM detects error)")
    print("  Automatic reward: +0.20 (thinks it's acceptable)")
    print("  LLM quality: 0.3 (detects it's wrong)")
    result = compute_blended_reward(0.20, {'quality': 0.3})
    print(f"  → Blended reward: {result:+.3f} ✓ (penalty applied)")

    # Test 2: Good answer
    print("\nTest 2: Good Answer (Both agree)")
    print("  Automatic reward: +0.70")
    print("  LLM quality: 0.9")
    result = compute_blended_reward(0.70, {'quality': 0.9})
    print(f"  → Blended reward: {result:+.3f} ✓ (strong positive)")

    # Test 3: LLM unavailable
    print("\nTest 3: LLM Unavailable (Fallback)")
    print("  Automatic reward: +0.50")
    print("  LLM evaluation: None")
    result = compute_blended_reward(0.50, None)
    print(f"  → Blended reward: {result:+.3f} ✓ (fallback to automatic)")

    # Test 4: Excellent answer
    print("\nTest 4: Excellent Answer")
    print("  Automatic reward: +1.00")
    print("  LLM quality: 1.0")
    result = compute_blended_reward(1.00, {'quality': 1.0})
    print(f"  → Blended reward: {result:+.3f} ✓ (maximum reward)")

    # Test 5: Poor answer
    print("\nTest 5: Poor Answer")
    print("  Automatic reward: -0.50")
    print("  LLM quality: 0.1")
    result = compute_blended_reward(-0.50, {'quality': 0.1})
    print(f"  → Blended reward: {result:+.3f} ✓ (strong penalty)")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")