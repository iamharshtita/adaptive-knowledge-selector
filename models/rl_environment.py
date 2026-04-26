# Experience replay buffer for training the RL agent

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Stores past experiences so we can sample random batches for training.

    Why random sampling? If we train on experiences in order, the network
    overfits to recent patterns. Random sampling breaks correlations and
    makes training more stable.
    """

    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        """Save one experience (what we did and what reward we got)"""
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        """Pull out a random batch of experiences for training"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Unpack into separate arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])

        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Empty the buffer"""
        self.buffer.clear()