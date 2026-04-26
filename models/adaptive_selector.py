# DQN agent that learns to pick the right knowledge source for each query

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class DQN(nn.Module):
    """Neural net that takes query embedding and outputs Q-value for each source"""

    def __init__(self, input_dim, num_sources):
        super(DQN, self).__init__()

        # Deep network with dropout to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_sources)  # one Q-value per source
        )

    def forward(self, x):
        return self.network(x)


class AdaptiveSelector:
    """
    RL agent that learns which knowledge source to query based on the question.

    Training uses DQN with two tricks:
    1. Epsilon-greedy: sometimes pick random source to explore
    2. Target network: stable learning target that updates slowly
    """

    def __init__(self, input_dim=384, num_sources=4, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Policy net: this is what we train
        self.policy_net = DQN(input_dim, num_sources).to(self.device)

        # Target net: copy of policy that we use for stable Q-value targets
        # Gets updated less frequently to avoid chasing a moving target
        self.target_net = DQN(input_dim, num_sources).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Map indices to actual source names
        self.sources = {
            0: "KnowledgeGraphSource",
            1: "ToolAPISource",
            2: "LLMSource",
            3: "PDFKnowledgeSource"
        }

        self.gamma = 0.95  # how much we care about future rewards

    def select_action(self, state, epsilon=0.0):
        """
        Pick which source to query.

        If epsilon > 0, we sometimes pick randomly to explore.
        Otherwise pick the source with highest predicted Q-value.
        """
        if np.random.random() < epsilon:
            return np.random.randint(len(self.sources))

        # Get Q-values and pick best
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self, states, actions, rewards):
        """
        Update the network based on a batch of experiences.

        We're doing simple immediate-reward learning here:
        teach the network that Q(query, source) should equal the reward we got.
        """
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)

        # What Q-values did the network predict for the actions we took?
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # What should those Q-values have been? Just the reward we got.
        # (single-step episodic setup, no future states)
        target_q = reward_batch

        # Train to minimize difference
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy current policy weights to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q_table(self, state):
        """Get predicted Q-values for all sources given a query embedding"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()

        return {self.sources[i]: float(q_values[i]) for i in range(len(self.sources))}

    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model from disk"""
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False