import random
import collections
import numpy as np
import torch.nn as nn


class QNetwork(nn.Module):
    """Dueling DQN: splits Q into V(s) + A(s,a) streams."""

    def __init__(self, state_dim, n_actions, hidden_dims):
        super().__init__()
        # shared feature layers (all hidden dims except last)
        shared = []
        prev = state_dim
        for h in hidden_dims[:-1]:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.ReLU())
            prev = h
        self.features = nn.Sequential(*shared)

        d = hidden_dims[-1]
        self.val_stream = nn.Sequential(
            nn.Linear(prev, d), nn.ReLU(), nn.Linear(d, 1))
        self.adv_stream = nn.Sequential(
            nn.Linear(prev, d), nn.ReLU(), nn.Linear(d, n_actions))

    def forward(self, x):
        feat = self.features(x)
        val = self.val_stream(feat)
        adv = self.adv_stream(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)
