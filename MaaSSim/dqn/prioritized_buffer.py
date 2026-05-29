import random
import numpy as np


class PrioritizedReplayBuffer:
    """Rank-based prioritized experience replay."""

    def __init__(self, capacity, alpha=0.6, priority_eps=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.data = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.write_idx = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.data[self.write_idx] = (state, action, reward, next_state, done)
        self.priorities[self.write_idx] = self.max_priority
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        # rank by priority descending
        active = self.priorities[:self.size]
        sorted_idx = np.argsort(-active)
        ranks = np.empty(self.size, dtype=np.float64)
        ranks[sorted_idx] = np.arange(1, self.size + 1, dtype=np.float64)

        # P(i) = (1/rank)^alpha, normalized
        probs = (1.0 / ranks) ** self.alpha
        probs /= probs.sum()

        # build CDF for stratified sampling
        cdf = np.cumsum(probs)

        indices = []
        segment = 1.0 / batch_size
        for batch_idx in range(batch_size):
            lower_bound = segment * batch_idx
            upper_bound = segment * (batch_idx + 1)
            sample_point = min(random.uniform(lower_bound, upper_bound), cdf[-1])
            idx = np.searchsorted(cdf, sample_point, side='left').item()
            idx = min(idx, self.size - 1)
            indices.append(idx)

        indices = np.array(indices, dtype=np.int64)
        batch = [self.data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors): # 
        for idx, td in zip(indices, td_errors):
            p = abs(td) + self.priority_eps
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p) 

    def iter_data(self): 
        for i in range(self.size):
            data_idx = (self.write_idx - self.size + i) % self.capacity
            yield data_idx, self.data[data_idx]

    def replace_data(self, data_idx, transition):
        self.data[data_idx] = transition

    def __len__(self):
        return self.size
