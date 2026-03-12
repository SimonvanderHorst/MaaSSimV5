import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MaaSSim.dqn.actions import build_action_table
from MaaSSim.dqn.state import StateNormalizer
from MaaSSim.dqn import DQN_DEFAULTS


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dims):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


class DQNAgent:
    def __init__(self, config=None):
        cfg = {**DQN_DEFAULTS, **(config or {})}
        self.cfg = cfg

        self.action_table = build_action_table(cfg['action_n_steps'])
        n_actions = len(self.action_table)

        self.q_net = QNetwork(cfg['state_dim'], n_actions, cfg['hidden_dims'])
        self.target_net = QNetwork(cfg['state_dim'], n_actions, cfg['hidden_dims'])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg['lr'])
        self.buffer = ReplayBuffer(cfg['buffer_capacity'])

        self.normalizer = StateNormalizer(cfg['state_dim'])
        self.global_step = 0
        self._last_loss = None

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(self.action_table))
        with torch.no_grad():
            q = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def train_step(self):
        if len(self.buffer) < self.cfg['batch_size']:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg['batch_size'])

        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions).unsqueeze(1)
        rewards_t = torch.tensor(rewards)
        next_states_t = torch.tensor(next_states)
        dones_t = torch.tensor(dones)

        # current Q values for chosen actions
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # target: r + gamma * max Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.cfg['gamma'] * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        return self._last_loss

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def store_transitions(self, transitions):
        for s, a, r, ns, d in transitions:
            self.buffer.push(s, a, r, ns, d)
            self.global_step += 1

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.cfg,
            'normalizer': self.normalizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt['global_step']
        if 'normalizer' in ckpt:
            self.normalizer.load_state_dict(ckpt['normalizer'])
