import os
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MaaSSim.dqn.actions import build_action_table
from MaaSSim.dqn.state import StateNormalizer, RewardRangeTracker
from MaaSSim.dqn.prioritized_buffer import PrioritizedReplayBuffer
from MaaSSim.dqn import DQN_DEFAULTS


class C51Network(nn.Module):
    """Dueling C51: V + A streams in logit space, softmax over atoms."""

    def __init__(self, state_dim, n_actions, hidden_dims, n_atoms):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        shared = []
        prev = state_dim
        for h in hidden_dims[:-1]:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.GELU())
            prev = h
        self.features = nn.Sequential(*shared)

        d = hidden_dims[-1]
        self.val_stream = nn.Sequential(
            nn.Linear(prev, d), nn.GELU(), nn.Linear(d, n_atoms))
        self.adv_stream = nn.Sequential(
            nn.Linear(prev, d), nn.GELU(), nn.Linear(d, n_actions * n_atoms))

    def forward(self, x):
        feat = self.features(x)
        val = self.val_stream(feat).unsqueeze(1)                    # (B, 1, n_atoms)
        adv = self.adv_stream(feat).view(-1, self.n_actions, self.n_atoms)  # (B, A, n_atoms)
        logits = val + adv - adv.mean(dim=1, keepdim=True)          # (B, A, n_atoms)
        return torch.softmax(logits, dim=-1)                        # probabilities


class DQNAgent:
    def __init__(self, config=None):
        cfg = {**DQN_DEFAULTS, **(config or {})}
        self.cfg = cfg

        self.action_table = build_action_table(cfg['action_n_steps'])
        n_actions = len(self.action_table)

        # C51 distributional params, these are placeholders until freeze_support()
        self.n_atoms = cfg['n_atoms']
        self.v_min = 0.0
        self.v_max = 1.0
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.reward_tracker = RewardRangeTracker()
        self._support_margin = cfg.get('c51_support_margin', 0.2)

        self.q_net = C51Network(cfg['state_dim'], n_actions, cfg['hidden_dims'], self.n_atoms)
        self.target_net = C51Network(cfg['state_dim'], n_actions, cfg['hidden_dims'], self.n_atoms)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg['lr']) # Adam is the simplest choice for C51, no need for RAdam or AdamW I think

        self.buffer = PrioritizedReplayBuffer(
            cfg['buffer_capacity'],
            alpha=cfg['per_alpha'],
            priority_eps=cfg.get('per_priority_eps', 1e-5))
        self.use_per = True

        self.normalizer = StateNormalizer(cfg['state_dim'])
        self.global_step = 0
        self._last_loss = None

    def select_action(self, state, epsilon): 
        with torch.no_grad(): 
            probs = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            q = (probs * self.support).sum(dim=-1)  # expected Q per action
        q_max = float(q.max())
        q_mean = float(q.mean())
        if random.random() < epsilon: # epsilon-greedy exploration
            action = random.randrange(len(self.action_table))
        else:
            action = int(q.argmax().item())
        # distributional std for the chosen action
        p_a = probs[action]
        q_std = float(((self.support ** 2 * p_a).sum() - q[action] ** 2).clamp(min=0).sqrt())
        return {
            'action': action,
            'q_chosen': float(q[action]),
            'q_max': q_max,
            'q_mean': q_mean,
            'q_std_chosen': q_std,
        }

    def get_distribution(self, state):
        """Full C51 output: probs (n_actions, n_atoms), support (n_atoms,)."""
        with torch.no_grad():
            probs = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        return probs.numpy(), self.support.numpy() 

    def train_step(self, beta=None):
        if len(self.buffer) < self.cfg['batch_size']: 
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.cfg['batch_size'], beta=beta or 0.4)

        B = len(states)
        states_t = torch.tensor(states)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards)
        next_states_t = torch.tensor(next_states)
        dones_t = torch.tensor(dones)
        weights_t = torch.tensor(weights)

        # current distribution for chosen actions: (B, n_atoms)
        curr_probs = self.q_net(states_t)  # (B, A, n_atoms)
        curr_dist = curr_probs[range(B), actions_t]  # (B, n_atoms)

        with torch.no_grad():
            # double DQN: online net picks action, target net provides distribution
            next_probs_online = self.q_net(next_states_t)
            next_q = (next_probs_online * self.support).sum(dim=-1)
            best_actions = next_q.argmax(dim=1)

            next_probs_target = self.target_net(next_states_t) 
            next_dist = next_probs_target[range(B), best_actions]  # (B, n_atoms)

            # project: Tz = r + gamma^n * support, clipped to [v_min, v_max]
            gamma = self.cfg['gamma'] ** self.cfg.get('n_step', 1)
            Tz = rewards_t.unsqueeze(1) + gamma * (1 - dones_t).unsqueeze(1) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)

            # distribute probability mass to neighboring atoms
            b = (Tz - self.v_min) / self.delta_z  # fractional atom index
            l = b.floor().long()
            u = b.ceil().long()
            # handle edge case where l == u (exact atom hit)
            l = l.clamp(0, self.n_atoms - 1)
            u = u.clamp(0, self.n_atoms - 1)

            target_dist = torch.zeros(B, self.n_atoms)
            # lower neighbor gets (u - b) share, upper gets (b - l) share
            target_dist.scatter_add_(1, l, next_dist * (u.float() - b))
            target_dist.scatter_add_(1, u, next_dist * (b - l.float()))
            # when l == u, both scatter_adds contribute 0; fix by adding full mass
            eq_mask = (l == u)
            target_dist.scatter_add_(1, l, next_dist * eq_mask.float())

        # cross-entropy loss
        log_curr = torch.log(curr_dist.clamp(min=1e-8))
        element_loss = -(target_dist * log_curr).sum(dim=-1)  # (B,)
        loss = (weights_t * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # PER priorities: per-sample cross-entropy (always positive)
        self.buffer.update_priorities(indices, element_loss.detach().cpu().numpy())

        self._last_loss = loss.item()
        return self._last_loss

    def sync_target(self):
        tau = self.cfg.get('target_sync_tau', 0.005)
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

    def freeze_support(self):
        self.reward_tracker.freeze(margin=self._support_margin)
        self.v_min = self.reward_tracker.v_min
        self.v_max = self.reward_tracker.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def store_transitions(self, transitions):
        for s, a, r, ns, d in transitions:
            self.buffer.push(s, a, r, ns, d)
            self.global_step += 1

    def observe_mc_range(self, mc_min, mc_max):
        self.reward_tracker.observe(mc_min)
        self.reward_tracker.observe(mc_max)

    def save(self, path, include_buffer=True):
        data = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.cfg,
            'normalizer': self.normalizer.state_dict(),
            'reward_tracker': self.reward_tracker.state_dict(),
        }
        if include_buffer and self.buffer.size > 0:
            data['buffer_data'] = self.buffer.data[:self.buffer.size]
            data['buffer_priorities'] = self.buffer.priorities[:self.buffer.size].copy()
            data['buffer_write_idx'] = self.buffer.write_idx
            data['buffer_size'] = self.buffer.size
            data['buffer_max_priority'] = self.buffer.max_priority
        # atomic write: temp file + replace so a crash mid-save (rip) can't corrupt the checkpoint
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
        try:
            os.close(tmp_fd)
            torch.save(data, tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self, path): # load entire checkpoint including buffer and normalizer 
        ckpt = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt['global_step']
        if 'normalizer' in ckpt:
            self.normalizer.load_state_dict(ckpt['normalizer'])
        if 'reward_tracker' in ckpt:
            self.reward_tracker.load_state_dict(ckpt['reward_tracker'])
            if self.reward_tracker.frozen:
                self.freeze_support()
        if 'buffer_data' in ckpt:
            buf = self.buffer
            for i, d in enumerate(ckpt['buffer_data']):
                buf.data[i] = d
            buf.priorities[:ckpt['buffer_size']] = ckpt['buffer_priorities']
            buf.write_idx = ckpt['buffer_write_idx']
            buf.size = ckpt['buffer_size']
            buf.max_priority = ckpt['buffer_max_priority']
