import numpy as np

from MaaSSim.dqn.state import build_state_vector
from MaaSSim.dqn.rewards import compute_reward
from MaaSSim.dqn.actions import action_to_splits


class DQNIncentivePolicy:
    def __init__(self, agent):
        self.agent = agent
        self.epsilon = 1.0
        self._pending = []  # list of {state, action_idx, ...} awaiting outcome
        self._transitions_raw = []  # completed (state, action_idx, reward)

    def reset(self, epsilon):
        self.epsilon = epsilon
        self._pending = []
        self._transitions_raw = []

    def get_action(self, state):
        self.agent.normalizer.observe(state)
        norm_state = self.agent.normalizer.normalize(state)
        action_idx = self.agent.select_action(norm_state, self.epsilon)
        pi_r, pi_d = action_to_splits(action_idx, self.agent.action_table)

        self._pending.append({
            'state': norm_state,
            'action_idx': action_idx,
            'pi_r': pi_r,
            'pi_d': pi_d,
        })
        return pi_r, pi_d

    def record_outcome(self, delta_dist_m, accepted, alpha_d=None, alpha_r=None):
        if not self._pending:
            return
        entry = self._pending.pop(-1)
        reward = compute_reward(
            delta_dist_m, accepted,
            alpha_d=alpha_d, alpha_r=alpha_r,
            smoothed=self.agent.cfg.get('use_smoothed_reward', False),
        )
        self._transitions_raw.append({
            'state': entry['state'],
            'action_idx': entry['action_idx'],
            'reward': reward,
            'pi_r': entry['pi_r'],
            'pi_d': entry['pi_d'],
            'accepted': accepted,
            'delta_dist_m': delta_dist_m,
        })

    def get_episode_transitions(self):
        """Chain consecutive matches: next_state[i] = state[i+1], last is terminal."""
        n = len(self._transitions_raw)
        transitions = []
        for i, t in enumerate(self._transitions_raw):
            if i + 1 < n:
                next_state = self._transitions_raw[i + 1]['state']
                done = False
            else:
                next_state = np.zeros(self.agent.cfg['state_dim'], dtype=np.float32)
                done = True
            transitions.append((
                t['state'], t['action_idx'], t['reward'], next_state, done
            ))
        return transitions

    def get_episode_metrics(self):
        n = len(self._transitions_raw)
        if n == 0:
            return {'num_matches': 0, 'episode_reward': 0.0, 'acceptance_rate': 0.0,
                    'avg_pi_r': 0.0, 'avg_pi_d': 0.0, 'vkt_savings_m': 0.0}

        accepted = sum(1 for t in self._transitions_raw if t['accepted'])
        total_reward = sum(t['reward'] for t in self._transitions_raw)
        vkt = sum(t['delta_dist_m'] for t in self._transitions_raw if t['accepted'])

        return {
            'num_matches': n,
            'episode_reward': total_reward,
            'acceptance_rate': accepted / n,
            'avg_pi_r': sum(t['pi_r'] for t in self._transitions_raw) / n,
            'avg_pi_d': sum(t['pi_d'] for t in self._transitions_raw) / n,
            'vkt_savings_m': vkt,
        }
