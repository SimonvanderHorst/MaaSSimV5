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
        # populated by train.py from sim platform.revenue_model
        self.reward_type = 'commission'
        self.booking_fee = 0.0

    def reset(self, epsilon):
        self.epsilon = epsilon
        self._pending = []
        self._transitions_raw = []

    def get_action(self, state):
        self.agent.normalizer.observe(state)
        norm_state = self.agent.normalizer.normalize(state)
        result = self.agent.select_action(norm_state, self.epsilon)
        action_idx = result['action']
        pi_r, pi_d = action_to_splits(action_idx, self.agent.action_table)

        self._pending.append({
            'state': norm_state,
            'raw_state': state,
            'action_idx': action_idx,
            'pi_r': pi_r,
            'pi_d': pi_d,
            'q_chosen': result['q_chosen'],
            'q_max': result['q_max'],
            'q_mean': result['q_mean'],
            'q_std_chosen': result['q_std_chosen'],
        })
        return pi_r, pi_d

    def stash_context(self, ctx):
        if self._pending:
            self._pending[-1]['ctx'] = ctx

    def record_outcome(self, delta_pi, delta_dist_m, driver_accepted, alpha_d=None, alpha_r=None):
        if not self._pending:
            return
        entry = self._pending.pop(-1)
        # D2D skip action: no offer was made, reward = 0
        if entry['pi_r'] == -1:
            reward = 0.0
        else:
            reward = compute_reward(
                delta_pi, delta_dist_m,
                entry['pi_r'], entry['pi_d'], driver_accepted,
                c_ext=self.agent.cfg['c_ext'],
                alpha_d=alpha_d, alpha_r=alpha_r,
                smoothed=self.agent.cfg['use_smoothed_reward'],
                c_reject=self.agent.cfg.get('c_reject', 0.0),
                reward_type=self.reward_type,
                booking_fee=self.booking_fee,
            )
        t = {
            'state': entry['state'],
            'raw_state': entry.get('raw_state'),
            'action_idx': entry['action_idx'],
            'reward': reward,
            'pi_r': entry['pi_r'],
            'pi_d': entry['pi_d'],
            'driver_accepted': driver_accepted,
            'delta_pi': delta_pi,
            'delta_dist_m': delta_dist_m,
            'alpha_r': alpha_r,
            'alpha_d': alpha_d,
            'q_chosen': entry['q_chosen'],
            'q_max': entry['q_max'],
            'q_mean': entry['q_mean'],
            'q_std_chosen': entry['q_std_chosen'],
        }
        if 'ctx' in entry:
            t['ctx'] = entry['ctx']
        self._transitions_raw.append(t)

    def patch_rider_alpha(self, req_id, alpha_r):
        for t in reversed(self._transitions_raw):
            if t.get('ctx', {}).get('req_id') == req_id:
                t['alpha_r'] = alpha_r
                if self.agent.cfg['use_smoothed_reward'] and t['alpha_d'] is not None:
                    t['reward'] = compute_reward(
                        t['delta_pi'], t['delta_dist_m'],
                        t['pi_r'], t['pi_d'], t['driver_accepted'],
                        c_ext=self.agent.cfg['c_ext'],
                        alpha_d=t['alpha_d'], alpha_r=alpha_r,
                        smoothed=True,
                        c_reject=self.agent.cfg.get('c_reject', 0.0),
                        reward_type=self.reward_type,
                        booking_fee=self.booking_fee,
                    )
                break

    def get_episode_transitions(self):
        """Chain per-driver with n-step return accumulation."""
        zero = np.zeros(self.agent.cfg['state_dim'], dtype=np.float32)
        n_step = self.agent.cfg.get('n_step', 1)
        gamma = self.agent.cfg['gamma']

        # group by driver, preserving time order
        driver_chains = {}
        orphans = []
        for t in self._transitions_raw:
            vid = t.get('ctx', {}).get('veh_id')
            if vid is None:
                orphans.append(t)
            else:
                driver_chains.setdefault(vid, []).append(t)

        transitions = []
        self._mc_min = float('inf')
        self._mc_max = float('-inf')
        all_chains = list(driver_chains.values()) + [[o] for o in orphans]
        for chain in all_chains:
            L = len(chain)
            for i in range(L):
                # fold up to n_step rewards
                G = 0.0
                k = 0
                for k in range(min(n_step, L - i)):
                    G += (gamma ** k) * chain[i + k]['reward']
                # next_state is n steps ahead (or terminal)
                j = i + k + 1  # index of bootstrap state
                if j < L:
                    next_state = chain[j]['state']
                    done = False
                else:
                    next_state = zero
                    done = True
                transitions.append((
                    chain[i]['state'], chain[i]['action_idx'], G, next_state, done
                ))
            # MC return at position 0 for support calibration
            mc = 0.0
            for i in range(L - 1, -1, -1):
                mc = chain[i]['reward'] + gamma * mc
            if mc < self._mc_min:
                self._mc_min = mc
            if mc > self._mc_max:
                self._mc_max = mc
        return transitions

    _MATCH_DIMS = ['s_delta_pi', 's_delta_dist_m', 's_delta_time_s', 's_d_walk',
                   's_wait_time', 's_baseline_fare', 's_request_dist', 's_beta_zero',
                   's_base_rider_utility', 's_base_driver_utility']
    _FLEET_DIMS = ['s_veh_req_ratio', 's_sin_tod', 's_cos_tod']
    _FLEET_TAIL = ['s_fleet_util']

    @property
    def _state_dims(self):
        if self.agent.cfg.get('use_fleet_state', True):
            return (self._MATCH_DIMS[:8] + self._FLEET_DIMS +
                    self._MATCH_DIMS[8:] + self._FLEET_TAIL)
        return self._MATCH_DIMS

    def get_match_rows(self, episode):
        rows = []
        for i, t in enumerate(self._transitions_raw):
            row = {
                'episode': episode,
                'match_idx': i,
                'action_idx': t['action_idx'],
                'pi_r': t['pi_r'],
                'pi_d': t['pi_d'],
                'driver_accepted': t['driver_accepted'],
                'reward': t['reward'],
                'delta_pi': t['delta_pi'],
                'delta_dist_m': t['delta_dist_m'],
                'alpha_r': t.get('alpha_r'),
                'alpha_d': t.get('alpha_d'),
                'q_chosen': t['q_chosen'],
                'q_max': t['q_max'],
                'q_mean': t['q_mean'],
                'q_std_chosen': t['q_std_chosen'],
            }
            if 'ctx' in t:
                row.update(t['ctx'])
            raw = t.get('raw_state')
            if raw is not None:
                for j, name in enumerate(self._state_dims):
                    row[name] = float(raw[j])
            rows.append(row)
        return rows

    def get_episode_metrics(self):
        n = len(self._transitions_raw)
        if n == 0:
            return {'num_matches': 0, 'episode_reward': 0.0, 'acceptance_rate': 0.0,
                    'avg_pi_r': 0.0, 'avg_pi_d': 0.0, 'vkt_savings_m': 0.0}

        accepted = sum(1 for t in self._transitions_raw if t['driver_accepted'])
        total_reward = sum(t['reward'] for t in self._transitions_raw)
        vkt = sum(t['delta_dist_m'] for t in self._transitions_raw if t['driver_accepted'])
        margin = sum(t['delta_pi'] * (1 - t['pi_r'] - t['pi_d'])
                     for t in self._transitions_raw if t['driver_accepted'])

        return {
            'num_matches': n,
            'episode_reward': total_reward,
            'acceptance_rate': accepted / n,
            'avg_pi_r': sum(t['pi_r'] for t in self._transitions_raw if t['pi_r'] != -1) / max(1, sum(1 for t in self._transitions_raw if t['pi_r'] != -1)),
            'avg_pi_d': sum(t['pi_d'] for t in self._transitions_raw if t['pi_d'] != -1) / max(1, sum(1 for t in self._transitions_raw if t['pi_d'] != -1)),
            'vkt_savings_m': vkt,
            'margin_capture': margin,
            'avg_q_chosen': sum(t['q_chosen'] for t in self._transitions_raw) / n,
            'avg_q_max': sum(t['q_max'] for t in self._transitions_raw) / n,
            'avg_q_mean': sum(t['q_mean'] for t in self._transitions_raw) / n,
            'avg_q_std_chosen': sum(t['q_std_chosen'] for t in self._transitions_raw) / n,
        }
