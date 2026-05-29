import math
import numpy as np


class StateNormalizer:
    """Welford online mean/std — collects during warmup, freezes before training."""

    def __init__(self, dim):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.frozen = False
        self._std = None

    def observe(self, x):
        if self.frozen:
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def freeze(self):
        self.frozen = True
        variance = self.M2 / max(self.n, 1)
        self._std = np.sqrt(variance).astype(np.float32)
        self._std[self._std < 1e-6] = 1.0  # avoid div-by-zero on constant dims
        self.mean = self.mean.astype(np.float32)

    def normalize(self, x):
        if not self.frozen:
            return x.astype(np.float32)
        return ((x - self.mean) / self._std).astype(np.float32)

    def state_dict(self):
        return {'n': self.n, 'mean': self.mean, 'M2': self.M2,
                'frozen': self.frozen, '_std': self._std}

    def load_state_dict(self, d):
        self.n = d['n']
        self.mean = d['mean']
        self.M2 = d['M2']
        self.frozen = d['frozen']
        self._std = d['_std']


class RewardRangeTracker:
    """Track reward min/max during warmup, freeze to set C51 support range."""

    def __init__(self):
        self.frozen = False
        self.min = float('inf')
        self.max = float('-inf')
        self.v_min = None
        self.v_max = None

    def observe(self, reward):
        if reward < self.min:
            self.min = reward
        if reward > self.max:
            self.max = reward

    def freeze(self, margin=0.2):
        self.frozen = True
        span = self.max - self.min
        pad = span * margin
        self.v_min = math.floor((self.min - pad) * 2) / 2
        self.v_max = math.ceil((self.max + pad) * 2) / 2

    def state_dict(self):
        return {'min': self.min, 'max': self.max, 'frozen': self.frozen,
                'v_min': self.v_min, 'v_max': self.v_max}

    def load_state_dict(self, d):
        self.min = d['min']
        self.max = d['max']
        self.frozen = d['frozen']
        self.v_min = d['v_min']
        self.v_max = d['v_max']


def build_state_vector(sim, platform, economics, match, params, beta_zero, batch_counts,
                       use_fleet_state=True):
    """Raw state from offer context. 14-dim with fleet state, 10-dim without."""
    req_id = match['req_id']
    request = sim.inData.requests.loc[req_id]
    request_dist = float(request.dist)

    # -- base rider utility (split-independent) --
    behavioral = params.pudo.get('behavioral', {})
    beta_walk_time = behavioral.get('rider_beta_walk_time', 0.24)
    beta_wait = behavioral.get('rider_beta_wait', 0.22)
    walk_speed = params.speeds.walk
    walk_to_s = economics['walk_to_pickup'] / walk_speed
    walk_from_s = economics['walk_from_dropoff'] / walk_speed
    t_walk_min = (walk_to_s + walk_from_s) / 60.0
    t_wait_savings_min = min(walk_to_s, economics['wait_time']) / 60.0
    base_rider_utility = beta_wait * t_wait_savings_min - beta_walk_time * t_walk_min - beta_zero

    # -- base driver utility (split-independent) --
    beta_time = behavioral.get('driver_beta_time', 0.004)
    beta_dist = params.pudo.operating_cost_per_km / 1000.0
    c_friction = behavioral.get('driver_C_friction', 0.15)
    is_pudo = (economics['delta_time_s'] != 0 or economics['delta_dist_m'] != 0)
    base_driver_utility = (beta_time * economics['delta_time_s']
                   + beta_dist * economics['delta_dist_m']
                   - c_friction * is_pudo)

    match_features = [
        economics['delta_pi'],
        economics['delta_dist_m'],
        economics['delta_time_s'],
        economics['d_walk_meters'],
        economics['wait_time'],
        economics['baseline_fare'],
        request_dist,
        beta_zero,
        base_rider_utility,
        base_driver_utility,
    ]

    if use_fleet_state:
        n_available_veh = batch_counts['n_available_veh']
        n_req = max(batch_counts['n_req'], 1)
        angle = 2 * math.pi * float(sim.env.now) / float(sim.t1)
        total_fleet = len(sim.vehicles)
        fleet_util = (total_fleet - n_available_veh) / max(total_fleet, 1)

        match_features[8:8] = [
            min(n_available_veh / n_req, 5.0),
            math.sin(angle),
            math.cos(angle),
        ]
        match_features.append(fleet_util)

    return np.array(match_features, dtype=np.float32)
