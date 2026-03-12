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


def build_state_vector(sim, platform, economics, match, params, beta_zero=0.0):
    """14-dim raw state from offer context. Normalization handled by StateNormalizer."""
    req_id = match['req_id']
    request = sim.inData.requests.loc[req_id]
    request_dist = float(request.dist)

    n_veh = len(platform.vehQ)
    n_req = max(len(platform.reqQ), 1)

    # cyclical time-of-day
    t_day = float(sim.t1)
    now = float(sim.env.now)
    angle = 2 * math.pi * now / t_day

    # -- base rider utility (split-independent) --
    behavioral = params.pudo.get('behavioral', {})
    beta_walk_time = behavioral.get('rider_beta_walk_time', 0.24)
    beta_wait = behavioral.get('rider_beta_wait', 0.22)
    walk_speed = params.speeds.walk
    walk_to_s = economics['walk_to_pickup'] / walk_speed
    walk_from_s = economics['walk_from_dropoff'] / walk_speed
    t_walk_min = (walk_to_s + walk_from_s) / 60.0
    t_wait_savings_min = min(walk_to_s, economics['wait_time']) / 60.0
    base_rider = beta_wait * t_wait_savings_min - beta_walk_time * t_walk_min - beta_zero

    # -- base driver utility (split-independent) --
    beta_time = behavioral.get('driver_beta_time', 0.004)
    beta_dist = params.pudo.operating_cost_per_km / 1000.0
    c_friction = behavioral.get('driver_C_friction', 0.15)
    is_pudo = (economics['delta_time_s'] != 0 or economics['delta_dist_m'] != 0)
    base_driver = (beta_time * economics['delta_time_s']
                   + beta_dist * economics['delta_dist_m']
                   - c_friction * is_pudo)

    state = np.array([
        economics['delta_pi'],
        economics['delta_dist_m'],
        economics['delta_time_s'],
        economics['d_walk_meters'],
        economics['wait_time'],
        economics['baseline_fare'],
        request_dist,
        beta_zero,
        min(n_veh / n_req, 5.0),
        float(n_veh),
        math.sin(angle),
        math.cos(angle),
        base_rider,
        base_driver,
    ], dtype=np.float32)

    return state
