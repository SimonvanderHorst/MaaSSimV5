def compute_reward(delta_dist_m, accepted, alpha_d=None, alpha_r=None, smoothed=False):
    if smoothed and alpha_d is not None and alpha_r is not None:
        return float(alpha_d * alpha_r * delta_dist_m)
    return float(delta_dist_m) if accepted else 0.0
