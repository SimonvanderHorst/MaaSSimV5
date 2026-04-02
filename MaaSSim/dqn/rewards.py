def compute_reward(delta_pi, delta_dist_m, pi_r, pi_d, accepted, c_ext,
                   alpha_d=None, alpha_r=None, smoothed=False, c_reject=0.0):
    margin = delta_pi * (1 - pi_r - pi_d)
    externality = c_ext * delta_dist_m / 1000.0
    raw = margin + externality

    if smoothed and alpha_d is not None and alpha_r is not None:
        p_accept = alpha_d * alpha_r
        return float(p_accept * raw - (1 - p_accept) * c_reject * externality)
    return float(raw) if accepted else 0.0
