def compute_reward(delta_pi, delta_dist_m, pi_r, pi_d, driver_accepted, c_ext,
                   alpha_d=None, alpha_r=None, smoothed=False, c_reject=0.0,
                   reward_type='commission', booking_fee=0.0):
    if reward_type == 'commission':
        match_value = delta_pi * (1 - pi_r - pi_d)
    elif reward_type == 'flat_fee':
        match_value = booking_fee
    elif reward_type == 'social_welfare':
        match_value = delta_pi
    elif reward_type == 'driver_retention':
        match_value = pi_d * delta_pi
    elif reward_type == 'green':
        match_value = 0.0
    else:
        match_value = delta_pi * (1 - pi_r - pi_d)

    externality = c_ext * delta_dist_m / 1000.0
    raw = match_value + externality

    if smoothed and alpha_d is not None and alpha_r is not None:
        p_accept = alpha_d * alpha_r
        return float(p_accept * raw - (1 - p_accept) * c_reject * externality)
    return float(raw) if driver_accepted else 0.0
