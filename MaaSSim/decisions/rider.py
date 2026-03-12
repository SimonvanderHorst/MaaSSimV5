################################################################################
# Module: decisions/rider.py
# Description: Rider/traveller agent decision functions
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
import random
from math import exp
from numpy.random.mtrand import choice

from MaaSSim.decisions._helpers import _safe_sigmoid, _downgrade_offer_to_d2d


def f_platform_opt_out(*args, **kwargs):
    pax = kwargs.get('pax', None)
    return pax.request.platform == -1


def f_out(*args, **kwargs):
    # it uses pax_exp of a passenger populated in previous run
    # prev_exp is a pd.Series of this pd.DataFrame
    # pd.DataFrame(columns=['wait_pickup','wait_match','tt'])
    # returns boolean True if passanger decides to opt out
    prev_exp = kwargs.get('prev_exp', None)
    if prev_exp is None:
        # no prev exepreince
        return False
    else:
        if prev_exp.iloc[0].outcome == 1:
            return False
        else:
            return True


def f_mode(*args, **kwargs):
    # returns boolean True if passenger decides not to use MaaS (bad offer)
    offer = kwargs.get('offer', None)
    delta = 0.5
    trip = kwargs.get('trip')

    pass_walk_time = trip.pass_walk_time
    veh_pickup_time = trip.sim.skims.ride.T[trip.veh.pos][trip.request.origin]
    pass_matching_time = trip.sim.env.now - trip.t_matching
    tt = trip.request.ttrav
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt.seconds > delta


def _d2d_ratio_check(sim, traveller, offer):
    """Standard D2D rider quality check (same as f_mode).

    Returns True if rider REJECTS (overhead/trip ratio > 0.5).
    Used as Stage 1 baseline for both D2D and PUDO offers.
    """
    pass_walk_time = sim.skims.walk[traveller.pax.pos][traveller.request.origin]
    veh_pickup_time = offer.get('wait_time', 0)
    pass_matching_time = sim.env.now - traveller.t_matching
    tt = traveller.request.ttrav
    tt_seconds = tt.total_seconds() if hasattr(tt, 'total_seconds') else float(tt)
    if tt_seconds <= 0:
        return False
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt_seconds > 0.5


def f_pudo_rider_mode(*args, **kwargs):
    """Rider acceptance: D2D baseline check + PUDO behavioral increment.

    Stage 1 (D2D ratio check): Skipped for PUDO offers that already passed
    the early S1 gate in ``create_pudo_offer()`` (flagged with
    ``_rider_s1_passed=True``). Still runs for non-PUDO offers.

    Stage 2 (PUDO only): Sigmoid on PUDO-incremental utility:
    ΔU_PUDO = π_r · ΔΠ + β_wait · t_wait_savings − β_walk_time · t_walk − β_zero

    Where:
      ΔΠ       = fare margin (EUR) from PUDO route savings
      t_wait_savings = min(walk_to_pickup_time, driver_dispatch_time) in minutes
      t_walk   = total walking time (to pickup + from dropoff) in minutes
      β_zero   = fixed hassle cost of any PUDO offer (EUR)

    Returns True if rider REJECTS (opts out), False if rider ACCEPTS.
    Called from traveller.py with traveller=self (a PassengerAgent).
    """
    traveller = kwargs.get('traveller', None)
    sim = traveller.sim
    params = sim.params

    # Get the offer (single-platform case)
    if len(traveller.offers) == 0:
        return False

    platform_id, offer = list(traveller.offers.items())[0]

    # ── Stage 1: D2D baseline ratio check (same for ALL offers) ──
    # For PUDO offers, already checked in create_pudo_offer() (early S1 gate)
    if not offer.get('_rider_s1_passed', False):
        if _d2d_ratio_check(sim, traveller, offer):
            return True  # Rider rejects on D2D grounds (overhead too high)

    # ── Stage 2: PUDO-specific behavioral increment ──
    # Non-PUDO offers: passed Stage 1, accept
    if not offer.get('pudo_enabled', False):
        return False

    behavioral = params.pudo.get('behavioral', {})
    if not behavioral.get('enabled', False):
        return False  # Behavioral model disabled, accept (Stage 1 already passed)

    # D2D fallback (no savings, no walking): no PUDO-specific disutility, accept
    savings = offer.get('savings', 0)
    d_walk = offer.get('d_walk_meters', 0)
    if savings == 0 and d_walk == 0:
        return False

    # Get behavioral parameters
    pi_r = offer.get('_pi_r', params.pudo.static_rider_incentive)
    # Backward compat: convert legacy distance-based beta to time-based
    if 'rider_beta_walk_time' not in behavioral and 'rider_beta_walk' in behavioral:
        walk_speed = params.speeds.walk  # m/s
        beta_walk_time = behavioral['rider_beta_walk'] * walk_speed * 60  # EUR/m -> EUR/min
        beta_wait = 0.0  # legacy formula had no wait savings term
    else:
        beta_walk_time = behavioral.get('rider_beta_walk_time', 0.24)  # EUR/min
        beta_wait = behavioral.get('rider_beta_wait', 0.22)            # EUR/min
    # Rider heterogeneity: per-rider beta_zero overrides config default
    beta_zero = offer.get('_rider_beta_zero', behavioral.get('rider_beta_zero', 0.0))  # EUR
    sigmoid_scale = behavioral.get('rider_sigmoid_scale', 1.0)        # mu

    # Get fare margin and walking/wait data from offer
    delta_pi = offer.get('delta_pi', 0)
    walk_to_pickup = offer.get('walk_to_pickup', 0)        # meters
    walk_from_dropoff = offer.get('walk_from_dropoff', 0)  # meters
    walk_speed = params.speeds.walk                         # m/s
    wait_time = offer.get('wait_time', 0)                   # seconds

    # Walking times (seconds -> minutes)
    walk_to_pickup_time_s = walk_to_pickup / walk_speed
    walk_from_dropoff_time_s = walk_from_dropoff / walk_speed
    t_walk_min = (walk_to_pickup_time_s + walk_from_dropoff_time_s) / 60.0

    # Walk-wait overlap: rider walks to pickup DURING driver dispatch
    t_wait_savings_min = min(walk_to_pickup_time_s, wait_time) / 60.0

    # Compute PUDO-incremental rider utility (thesis formula)
    U_rider = (pi_r * delta_pi
               + beta_wait * t_wait_savings_min
               - beta_walk_time * t_walk_min
               - beta_zero)

    # Sigmoid: P = sigma(mu * U)
    # At U=0, P = 50% (neutral — PUDO equivalent to D2D)
    sigmoid_input = sigmoid_scale * U_rider
    alpha_r = _safe_sigmoid(sigmoid_input)

    # Probabilistic acceptance
    accepts = random.random() < alpha_r

    # Store utility components on offer for logging
    offer['_rider_utility'] = U_rider
    offer['_rider_alpha'] = alpha_r
    offer['_rider_accepted'] = accepts

    # Accumulate rider behavioral data on sim-level log (for post-hoc analysis)
    if not hasattr(sim, '_rider_behavioral_log'):
        sim._rider_behavioral_log = []
    sim._rider_behavioral_log.append({
        'pax_id': int(traveller.id),
        'req_id': int(offer.get('req_id', -1)),
        'veh_id': int(offer.get('veh_id', -1)),
        'rider_utility': float(U_rider),
        'rider_alpha': float(alpha_r),
        'rider_accepted': bool(accepts),
        # Input quantities
        'delta_pi': float(delta_pi),
        'd_walk_meters': float(d_walk),
        'savings': float(savings),
        'wait_time_s': float(wait_time),
        'walk_to_pickup_m': float(walk_to_pickup),
        'walk_from_dropoff_m': float(walk_from_dropoff),
        # Computed intermediaries
        't_walk_min': float(t_walk_min),
        't_wait_savings_min': float(t_wait_savings_min),
        # Utility decomposition
        'u_fare_component': float(pi_r * delta_pi),
        'u_wait_component': float(beta_wait * t_wait_savings_min),
        'u_walk_component': float(-beta_walk_time * t_walk_min),
        'u_zero_component': float(-beta_zero),
        'sigmoid_scale': float(sigmoid_scale),
        'sigmoid_input': float(sigmoid_input),
        'd2d_fallback_triggered': False,
    })

    # D2D fallback: rider rejected PUDO, try D2D offer instead
    if not accepts and params.pudo.get('d2d_fallback', False) and savings > 0:
        plat = sim.plats[platform_id]
        _downgrade_offer_to_d2d(sim, plat, offer)
        offer['_rider_pudo_rejected'] = True
        offer['_rider_pudo_utility'] = U_rider
        offer['_rider_pudo_alpha'] = alpha_r

        # Mark the rider log entry as fallback-triggered
        if sim._rider_behavioral_log and sim._rider_behavioral_log[-1]['pax_id'] == int(traveller.id):
            sim._rider_behavioral_log[-1]['d2d_fallback_triggered'] = True

        # D2D fallback already passed Stage 1 (ratio check), so accept
        # (the rider only rejected the PUDO increment, not the base service)
        if params.pudo.get('decision_log_level', 'off') != 'off':
            sim.logger.info(
                "PUDO_RIDER_D2D_FALLBACK: pax={} req={} pudo_U={:.4f} pudo_alpha={:.4f} "
                "d2d_fallback=accepted".format(
                    traveller.id, offer.get('req_id', '?'),
                    U_rider, alpha_r))

        return False  # Accept D2D fallback (Stage 1 already passed)

    # Log if decision logging enabled
    if params.pudo.get('decision_log_level', 'off') != 'off':
        sim.logger.info(
            "PUDO_RIDER: pax={} req={} U={:.4f} sig_in={:.4f} alpha={:.4f} accepted={} "
            "delta_pi={:.4f} d_walk_m={:.1f}".format(
                traveller.id, offer.get('req_id', '?'),
                U_rider, sigmoid_input, alpha_r, accepts, delta_pi, d_walk))

    return not accepts  # True = reject


def f_platform_choice(*args, **kwargs):
    traveller = kwargs.get('traveller')
    sim = traveller.sim

    betas = sim.params.platform_choice
    offers = traveller.offers

    # calc utilities
    exps = list()

    add_opt_out = True

    for platform, offer in offers.items():
        if add_opt_out:
            u = offer['wait_time'] * 2 * betas.Beta_wait + \
                offer['travel_time'] * 2 * betas.Beta_time + \
                offer['fare'] / 2 * betas.Beta_cost
            exps.append(exp(u))
            add_opt_out = False

        u = offer['wait_time'] * betas.Beta_wait + \
            offer['travel_time'] * betas.Beta_time + \
            offer['fare'] * betas.Beta_cost
        exps.append(exp(u))

    p = [_ / sum(exps) for _ in exps]
    platform_chosen = choice([-1] + list(offers.keys()), 1, p=p)[0]  # random choice with p

    if platform_chosen == -1:
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'chosen to opt out',
                                                     sim.print_now()))
    else:
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'chosen platform ' + str(platform_chosen),
                                                     sim.print_now()))
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'platform probs: ' + str(p),
                                                     sim.print_now()))

    # handle requests
    for platform_id, offer in offers.items():
        if int(platform_id) == platform_chosen:
            sim.plats[platform_id].handle_accepted(offer['pax_id'])
        else:
            sim.plats[platform_id].handle_rejected(offer['pax_id'])
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id,
                                                     "wait: {}, travel: {}, fare: {}".format(offer['wait_time'],
                                                                                             int(offer['travel_time']),
                                                                                             int(offer[
                                                                                                     'fare'] * 100) / 100),
                                                     sim.print_now()))
    return platform_chosen == -1
