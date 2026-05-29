################################################################################
# Module: decisions/driver.py
# Description: Driver agent decision functions
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
import random
import pandas as pd
from dotmap import DotMap

from MaaSSim.decisions._helpers import _safe_sigmoid, _driver_s1_check


def f_driver_out(*args, **kwargs):
    # returns boolean True if vehicle decides to opt out
    leave_threshold = 0.25
    back_threshold = 0.5
    unserved_threshold = 0.005
    anneal = 0.2

    veh = kwargs.get('veh', None)  # input
    sim = veh.sim  # input
    flag = False  # output
    if len(sim.runs) == 0: # first day
        msg = 'veh {} stays on'.format(veh.id)
    else:
        last_run = sim.run_ids[-1]
        avg_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(
            back_threshold)  # how many rides was there on average
        quant_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(
            leave_threshold)  # what was the lower quantile of rides

        prev_rides = pd.Series([sim.res[_].veh_exp.loc[veh.id].nRIDES for _ in
                                sim.run_ids]).mean()  # how many rides did I have on average before

        rides_yesterday = sim.res[last_run].veh_exp.loc[veh.id].nRIDES # how many rides did I have yesterday

        unserved_demand_yesterday = sim.res[last_run].pax_exp[sim.res[last_run].pax_exp.LOSES_PATIENCE > 0].shape[0] / \
                                    sim.res[last_run].pax_exp.shape[0]  # what is the share of unserved demand
        did_i_work_yesterday = sim.res[last_run].veh_exp.loc[veh.id].ENDS_SHIFT > 0

        if not did_i_work_yesterday:
            if avg_yesterday < prev_rides:
                msg = 'veh {} stays out'.format(veh.id)
                flag = True
            elif unserved_demand_yesterday > unserved_threshold:
                if random.random() < anneal:
                    msg = 'veh {} comes to serve unserved'.format(veh.id)
                    flag = False
                else:
                    msg = 'veh {} someone else come to serve unserved'.format(veh.id)
                    flag = False
            else:
                msg = 'veh {} comes back'.format(veh.id)
                flag = False

            pass
        else:
            if rides_yesterday > quant_yesterday:
                msg = 'veh {} stays in'.format(veh.id)
                flag = False
            else:
                msg = 'veh {} leaves'.format(veh.id)
                flag = True

    sim.logger.info('DRIVER OUT: ' + msg)
    return flag


def f_repos(*args, **kwargs):
    """
    handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    :param args:
    :param kwargs: vehicle and simulation object (veh.sim)
    :return: structure with flag = bool, position to reposition to and time that it will take to reposition there.
    """

    import random
    repos = DotMap()
    if random.random() > 0.8:  # 20% of cases driver will repos
        driver = kwargs.get('veh', None)
        sim = driver.sim
        neighbors = list(sim.inData.G.neighbors(driver.veh.pos))
        if len(neighbors) == 0:
            # escape from dead-end (teleport)
            repos.pos = sim.inData.nodes.sample(1).squeeze().name
            repos.time = 300
        else:
            repos.pos = random.choice(neighbors)
            repos.time = driver.sim.skims.ride[repos.pos][driver.veh.pos]
        repos.flag = True
    else:
        repos.flag = False

    return repos



def f_pudo_driver_decline(*args, **kwargs):
    """Two-stage driver decline: distance-based baseline check + PUDO behavioral increment.

    Stage 1 (all offers): Fixed 480s cap on dispatch time (2000m / speeds.ride).
    Peak-hour rejection rises with congestion (intended). Fare floor: EUR 0.10.

    Stage 2 (PUDO only): Sigmoid on PUDO-incremental utility:
    U_pudo = (λ·π_d - 1)·ΔΠ + β_time·Δtime + β_dist·Δdist - C_friction

    This ensures D2D and PUDO face the same baseline rejection, with PUDO
    adding only the incremental acceptance cost of walking/friction.

    Returns True if driver DECLINES, False if driver ACCEPTS.
    """
    veh = kwargs.get('veh', None)
    offers = veh.platform.offers
    sim = veh.sim
    params = sim.params

    # Find my current offer
    my_offer = None
    for key, offer in offers.items():
        if offer['status'] == 0 and offer['veh_id'] == veh.id:
            my_offer = offer
            break

    if my_offer is None:
        return False

    # ── Stage 1: Distance-based baseline check (same for ALL offers) ──
    if _driver_s1_check(my_offer['wait_time'], my_offer['fare'], params.speeds.ride):
        return True

    # ── Stage 2: PUDO-specific behavioral increment ──
    # Non-PUDO offers: passed Stage 1, accept
    if not my_offer.get('pudo_enabled', False):
        return False

    behavioral = params.pudo.get('behavioral', {})
    if not behavioral.get('enabled', False):
        return False  # Behavioral model disabled, accept (Stage 1 already passed)

    # D2D fallback (savings=0): no PUDO-specific disutility, accept
    savings = my_offer.get('savings', 0)
    if savings == 0:
        return False

    # Get behavioral parameters
    pi_d = my_offer.get('_pi_d', params.pudo.static_driver_incentive)
    lambda_bonus = behavioral.get('driver_lambda_bonus', 1.0)
    beta_time = behavioral.get('driver_beta_time', 0.004)          # EUR/sec
    beta_dist = params.pudo.operating_cost_per_km / 1000.0         # EUR/meter
    c_friction = behavioral.get('driver_C_friction', 0.15)         # EUR
    sigmoid_scale_d = behavioral.get('driver_sigmoid_scale', 1.0)  # mu

    # Get offer quantities
    delta_pi = my_offer.get('delta_pi', 0)
    delta_time = my_offer.get('delta_time_s', 0)
    delta_dist = my_offer.get('delta_dist_m', 0)

    # Compute PUDO-incremental driver utility (4 components)
    is_pudo = (delta_time != 0 or delta_dist != 0)
    U_driver = ((lambda_bonus * pi_d - 1) * delta_pi
                + beta_time * delta_time
                + beta_dist * delta_dist
                - c_friction * is_pudo)

    # Sigmoid: P = sigma(mu * U)
    # At U=0, P = 50% (neutral — PUDO equivalent to D2D)
    sigmoid_input_d = sigmoid_scale_d * U_driver
    alpha_d = _safe_sigmoid(sigmoid_input_d)

    # Probabilistic acceptance
    accepts = random.random() < alpha_d

    # Store utility components on offer for logging
    my_offer['_driver_utility'] = U_driver
    my_offer['_driver_alpha'] = alpha_d
    my_offer['_driver_accepted'] = accepts
    my_offer['_driver_sigmoid_input'] = sigmoid_input_d

    return not accepts  # True = decline
