################################################################################
# Module: decision.py
# Description: Agent decision function templates
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
from math import exp
import random
import pandas as pd
from dotmap import DotMap
from numpy.random.mtrand import choice

from MaaSSim.pudo_optimizer import optimize_pudo_matching

from MaaSSim.driver import driverEvent
from MaaSSim.traveller import travellerEvent


def _compute_baseline_fare(platform_obj, dist_m, ride_time_s):
    """Compute rider-facing fare from distance + time components.

    Args:
        platform_obj: Platform row with 'fare' (EUR/km) and optional 'fare_per_min' (EUR/min)
        dist_m: Trip distance in meters
        ride_time_s: Trip ride time in seconds

    Returns:
        fare in EUR
    """
    fare_per_km = platform_obj.fare
    fare_per_min = getattr(platform_obj, 'fare_per_min', 0.0)
    return fare_per_km * dist_m / 1000 + fare_per_min * ride_time_s / 60


def _downgrade_offer_to_d2d(sim, platform, offer):
    """Rewrite a PUDO offer in-place to Door-to-Door terms.

    Sets pickup=origin, dropoff=destination, no walking, full fare.
    Used as fallback when driver or rider rejects PUDO but may accept D2D.
    """
    request = offer['request']
    veh_id = offer['veh_id']
    vehicle = sim.vehicles.loc[veh_id]

    offer['pickup_node'] = request.origin
    offer['dropoff_node'] = request.destination
    offer['walk_to_pickup'] = 0
    offer['walk_from_dropoff'] = 0
    offer['savings'] = 0
    offer['rider_incentive'] = 0
    offer['_d2d_fallback'] = True

    # Thesis behavioral model fields (all zero for D2D)
    offer['delta_dist_m'] = 0
    offer['delta_time_s'] = 0
    offer['delta_pi'] = 0
    offer['d_walk_meters'] = 0

    offer['wait_time'] = sim.skims.ride[vehicle.pos][request.origin]
    offer['travel_time'] = sim.skims.ride[request.origin][request.destination]

    d2d_ride_time = sim.skims.ride[request.origin][request.destination]
    offer['fare'] = _compute_baseline_fare(platform.platform, request.dist, d2d_ride_time)
    return offer


#################
#    DUMMIES    #
#################


def dummy_False(*args, **kwargs):
    # dummy function to always return False,
    # used as default function inside of functionality
    # (if the behaviour is not modelled)
    return False


def dummy_True(*args, **kwargs):
    # dummy function to always return True
    return True


def f_dummy_repos(*args, **kwargs):
    # handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    repos = DotMap()
    repos.flag = False
    # repos.pos = None
    # repos.time = 0
    return repos


################
#    DRIVER    #
################


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


def f_decline(*args, **kwargs):
    # determines whether driver will pick up the request or not
    # now it accepts requests only in the first quartile of travel times
    wait_limit = 200
    fare_limit = 0.1
    veh = kwargs.get('veh',None)
    offers = veh.platform.offers
    my_offer = None
    for key, offer in offers.items():
        if offer['status'] == 0 and offer['veh_id'] == veh.id:
            my_offer = offer
            break
    if my_offer is None:
        return False


    wait_time = my_offer['wait_time']
    fare = my_offer['fare']

    flag = False # i do not decline
    if wait_time  >= wait_limit:
        flag = True  # unless I have ot wait a lot
    if fare < fare_limit:
        flag = True  # or fare is low
    #if flag:
    #    veh.sim.logger.critical('Veh {} declined offer with {} wait time and fare {}'.format(veh.id, wait_time,fare))

    return flag


def _safe_sigmoid(x):
    """sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        ex = exp(x)
        return ex / (1.0 + ex)


def f_pudo_driver_decline(*args, **kwargs):
    """Two-stage driver decline: distance-based baseline check + PUDO behavioral increment.

    Stage 1 (all offers): Distance-based wait check — reject if dispatch
    distance exceeds 2000m (equivalent to 200s at 36 km/h). The threshold
    scales with ride speed so that low-speed scenarios (congestion) are not
    penalised by a fixed time limit. Fare floor: EUR 0.10.

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
    # 2000m dispatch limit = 200s at 36 km/h; scales with current ride speed
    WAIT_DIST_LIMIT_M = 2000.0
    wait_limit_s = WAIT_DIST_LIMIT_M / params.speeds.ride
    fare_limit = 0.10

    wait_time = my_offer['wait_time']
    fare = my_offer['fare']

    if wait_time >= wait_limit_s:
        return True  # Dispatch distance exceeds 2km equivalent
    if fare < fare_limit:
        return True  # Fare too low

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
    pi_d = params.pudo.static_driver_incentive
    lambda_bonus = behavioral.get('driver_lambda_bonus', 2.0)
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


# ######### #
# PLATFORM  #
# ######### #


def create_pudo_offer(sim, platform, match, params, decision_log=None):
    """Create offer with PUDO pickup/dropoff locations.

    Implements the restructured acceptance pipeline:
      Stage 1 (both parties): Driver S1 (distance/fare) → Rider S1 (ratio check)
      Stage 2 (behavioral):   Driver S2 (sigmoid) → Rider S2 (sigmoid, async)

    All Stage 1 feasibility checks run before any Stage 2 behavioral decisions.
    If both S1 checks pass, offers are flagged with ``_rider_s1_passed=True``
    so that ``f_pudo_rider_mode()`` can skip the redundant rider S1 check.

    Args:
        sim: Simulator object
        platform: Platform object
        match: Dict with keys: veh_id, req_id, pickup_node, dropoff_node, savings, cost_d2d, cost_pudo
        params: Configuration parameters
        decision_log: Optional PudoDecisionLog for recording offer outcomes
    """
    veh_id = match['veh_id']
    req_id = match['req_id']
    pickup_node = match['pickup_node']
    dropoff_node = match['dropoff_node']
    savings = match['savings']

    # Get request and vehicle
    request = sim.inData.requests.loc[req_id]
    vehicle = sim.vehicles.loc[veh_id]
    veh = sim.vehs[veh_id]

    # Get simpaxes (shared ride passengers)
    simpaxes = request.sim_schedule.req_id.dropna().unique()
    simpax = sim.pax[simpaxes[0]]  # first traveller is decision maker

    # Update events
    veh.update(event=driverEvent.RECEIVES_REQUEST)
    for i in simpaxes:
        sim.pax[i].update(event=travellerEvent.RECEIVES_OFFER)

    # Check if traveller already assigned
    if simpax.veh is not None:
        if req_id in platform.reqQ:
            platform.reqQ.pop(platform.reqQ.index(req_id))
        if decision_log is not None:
            decision_log.log_offer(
                veh_id=veh_id, req_id=req_id,
                pickup_node=pickup_node, dropoff_node=dropoff_node,
                wait_time=0, travel_time=0, baseline_fare=0,
                rider_discount=0, final_fare=0,
                walk_to_pickup_m=0, walk_from_dropoff_m=0,
                outcome='already_assigned',
            )
        return

    # Calculate wait time (vehicle to pickup)
    wait_time = sim.skims.ride[vehicle.pos][pickup_node]

    # Calculate travel time (pickup to dropoff)
    travel_time = sim.skims.ride[pickup_node][dropoff_node]

    # Calculate walking distances (undirected — pedestrians ignore one-way streets)
    walk_to_pickup = sim.skims.walk_dist[request.origin][pickup_node]
    walk_from_dropoff = sim.skims.walk_dist[dropoff_node][request.destination]

    # Calculate fare (baseline D2D fare minus rider incentive)
    d2d_ride_time = sim.skims.ride[request.origin][request.destination]  # seconds
    baseline_fare = _compute_baseline_fare(platform.platform, request.dist, d2d_ride_time)

    # Thesis: time-based and distance-based savings (total including dispatch)
    # Note: skim_dist[B][A] = forward distance from A to B (pandas col/row convention)
    dist_d2d = float(sim.skims.dist[request.origin][vehicle.pos] +
                      sim.skims.dist[request.destination][request.origin])
    dist_pudo = float(sim.skims.dist[pickup_node][vehicle.pos] +
                       sim.skims.dist[dropoff_node][pickup_node])
    delta_dist_m = dist_d2d - dist_pudo  # meters saved

    time_d2d = float(sim.skims.ride[vehicle.pos][request.origin] +
                      sim.skims.ride[request.origin][request.destination])
    time_pudo = float(sim.skims.ride[vehicle.pos][pickup_node] +
                       sim.skims.ride[pickup_node][dropoff_node])
    delta_time_s = time_d2d - time_pudo  # seconds saved

    # Fare margin from existing fare model (ΔΠ)
    fare_per_km = platform.platform.fare
    fare_per_min = getattr(platform.platform, 'fare_per_min', 0.0)
    delta_pi = fare_per_km * delta_dist_m / 1000.0 + fare_per_min * delta_time_s / 60.0

    # Walking distances (for behavioral model, in meters)
    d_walk_meters = float(walk_to_pickup + walk_from_dropoff)

    # Incentive allocation: static config values
    pi_r = params.pudo.static_rider_incentive
    pi_d = params.pudo.static_driver_incentive

    # Rider fare: baseline minus rider's share of ΔΠ
    rider_discount = delta_pi * pi_r
    final_fare = max(baseline_fare - rider_discount, 0)

    # Create offers for each passenger
    for i in simpaxes:
        pax_request = sim.pax[i].request

        offer = {
            'pax_id': i,
            'req_id': pax_request.name,
            'simpaxes': simpaxes,
            'veh_id': veh_id,
            'status': 0,  # 0 - offer made, 1 - accepted, -1 rejected by traveller, -2 rejected by veh
            'request': pax_request,
            'wait_time': wait_time,
            'travel_time': travel_time,
            'fare': final_fare,

            # PUDO-specific fields
            'pudo_enabled': True,
            'pickup_node': pickup_node,
            'dropoff_node': dropoff_node,
            'walk_to_pickup': walk_to_pickup,
            'walk_from_dropoff': walk_from_dropoff,
            'savings': savings,
            'rider_incentive': rider_discount,

            # Thesis behavioral model fields
            'delta_dist_m': delta_dist_m,
            'delta_time_s': delta_time_s,
            'delta_pi': delta_pi,
            'd_walk_meters': d_walk_meters,

            # Incentive splits (from config)
            '_pi_r': pi_r,
            '_pi_d': pi_d,
        }

        # Rider heterogeneity: per-rider beta_zero
        pax_beta = sim.inData.passengers.loc[i].get('beta_zero', None)
        if pax_beta is not None:
            offer['_rider_beta_zero'] = float(pax_beta)

        platform.offers[i] = offer
        sim.pax[i].offers[platform.platform.name] = offer

    # ── Stage 1: Baseline feasibility checks (both driver and rider) ──
    # All S1 checks run before any S2 behavioral decisions.

    # Driver S1: distance-based dispatch limit (2000m ≈ 200s at 36 km/h)
    WAIT_DIST_LIMIT_M = 2000.0
    wait_limit_s = WAIT_DIST_LIMIT_M / params.speeds.ride
    driver_s1_fail = (wait_time >= wait_limit_s) or (final_fare < 0.10)

    if driver_s1_fail:
        # Driver S1 rejection
        veh.update(event=driverEvent.REJECTS_REQUEST)
        platform.offers[simpaxes[0]]['status'] = -2
        for i in simpaxes:
            sim.pax[i].update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
            sim.pax[i].offers[platform.platform.name]['status'] = -2
            del sim.pax[i].offers[platform.platform.name]
        sim.logger.warning("pax {:>4}  {:40} {}".format(request.name,
                                                     'got rejected by vehicle ' + str(veh_id),
                                                     sim.print_now()))
        platform.tabu.append((veh_id, req_id))
        offer_outcome = 'driver_declined'

    else:
        # Rider S1: D2D ratio check (overhead/trip ratio > 0.5)
        lead_offer = platform.offers[simpaxes[0]]
        rider_s1_fail = _d2d_ratio_check(sim, simpax, lead_offer)

        if rider_s1_fail:
            # Rider S1 rejection — trip overhead too high for rider
            platform.offers[simpaxes[0]]['status'] = -1
            for i in simpaxes:
                sim.pax[i].update(event=travellerEvent.REJECTS_OFFER)
                sim.pax[i].offers[platform.platform.name]['status'] = -1
                del sim.pax[i].offers[platform.platform.name]
            sim.logger.warning("pax {:>4}  {:40} {}".format(request.name,
                                                         'rider S1 rejected (ratio check) veh ' + str(veh_id),
                                                         sim.print_now()))
            platform.tabu.append((veh_id, req_id))
            # No other vehicle can pass — closest already failed, farther ones have worse ratios
            if req_id in platform.reqQ:
                platform.reqQ.pop(platform.reqQ.index(req_id))
            offer_outcome = 'rider_s1_declined'

        else:
            # Both S1 passed — mark offers and proceed to Stage 2
            for i in simpaxes:
                platform.offers[i]['_rider_s1_passed'] = True

            # ── Stage 2: Behavioral checks (driver first, rider async) ──
            if veh.f_driver_decline(veh=veh):
                # Driver S2 rejection — try D2D fallback
                d2d_fallback_accepted = False
                if params.pudo.get('d2d_fallback', False) and savings > 0:
                    for i in simpaxes:
                        _downgrade_offer_to_d2d(sim, platform, platform.offers[i])
                        sim.pax[i].offers[platform.platform.name] = platform.offers[i]
                    if not veh.f_driver_decline(veh=veh):
                        d2d_fallback_accepted = True

                if d2d_fallback_accepted:
                    for i in simpaxes:
                        if not sim.pax[i].got_offered.triggered:
                            sim.pax[i].got_offered.succeed()
                    platform.vehQ.pop(platform.vehQ.index(veh_id))
                    platform.reqQ.pop(platform.reqQ.index(req_id))
                    offer_outcome = 'd2d_fallback_driver'
                else:
                    veh.update(event=driverEvent.REJECTS_REQUEST)
                    platform.offers[simpaxes[0]]['status'] = -2
                    for i in simpaxes:
                        sim.pax[i].update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
                        sim.pax[i].offers[platform.platform.name]['status'] = -2
                        del sim.pax[i].offers[platform.platform.name]
                    sim.logger.warning("pax {:>4}  {:40} {}".format(request.name,
                                                                 'got rejected by vehicle ' + str(veh_id),
                                                                 sim.print_now()))
                    platform.tabu.append((veh_id, req_id))
                    offer_outcome = 'driver_declined'
            else:
                # Accept offer - trigger events and remove from queues
                for i in simpaxes:
                    if not sim.pax[i].got_offered.triggered:
                        sim.pax[i].got_offered.succeed()
                platform.vehQ.pop(platform.vehQ.index(veh_id))
                platform.reqQ.pop(platform.reqQ.index(req_id))
                offer_outcome = 'accepted'

    # Log offer details
    if decision_log is not None:
        # Extract behavioral data from offer if present
        lead_offer = platform.offers.get(simpaxes[0], {})
        behavioral_data = {}
        if '_driver_utility' in lead_offer:
            behavioral_data = {
                'driver_utility': lead_offer['_driver_utility'],
                'driver_alpha': lead_offer['_driver_alpha'],
                'driver_accepted': lead_offer['_driver_accepted'],
            }
        d2d_fb = lead_offer.get('_d2d_fallback', False)
        decision_log.log_offer(
            veh_id=veh_id, req_id=req_id,
            pickup_node=lead_offer.get('pickup_node', pickup_node),
            dropoff_node=lead_offer.get('dropoff_node', dropoff_node),
            wait_time=lead_offer.get('wait_time', wait_time),
            travel_time=lead_offer.get('travel_time', travel_time),
            baseline_fare=baseline_fare,
            rider_discount=lead_offer.get('rider_incentive', rider_discount),
            final_fare=lead_offer.get('fare', final_fare),
            walk_to_pickup_m=lead_offer.get('walk_to_pickup', walk_to_pickup),
            walk_from_dropoff_m=lead_offer.get('walk_from_dropoff', walk_from_dropoff),
            outcome=offer_outcome,
            d2d_fallback=d2d_fb,
            **behavioral_data,
        )


def f_match_pudo(**kwargs):
    """
    PUDO-enabled matching function.
    Replaces greedy nearest-neighbor with MILP optimization to maximize operational savings.

    Args:
        kwargs: Must include 'platform' key with platform object

    Returns:
        List of matches from MILP optimization
    """
    from MaaSSim.pudo_logger import PudoDecisionLog

    platform = kwargs.get('platform')
    sim = platform.sim
    params = sim.params

    # Snapshot queue state before matching
    vehQ_snapshot = list(platform.vehQ)
    reqQ_snapshot = list(platform.reqQ)

    # Create decision log if enabled
    log_level = params.pudo.get('decision_log_level', 'off')
    decision_log = None
    if log_level != 'off':
        decision_log = PudoDecisionLog(
            batch_time=sim.env.now, batch_type='milp',
            vehQ=vehQ_snapshot, reqQ=reqQ_snapshot, log_level=log_level,
        )

    # Run MILP optimization
    matches, timing = optimize_pudo_matching(sim, platform, params,
                                              decision_log=decision_log)

    # Create offers for matched pairs
    import time as _time
    t_d = _time.perf_counter()
    for match in matches:
        create_pudo_offer(sim, platform, match, params,
                          decision_log=decision_log)
    timing['phase_d_offers_s'] = _time.perf_counter() - t_d

    # Record batch history
    batch_matches = []
    for m in matches:
        req = sim.inData.requests.loc[m['req_id']]
        match_entry = {
            'veh_id': m['veh_id'],
            'veh_pos': int(sim.vehicles.loc[m['veh_id']].pos),
            'req_id': m['req_id'],
            'origin': int(req.origin),
            'destination': int(req.destination),
            'pickup_node': int(m['pickup_node']),
            'dropoff_node': int(m['dropoff_node']),
            'savings': m['savings'],
            'cost_d2d': m['cost_d2d'],
            'cost_pudo': m['cost_pudo'],
        }
        if 'ranking_score' in m:
            match_entry['ranking_score'] = m['ranking_score']
            match_entry['rider_side_cost'] = m['rider_side_cost']
        batch_matches.append(match_entry)

    batch_entry = {
        't': sim.env.now,
        'batch_type': 'milp',
        'vehQ_size': len(vehQ_snapshot),
        'reqQ_size': len(reqQ_snapshot),
        'num_matches': len(matches),
        'matches': batch_matches,
        'timing': timing,
    }
    if decision_log is not None:
        decision_log.assignments = batch_matches
        batch_entry['decision_log'] = decision_log.to_dict()
    platform.batch_history.append(batch_entry)

    # Update queues after all offers created
    platform.updateQs()

    return matches


def f_match_greedy_first_pudo(**kwargs):
    """
    Greedy-First PUDO matching function.
    Combines greedy nearest-neighbor matching (preserves spatial coherence)
    with per-pair PUDO optimization (reduces passenger distance).

    This avoids the vehicle scattering problem of batch MILP optimization.

    Args:
        kwargs: Must include 'platform' key with platform object

    Returns:
        List of batch match dicts
    """
    from MaaSSim.pudo_optimizer import optimize_single_pair_pudo
    from MaaSSim.pudo_logger import PudoDecisionLog

    platform = kwargs.get('platform')
    sim = platform.sim
    params = sim.params

    vehQ = platform.vehQ
    reqQ = platform.reqQ

    # Snapshot queue state before matching
    vehQ_snapshot = list(vehQ)
    reqQ_snapshot = list(reqQ)

    # Create decision log if enabled
    log_level = params.pudo.get('decision_log_level', 'off')
    decision_log = None
    if log_level != 'off':
        decision_log = PudoDecisionLog(
            batch_time=sim.env.now, batch_type='greedy',
            vehQ=vehQ_snapshot, reqQ=reqQ_snapshot, log_level=log_level,
        )

    matches_count = 0
    batch_matches = []
    iteration = 0

    # Greedy matching loop (same as D2D)
    while min(len(reqQ), len(vehQ)) > 0:
        requests = sim.inData.requests.loc[reqQ]
        vehicles = sim.vehicles.loc[vehQ]

        # Find closest (vehicle, request) pair
        # Use reset_index to handle duplicate positions/origins
        skimQ = sim.skims.ride[requests.origin].loc[vehicles.pos].copy()
        skimQ.index = vehicles.index  # Use vehicle IDs as index instead of positions
        skimQ.columns = requests.index  # Use request IDs as columns instead of origins
        skimQ = skimQ.stack()
        n_before_tabu = len(skimQ)
        skimQ = skimQ.drop(platform.tabu, errors='ignore')
        n_tabu_dropped = n_before_tabu - len(skimQ)

        if skimQ.shape[0] == 0:
            sim.logger.warn(f"Nobody likes each other, Qs {len(vehQ)}veh; {len(reqQ)}req; tabu {len(platform.tabu)}")
            break

        # Greedy selection
        veh_id, req_id = skimQ.idxmin()  # Now returns vehicle ID and request ID directly
        chosen_pickup_time = float(skimQ.min())
        vehicle = vehicles.loc[veh_id]
        request = requests.loc[req_id]

        # Optimize PUDO nodes for this single pair
        pudo_match = optimize_single_pair_pudo(sim, veh_id, req_id, params,
                                               decision_log=decision_log)

        # Create offer (handles decline, events, and queue removal internally)
        create_pudo_offer(sim, platform, pudo_match, params,
                          decision_log=decision_log)

        # Check if offer was accepted (create_pudo_offer removes from queues on accept)
        if veh_id not in vehQ:
            # Read realized values from offer (may differ from pudo_match if D2D fallback)
            req_simpaxes = request.sim_schedule.req_id.dropna().unique()
            lead_offer = platform.offers.get(req_simpaxes[0], {})
            is_fallback = lead_offer.get('_d2d_fallback', False)
            offer_outcome = 'd2d_fallback' if is_fallback else 'accepted'
            matches_count += 1
            greedy_match_entry = {
                'veh_id': veh_id,
                'veh_pos': int(vehicle.pos),
                'req_id': req_id,
                'origin': int(request.origin),
                'destination': int(request.destination),
                'pickup_node': int(lead_offer.get('pickup_node', pudo_match['pickup_node'])),
                'dropoff_node': int(lead_offer.get('dropoff_node', pudo_match['dropoff_node'])),
                'savings': lead_offer.get('savings', pudo_match['savings']),
                'cost_d2d': pudo_match['cost_d2d'],
                'cost_pudo': pudo_match['cost_pudo'],
                'd2d_fallback': is_fallback,
            }
            if 'ranking_score' in pudo_match:
                greedy_match_entry['ranking_score'] = pudo_match['ranking_score']
                greedy_match_entry['rider_side_cost'] = pudo_match['rider_side_cost']
            batch_matches.append(greedy_match_entry)
        else:
            offer_outcome = 'declined'

        # Log greedy iteration
        if decision_log is not None:
            # Build candidate list for logging (top candidates by pickup time)
            skimQ_candidates = []
            for (v, q), val in skimQ.items():
                skimQ_candidates.append({
                    'veh_id': int(v), 'req_id': int(q),
                    'pickup_time_s': float(val),
                })
            decision_log.log_greedy_iteration(
                iteration=iteration,
                vehQ_remaining=list(vehQ), reqQ_remaining=list(reqQ),
                skimQ_candidates=skimQ_candidates,
                n_tabu_dropped=n_tabu_dropped,
                chosen_veh_id=veh_id, chosen_req_id=req_id,
                chosen_pickup_time=chosen_pickup_time,
                offer_outcome=offer_outcome,
            )

        iteration += 1

    # Record batch history
    batch_entry = {
        't': sim.env.now,
        'batch_type': 'greedy',
        'vehQ_size': len(vehQ_snapshot),
        'reqQ_size': len(reqQ_snapshot),
        'num_matches': matches_count,
        'matches': batch_matches,
    }
    if decision_log is not None:
        decision_log.assignments = batch_matches
        batch_entry['decision_log'] = decision_log.to_dict()
    platform.batch_history.append(batch_entry)

    platform.updateQs()
    return batch_matches


def f_match(**kwargs):
    """
    for each platfrom, whenever one of the queues changes (new idle vehicle or new unserved request)
    this procedure handles the queue and prepares transactions between drivers and travellers
    it operates based on nearest vehicle and prepares and offer to accept by traveller/vehicle
    Can operate in two modes:
    - Greedy nearest-neighbor (original)
    - PUDO MILP optimization (if params.pudo.enabled=True)
    :param kwargs:
    :return:
    """

    platform = kwargs.get('platform')  # platform for which we perform matching
    sim = platform.sim  # reference to the simulation object

    # Dispatch to appropriate matcher
    if sim.params.get('pudo', {}).get('enabled', False):
        if sim.params.get('pudo', {}).get('greedy_first', False):
            # NEW: Greedy-First PUDO (preserves spatial coherence)
            return f_match_greedy_first_pudo(**kwargs)
        else:
            # EXISTING: Batch MILP optimization
            return f_match_pudo(**kwargs)

    # Otherwise continue with original greedy logic
    vehQ = platform.vehQ  # queue of idle vehicles
    reqQ = platform.reqQ  # queue of unserved requests

    while min(len(reqQ), len(vehQ)) > 0:  # loop until one of queues is empty (i.e. all requests handled)
        requests = sim.inData.requests.loc[reqQ]  # queued schedules of requests
        vehicles = sim.vehicles.loc[vehQ]  # vehicle agents
        skimQ = sim.skims.ride[requests.origin].loc[vehicles.pos].copy()
        skimQ.index = vehicles.index    # vehicle IDs instead of positions (handles duplicate positions)
        skimQ.columns = requests.index  # request IDs instead of origins (handles duplicate origins)
        skimQ = skimQ.stack()  # travel times between requests and vehicles in column vector form

        skimQ = skimQ.drop(platform.tabu, errors='ignore')  # drop already rejected matches


        if skimQ.shape[0] == 0:
            sim.logger.warn("Nobody likes each other, "
                            "Qs {}veh; {}req; tabu {}".format(len(vehQ), len(reqQ), len(platform.tabu)))
            break  # nobody likes each other - wait until new request or new vehicle

        veh_id, req_id = skimQ.idxmin()  # find the closest (vehicle, request) pair

        mintime = skimQ.min()  # and the travel time
        vehicle = vehicles.loc[veh_id]
        veh = sim.vehs[veh_id]  # vehicle agent

        request = requests.loc[req_id]
        simpaxes = request.sim_schedule.req_id.dropna().unique()
        simpax = sim.pax[simpaxes[0]]  # first traveller of shared ride (he is a leader and decision maker)

        veh.update(event=driverEvent.RECEIVES_REQUEST)
        for i in simpaxes:
            sim.pax[i].update(event=travellerEvent.RECEIVES_OFFER)

        if simpax.veh is not None:  # the traveller already assigned (to a different platform)
            if req_id in platform.reqQ:  # we were too late, forget about it
                platform.reqQ.pop(platform.reqQ.index(req_id))  # pop this request (vehicle still in the queue)
        else:
            for i in simpaxes:
                offer_id = i
                pax_request = sim.pax[i].request
                if isinstance(pax_request.ttrav, int):
                    ttrav = pax_request.ttrav
                else:
                    ttrav = pax_request.ttrav.total_seconds()
                offer = {'pax_id': i,
                         'req_id': pax_request.name,
                         'simpaxes': simpaxes,
                         'veh_id': veh_id,
                         'status': 0,  # 0 -  offer made, 1 - accepted, -1 rejected by traveller, -2 rejected by veh
                         'request': pax_request,
                         'wait_time': mintime,
                         'travel_time': ttrav,
                         'fare': platform.platform.fare * sim.pax[i].request.dist / 1000}  # make an offer
                platform.offers[offer_id] = offer  # bookkeeping of offers made by platform
                sim.pax[i].offers[platform.platform.name] = offer  # offer transferred to
            if veh.f_driver_decline(veh=veh):  # allow driver reject the request
                veh.update(event=driverEvent.REJECTS_REQUEST)
                platform.offers[offer_id]['status'] = -2
                for i in simpaxes:
                    sim.pax[i].update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
                    sim.pax[i].offers[platform.platform.name]['status'] = -2
                    del sim.pax[i].offers[platform.platform.name]  # prevent stale offer → KeyError race
                sim.logger.warning("pax {:>4}  {:40} {}".format(request.name,
                                                             'got rejected by vehicle ' + str(veh_id),
                                                             sim.print_now()))
                platform.tabu.append((veh_id, req_id))  # they are unmatchable
            else:
                for i in simpaxes:
                    if not sim.pax[i].got_offered.triggered:
                        sim.pax[i].got_offered.succeed()
                vehQ.pop(vehQ.index(veh_id))  # pop offered ones
                reqQ.pop(reqQ.index(req_id))  # from the queues

        platform.updateQs()


# ######### #
# TRAVELLER #
# ######### #

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


#############
# SIMULATOR #
#############

def f_stop_crit(*args, **kwargs):
    """
    Decision whether to stop experiment after current iterartion
    :param args:
    :param kwargs: sim object
    :return: boolean flag
    """
    sim = kwargs.get('sim', None)
    convergence_threshold = 0.001
    _ = sim.run_ids[-1]
    sim.logger.warning(sim.res[_].veh_exp[sim.res[_].veh_exp.ENDS_SHIFT > 0].shape[0])
    if len(sim.runs) < 2:
        sim.logger.warning('Early days')
        return False
    else:
        # example of convergence on waiting times
        convergence = abs((sim.res[sim.run_ids[-1]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean'] -
                           sim.res[sim.run_ids[-2]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean']) /
                          sim.res[sim.run_ids[-2]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean'])
        if convergence < convergence_threshold:
            sim.logger.warn('CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return True
        else:
            sim.logger.warn('NOT CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return False
