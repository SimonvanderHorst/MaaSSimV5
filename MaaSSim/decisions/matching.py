################################################################################
# Module: decisions/matching.py
# Description: Platform matching and offer creation logic
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
import time as _time
import pandas as pd

from MaaSSim.pudo_optimizer import optimize_pudo_matching
from MaaSSim.pudo_logger import PudoDecisionLog
from MaaSSim.driver import driverEvent
from MaaSSim.traveller import travellerEvent

from MaaSSim.decisions._helpers import (
    _compute_baseline_fare, _downgrade_offer_to_d2d, _driver_s1_check,
)
from MaaSSim.decisions.rider import _d2d_ratio_check


# ─── Shared helper functions ────────────────────────────────────────────────


def _make_base_offer(pax_id, req_id, simpaxes, veh_id, request,
                     wait_time, travel_time, fare):
    """Create base offer dict with fields common to both D2D and PUDO offers."""
    return {
        'pax_id': pax_id,
        'req_id': req_id,
        'simpaxes': simpaxes,
        'veh_id': veh_id,
        'status': 0,  # 0 = offer made, 1 = accepted, -1 rejected by traveller, -2 rejected by veh
        'request': request,
        'wait_time': wait_time,
        'travel_time': travel_time,
        'fare': fare,
    }


def _reject_offer(sim, platform, request, simpaxes, veh_id, req_id,
                  rejected_by, veh=None, log_msg=None, remove_from_reqQ=False):
    """Handle offer rejection bookkeeping for both driver and rider rejections.

    Args:
        rejected_by: 'driver' or 'rider' — determines status code and event type
        veh: Vehicle agent (required for driver rejections to update driver event)
        log_msg: Custom log message (default: auto-generated from rejected_by)
        remove_from_reqQ: If True, also remove request from platform queue
    """
    if rejected_by == 'driver':
        if veh is not None:
            veh.update(event=driverEvent.REJECTS_REQUEST)
        status = -2
        pax_event = travellerEvent.IS_REJECTED_BY_VEHICLE
        if log_msg is None:
            log_msg = 'got rejected by vehicle ' + str(veh_id)
    else:
        status = -1
        pax_event = travellerEvent.REJECTS_OFFER
        if log_msg is None:
            log_msg = 'rider S1 rejected (ratio check) veh ' + str(veh_id)

    platform.offers[simpaxes[0]]['status'] = status
    for i in simpaxes:
        sim.pax[i].update(event=pax_event)
        sim.pax[i].offers[platform.platform.name]['status'] = status
        del sim.pax[i].offers[platform.platform.name]

    sim.logger.info("pax {:>4}  {:40} {}".format(request.name, log_msg, sim.print_now()))
    platform.tabu.append((veh_id, req_id))

    if remove_from_reqQ and req_id in platform.reqQ:
        platform.reqQ.pop(platform.reqQ.index(req_id))


def _accept_offer(sim, platform, simpaxes, veh_id, req_id):
    """Handle offer acceptance: trigger passenger events and remove from queues."""
    for i in simpaxes:
        if not sim.pax[i].got_offered.triggered:
            sim.pax[i].got_offered.succeed()
    platform.vehQ.pop(platform.vehQ.index(veh_id))
    platform.reqQ.pop(platform.reqQ.index(req_id))


# ─── PUDO offer economics and construction ──────────────────────────────────


def _calculate_offer_economics(sim, platform, match, params):
    """Compute wait/travel times, fares, and savings for a PUDO offer.

    Returns dict with keys: wait_time, travel_time, walk_to_pickup, walk_from_dropoff,
    baseline_fare, delta_dist_m, delta_time_s, delta_pi, d_walk_meters,
    rider_discount, final_fare, pi_r, pi_d.
    """
    req_id = match['req_id']
    veh_id = match['veh_id']
    pickup_node = match['pickup_node']
    dropoff_node = match['dropoff_node']

    request = sim.inData.requests.loc[req_id]
    vehicle = sim.vehicles.loc[veh_id]

    # Wait time (vehicle to pickup) and travel time (pickup to dropoff)
    wait_time = sim.skims.ride[vehicle.pos][pickup_node]
    travel_time = sim.skims.ride[pickup_node][dropoff_node]

    # Walking distances (undirected — pedestrians ignore one-way streets)
    walk_to_pickup = sim.skims.walk_dist[request.origin][pickup_node]
    walk_from_dropoff = sim.skims.walk_dist[dropoff_node][request.destination]

    # Baseline D2D fare
    d2d_ride_time = sim.skims.ride[request.origin][request.destination]
    baseline_fare = _compute_baseline_fare(platform.platform, request.dist, d2d_ride_time)

    # Distance and time savings (D2D vs PUDO, including dispatch)
    # Note: skim_dist[B][A] = forward distance from A to B (pandas col/row convention)
    dist_d2d = float(sim.skims.dist[request.origin][vehicle.pos] +
                      sim.skims.dist[request.destination][request.origin])
    dist_pudo = float(sim.skims.dist[pickup_node][vehicle.pos] +
                       sim.skims.dist[dropoff_node][pickup_node])
    delta_dist_m = dist_d2d - dist_pudo

    time_d2d = float(sim.skims.ride[vehicle.pos][request.origin] +
                      sim.skims.ride[request.origin][request.destination])
    time_pudo = float(sim.skims.ride[vehicle.pos][pickup_node] +
                       sim.skims.ride[pickup_node][dropoff_node])
    delta_time_s = time_d2d - time_pudo

    # Fare margin (ΔΠ)
    fare_per_km = platform.platform.fare
    fare_per_min = getattr(platform.platform, 'fare_per_min', 0.0)
    delta_pi = fare_per_km * delta_dist_m / 1000.0 + fare_per_min * delta_time_s / 60.0

    d_walk_meters = float(walk_to_pickup + walk_from_dropoff)

    # Incentive allocation (static defaults; DQN overrides post-S1 in create_pudo_offer)
    pi_r = params.pudo.static_rider_incentive
    pi_d = params.pudo.static_driver_incentive
    rider_discount = delta_pi * pi_r
    final_fare = max(baseline_fare - rider_discount, 0)

    return {
        'wait_time': wait_time,
        'travel_time': travel_time,
        'walk_to_pickup': walk_to_pickup,
        'walk_from_dropoff': walk_from_dropoff,
        'baseline_fare': baseline_fare,
        'delta_dist_m': delta_dist_m,
        'delta_time_s': delta_time_s,
        'delta_pi': delta_pi,
        'd_walk_meters': d_walk_meters,
        'pi_r': pi_r,
        'pi_d': pi_d,
        'rider_discount': rider_discount,
        'final_fare': final_fare,
    }


def _build_pudo_offers(sim, platform, simpaxes, veh_id, economics, match):
    """Build and assign PUDO offer dicts for all passengers in the request group."""
    for i in simpaxes:
        pax_request = sim.pax[i].request
        offer = _make_base_offer(
            pax_id=i, req_id=pax_request.name, simpaxes=simpaxes,
            veh_id=veh_id, request=pax_request,
            wait_time=economics['wait_time'],
            travel_time=economics['travel_time'],
            fare=economics['final_fare'],
        )
        # PUDO-specific fields
        offer.update({
            'pudo_enabled': True,
            'pickup_node': match['pickup_node'],
            'dropoff_node': match['dropoff_node'],
            'walk_to_pickup': economics['walk_to_pickup'],
            'walk_from_dropoff': economics['walk_from_dropoff'],
            'savings': match['savings'],
            'rider_incentive': economics['rider_discount'],
            # Thesis behavioral model fields
            'delta_dist_m': economics['delta_dist_m'],
            'delta_time_s': economics['delta_time_s'],
            'delta_pi': economics['delta_pi'],
            'd_walk_meters': economics['d_walk_meters'],
            # Incentive splits (from config)
            '_pi_r': economics['pi_r'],
            '_pi_d': economics['pi_d'],
        })

        # Rider heterogeneity: per-rider beta_zero
        pax_beta = sim.inData.passengers.loc[i].get('beta_zero', None)
        if pax_beta is not None:
            offer['_rider_beta_zero'] = float(pax_beta)

        platform.offers[i] = offer
        sim.pax[i].offers[platform.platform.name] = offer


# ─── Main offer creation and matching functions ─────────────────────────────


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
    savings = match['savings']

    request = sim.inData.requests.loc[req_id]
    veh = sim.vehs[veh_id]

    # Get shared-ride passenger IDs; first is the decision maker
    simpaxes = request.sim_schedule.req_id.dropna().unique()
    simpax = sim.pax[simpaxes[0]]

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
                pickup_node=match['pickup_node'], dropoff_node=match['dropoff_node'],
                wait_time=0, travel_time=0, baseline_fare=0,
                rider_discount=0, final_fare=0,
                walk_to_pickup_m=0, walk_from_dropoff_m=0,
                outcome='already_assigned',
            )
        return

    # Compute economics and build offers
    economics = _calculate_offer_economics(sim, platform, match, params)
    _build_pudo_offers(sim, platform, simpaxes, veh_id, economics, match)

    # ── Stage 1: Baseline feasibility checks (both driver and rider) ──
    driver_s1_fail = _driver_s1_check(
        economics['wait_time'], economics['baseline_fare'], params.speeds.ride)

    if driver_s1_fail:
        _reject_offer(sim, platform, request, simpaxes, veh_id, req_id,
                      rejected_by='driver', veh=veh)
        offer_outcome = 'driver_declined'

    else:
        # Rider S1: D2D ratio check (overhead/trip ratio)
        lead_offer = platform.offers[simpaxes[0]]
        rider_s1_fail = _d2d_ratio_check(sim, simpax, lead_offer)

        if rider_s1_fail:
            # ratio too high for this vehicle — stay in queue for next batch
            _reject_offer(sim, platform, request, simpaxes, veh_id, req_id,
                          rejected_by='rider')
            offer_outcome = 'rider_s1_declined'

        else:
            # Both S1 passed — mark offers and proceed to Stage 2
            for i in simpaxes:
                platform.offers[i]['_rider_s1_passed'] = True

            # DQN override: choose incentive splits now that S1 passed
            dqn_policy = getattr(sim, '_dqn_policy', None)
            if dqn_policy is not None and economics['delta_pi'] > 0:
                from MaaSSim.dqn.state import build_state_vector
                _beta_zero = float(sim.inData.passengers.loc[simpaxes[0]].get('beta_zero', 0.0))
                state = build_state_vector(sim, platform, economics, match, params,
                                           beta_zero=_beta_zero)
                pi_r, pi_d = dqn_policy.get_action(state)
                rider_discount = economics['delta_pi'] * pi_r
                final_fare = max(economics['baseline_fare'] - rider_discount, 0)
                for i in simpaxes:
                    platform.offers[i]['_pi_r'] = pi_r
                    platform.offers[i]['_pi_d'] = pi_d
                    platform.offers[i]['rider_incentive'] = rider_discount
                    platform.offers[i]['fare'] = final_fare

            # ── Stage 2: Behavioral checks (driver first, rider async) ──
            if veh.f_driver_decline(veh=veh):
                # Driver S2 rejection — try D2D fallback
                offer_outcome = _handle_driver_s2_rejection(
                    sim, platform, veh, params, simpaxes, veh_id, req_id,
                    request, savings)
            else:
                _accept_offer(sim, platform, simpaxes, veh_id, req_id)
                offer_outcome = 'accepted'

    # DQN outcome recording (only if get_action was called, i.e. S1 passed)
    dqn_policy = getattr(sim, '_dqn_policy', None)
    if dqn_policy is not None and dqn_policy._pending:
        _accepted = (offer_outcome == 'accepted')
        lead_offer = platform.offers.get(simpaxes[0], {})
        dqn_policy.record_outcome(
            delta_dist_m=economics['delta_dist_m'],
            accepted=_accepted,
            alpha_d=lead_offer.get('_driver_alpha'),
            alpha_r=lead_offer.get('_rider_alpha'),
        )

    # Log offer details
    if decision_log is not None:
        _log_offer_outcome(decision_log, sim, platform, simpaxes,
                           veh_id, req_id, match, economics, offer_outcome)


def _handle_driver_s2_rejection(sim, platform, veh, params, simpaxes,
                                veh_id, req_id, request, savings):
    """Handle driver Stage 2 rejection with optional D2D fallback.

    Returns the offer outcome string.
    """
    d2d_fallback_accepted = False
    if params.pudo.get('d2d_fallback', False) and savings > 0:
        for i in simpaxes:
            _downgrade_offer_to_d2d(sim, platform, platform.offers[i])
            sim.pax[i].offers[platform.platform.name] = platform.offers[i]
        if not veh.f_driver_decline(veh=veh):
            d2d_fallback_accepted = True

    if d2d_fallback_accepted:
        _accept_offer(sim, platform, simpaxes, veh_id, req_id)
        return 'd2d_fallback_driver'
    else:
        _reject_offer(sim, platform, request, simpaxes, veh_id, req_id,
                      rejected_by='driver', veh=veh)
        return 'driver_declined'


def _log_offer_outcome(decision_log, sim, platform, simpaxes,
                       veh_id, req_id, match, economics, offer_outcome):
    """Log offer details to the decision log."""
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
        pickup_node=lead_offer.get('pickup_node', match['pickup_node']),
        dropoff_node=lead_offer.get('dropoff_node', match['dropoff_node']),
        wait_time=lead_offer.get('wait_time', economics['wait_time']),
        travel_time=lead_offer.get('travel_time', economics['travel_time']),
        baseline_fare=economics['baseline_fare'],
        rider_discount=lead_offer.get('rider_incentive', economics['rider_discount']),
        final_fare=lead_offer.get('fare', economics['final_fare']),
        walk_to_pickup_m=lead_offer.get('walk_to_pickup', economics['walk_to_pickup']),
        walk_from_dropoff_m=lead_offer.get('walk_from_dropoff', economics['walk_from_dropoff']),
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
    platform = kwargs.get('platform')
    sim = platform.sim
    params = sim.params

    # prune stale tabu entries during DQN training (prevents unbounded growth)
    if getattr(sim, '_dqn_policy', None) is not None:
        vehQ_set, reqQ_set = set(platform.vehQ), set(platform.reqQ)
        platform.tabu = [t for t in platform.tabu
                         if t[0] in vehQ_set and t[1] in reqQ_set]

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

    # Dispatch to PUDO MILP matching if enabled
    if sim.params.get('pudo', {}).get('enabled', False):
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
            sim.logger.warning("Nobody likes each other, "
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
                pax_request = sim.pax[i].request
                if isinstance(pax_request.ttrav, int):
                    ttrav = pax_request.ttrav
                else:
                    ttrav = pax_request.ttrav.total_seconds()
                offer = _make_base_offer(
                    pax_id=i, req_id=pax_request.name, simpaxes=simpaxes,
                    veh_id=veh_id, request=pax_request,
                    wait_time=mintime, travel_time=ttrav,
                    fare=_compute_baseline_fare(platform.platform, sim.pax[i].request.dist, ttrav),
                )
                platform.offers[i] = offer
                sim.pax[i].offers[platform.platform.name] = offer
            if veh.f_driver_decline(veh=veh):  # allow driver reject the request
                _reject_offer(sim, platform, request, simpaxes, veh_id, req_id,
                              rejected_by='driver', veh=veh)
            else:
                for i in simpaxes:
                    if not sim.pax[i].got_offered.triggered:
                        sim.pax[i].got_offered.succeed()
                vehQ.pop(vehQ.index(veh_id))  # pop offered ones
                reqQ.pop(reqQ.index(req_id))  # from the queues

        platform.updateQs()


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
            sim.logger.warning('CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return True
        else:
            sim.logger.warning('NOT CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return False
