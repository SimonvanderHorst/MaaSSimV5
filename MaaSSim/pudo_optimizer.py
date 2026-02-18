################################################################################
# Module: pudo_optimizer.py
# Description: MILP-based optimization for PUDO (Pick-Up/Drop-Off) matching
# Batch MILP optimization to maximize operational savings
################################################################################

import gurobipy as gp
import pandas as pd
import numpy as np
import logging
import time as _time


def get_feasible_pudo_nodes(node_id, max_walk_distance, skim_walk, skim_dist,
                            walking_speed, safe_nodes=None):
    """
    Find all nodes within walking distance of a given node.

    Args:
        node_id: Origin/destination node
        max_walk_distance: Maximum walking meters
        skim_walk: Walk time matrix (seconds)
        skim_dist: Distance matrix (meters)
        walking_speed: Walking speed (m/s)
        safe_nodes: Optional frozenset of node IDs on low-speed roads (speed <= threshold).
                    If provided, candidates are further filtered to this set.

    Returns:
        Tuple of (feasible_node_ids, walk_distances_dict, walk_times_dict)
        - feasible_node_ids: List of node IDs within walking distance
        - walk_distances_dict: {node_id: distance_meters} for feasible nodes
        - walk_times_dict: {node_id: time_seconds} for feasible nodes
    """
    # Use raw distance skim for filtering (avoids int-truncation mismatch with stored walk distances)
    walk_distances = skim_dist[node_id]
    walk_times = skim_walk[node_id]

    # Filter nodes within max walking distance
    feasible_dist = walk_distances[walk_distances <= max_walk_distance]

    # Exclude nodes on high-speed roads (if filter is active)
    if safe_nodes is not None:
        feasible_dist = feasible_dist[feasible_dist.index.isin(safe_nodes)]

    node_list = feasible_dist.index.tolist()
    dist_dict = {int(n): float(feasible_dist[n]) for n in node_list}
    time_dict = {int(n): float(walk_times[n]) for n in node_list}

    return node_list, dist_dict, time_dict


def calculate_pudo_costs(vehicles, requests, skim_dist, skim_ride,
                         feasible_origins, feasible_destinations, params,
                         fare_per_km, fare_per_min=0.0, decision_log=None,
                         skim_walk_dist=None):
    """
    Calculate HOLISTIC operational costs for all vehicle-request-PUDO combinations.
    Pre-selects the best PUDO pair for each (vehicle, request) combination to reduce MILP complexity.

    Accounts for vehicle cost (distance + time) + walking disutility + geometric efficiency (detour penalty)

    Args:
        vehicles: DataFrame of available vehicles with 'pos' column
        requests: DataFrame of requests with 'origin', 'destination'
        skim_dist: Distance matrix (meters)
        skim_ride: Ride time matrix (seconds); ride[A][B] = time from A to B
        feasible_origins: Dict[req_id -> List[node_ids]]
        feasible_destinations: Dict[req_id -> List[node_ids]]
        params: Configuration parameters
        fare_per_km: Platform fare rate (EUR/km) — distance component
        fare_per_min: Platform fare rate (EUR/min) — time component (default 0 = distance-only)
        decision_log: Optional PudoDecisionLog for recording cost details

    Returns:
        cost_matrix: DataFrame with columns [veh_id, req_id, savings, pickup_node, dropoff_node, cost_d2d, cost_pudo]
    """
    results = []
    _walk_dist = skim_walk_dist if skim_walk_dist is not None else skim_dist  # backward compat
    cost_per_meter = fare_per_km / 1000.0
    cost_per_second = fare_per_min / 60.0  # EUR/s from EUR/min
    # Walking cost as a multiple of distance fare (beta_walk = 1.0 means walking costs same as driving per meter)
    walking_cost_per_meter = cost_per_meter * params.pudo.beta_walk
    friction_cost = params.pudo.friction_cost  # Detour penalty weight

    # Rider-aware optimization: use rider-side costs to rank PUDO combos
    rider_aware = params.pudo.get('rider_aware_optimization', False)
    if rider_aware:
        behavioral = params.pudo.get('behavioral', {})
        ra_walk_speed = params.pudo.walking_speed                          # m/s
        ra_beta_walk = behavioral.get('rider_beta_walk_time', 0.24)        # EUR/min
        ra_beta_wait = behavioral.get('rider_beta_wait', 0.22)             # EUR/min
        ra_beta_zero = behavioral.get('rider_beta_zero', 0.0)              # EUR
        ra_alpha = params.pudo.get('rider_cost_weight', 1.0)               # weighting factor

    max_dispatch_ratio = params.pudo.get('max_dispatch_ratio', float('inf'))

    for veh_id, veh in vehicles.iterrows():
        for req_id, req in requests.iterrows():
            # === D2D BASELINE: Total cost (vehicle only, no walking) ===
            # Note: skim_dist[B][A] = forward distance from A to B (pandas col/row convention)
            dist_to_origin = skim_dist[req.origin][veh.pos]            # forward: veh → origin
            dist_origin_to_dest = skim_dist[req.destination][req.origin]  # forward: origin → dest

            # Dispatch ratio pre-filter: skip if dispatch >> trip distance
            if dist_origin_to_dest > 0 and dist_to_origin / dist_origin_to_dest > max_dispatch_ratio:
                continue

            # Dispatch distance pre-filter: skip vehicles beyond threshold from origin
            # Mirrors S1 driver check intent (congestion-independent)
            max_dispatch_m = params.pudo.get('max_dispatch_distance', float('inf'))
            if dist_to_origin > max_dispatch_m:
                continue

            time_to_origin = skim_ride[veh.pos][req.origin]  # seconds
            time_origin_to_dest = skim_ride[req.origin][req.destination]  # seconds
            cost_d2d_vehicle = ((dist_to_origin + dist_origin_to_dest) * cost_per_meter +
                                (time_to_origin + time_origin_to_dest) * cost_per_second)
            cost_d2d_walking = 0  # No walking in D2D
            cost_d2d_total = cost_d2d_vehicle

            # Direct path for geometric efficiency check (full D2D path)
            direct_vehicle_path = dist_to_origin + dist_origin_to_dest

            # Find best PUDO pair for this (vehicle, request)
            best_savings = 0
            best_ranking = 0  # ranking_score for comparison (= savings when rider_aware is off)
            best_rider_side_cost = 0.0
            best_pickup = req.origin  # Default to D2D
            best_dropoff = req.destination
            best_cost_pudo = cost_d2d_total
            best_components = None  # Track cost components of the best combo
            n_combos = 0

            # === TRY ALL FEASIBLE PUDO COMBINATIONS ===
            for pickup_node in feasible_origins.get(req_id, [req.origin]):
                for dropoff_node in feasible_destinations.get(req_id, [req.destination]):
                    n_combos += 1

                    # Vehicle cost (distance + time)
                    dist_to_pickup = skim_dist[pickup_node][veh.pos]             # forward: veh → pickup
                    dist_pickup_to_dropoff = skim_dist[dropoff_node][pickup_node]  # forward: pickup → dropoff
                    time_to_pickup = skim_ride[veh.pos][pickup_node]  # seconds
                    time_pickup_to_dropoff = skim_ride[pickup_node][dropoff_node]  # seconds
                    cost_pudo_vehicle = ((dist_to_pickup + dist_pickup_to_dropoff) * cost_per_meter +
                                         (time_to_pickup + time_pickup_to_dropoff) * cost_per_second)

                    # Walking cost (undirected — pedestrians ignore one-way streets)
                    walk_to_pickup = _walk_dist[req.origin][pickup_node]
                    walk_from_dropoff = _walk_dist[dropoff_node][req.destination]
                    cost_pudo_walking = (walk_to_pickup + walk_from_dropoff) * walking_cost_per_meter

                    # Detour penalty (geometric efficiency check)
                    actual_vehicle_path = dist_to_pickup + dist_pickup_to_dropoff
                    detour_ratio = actual_vehicle_path / max(direct_vehicle_path, 1)
                    detour_penalty = 0
                    if detour_ratio > 1.2:  # More than 20% detour from direct path
                        detour_penalty = (detour_ratio - 1.0) * friction_cost

                    # Total PUDO cost (vehicle operational only; walking gated by max_walking_distance)
                    cost_pudo_total = cost_pudo_vehicle + detour_penalty

                    # Calculate savings (holistic comparison)
                    savings = cost_d2d_total - cost_pudo_total

                    # Rider-aware: compute rider-side cost for ranking
                    if rider_aware and savings > 0:
                        t_walk_min = (walk_to_pickup + walk_from_dropoff) / ra_walk_speed / 60.0
                        t_wait_savings_min = min(walk_to_pickup / ra_walk_speed, time_to_pickup) / 60.0
                        rider_side_cost = (ra_beta_walk * t_walk_min
                                           - ra_beta_wait * t_wait_savings_min
                                           + ra_beta_zero)
                        ranking_score = savings - ra_alpha * rider_side_cost
                    else:
                        rider_side_cost = 0.0
                        ranking_score = savings

                    # Log per-combo cost detail (full mode only)
                    if decision_log is not None:
                        log_kwargs = dict(
                            veh_id=veh_id, req_id=req_id,
                            pickup_node=pickup_node, dropoff_node=dropoff_node,
                            dist_veh_to_origin=dist_to_origin,
                            dist_origin_to_dest=dist_origin_to_dest,
                            cost_d2d=cost_d2d_total,
                            dist_veh_to_pickup=dist_to_pickup,
                            dist_pickup_to_dropoff=dist_pickup_to_dropoff,
                            cost_pudo_vehicle=cost_pudo_vehicle,
                            walk_to_pickup_m=walk_to_pickup,
                            walk_from_dropoff_m=walk_from_dropoff,
                            cost_pudo_walking=cost_pudo_walking,
                            detour_ratio=detour_ratio,
                            detour_penalty=detour_penalty,
                            cost_pudo_total=cost_pudo_total,
                            savings=savings,
                        )
                        if rider_aware and savings > 0:
                            log_kwargs['rider_side_cost'] = rider_side_cost
                            log_kwargs['ranking_score'] = ranking_score
                        decision_log.log_cost_detail(**log_kwargs)

                    if ranking_score > best_ranking:
                        best_ranking = ranking_score
                        best_savings = savings
                        best_rider_side_cost = rider_side_cost
                        best_pickup = pickup_node
                        best_dropoff = dropoff_node
                        best_cost_pudo = cost_pudo_total
                        best_components = {
                            'cost_vehicle_driving': cost_pudo_vehicle,
                            'cost_walking': cost_pudo_walking,
                            'cost_detour_penalty': detour_penalty,
                            'walk_to_pickup_m': walk_to_pickup,
                            'walk_from_dropoff_m': walk_from_dropoff,
                            'dist_veh_to_pickup_m': dist_to_pickup,
                            'dist_pickup_to_dropoff_m': dist_pickup_to_dropoff,
                        }
                        if rider_aware:
                            best_components['rider_side_cost'] = rider_side_cost
                            best_components['ranking_score'] = ranking_score

            # Only store if there's positive savings (or include D2D as fallback)
            if best_savings >= 0:
                entry = {
                    'veh_id': veh_id,
                    'req_id': req_id,
                    'savings': best_savings,
                    'pickup_node': best_pickup,
                    'dropoff_node': best_dropoff,
                    'cost_d2d': cost_d2d_total,
                    'cost_pudo': best_cost_pudo,
                }
                # Always include ranking_score (= savings when rider_aware is off)
                entry['rider_side_cost'] = best_rider_side_cost
                entry['ranking_score'] = best_ranking
                results.append(entry)

                # Log best edge with cost components
                if decision_log is not None:
                    decision_log.log_best_edge(
                        veh_id=veh_id, req_id=req_id,
                        best_pickup=best_pickup, best_dropoff=best_dropoff,
                        cost_d2d=cost_d2d_total, cost_pudo=best_cost_pudo,
                        savings=best_savings, n_combos_evaluated=n_combos,
                        cost_components=best_components,
                    )

    return pd.DataFrame(results)


def _extract_matches(cost_matrix, selected_vq_pairs):
    """Extract match dicts from cost_matrix for selected (veh_id, req_id) pairs."""
    matches = []
    selected_set = set(selected_vq_pairs)
    for _, row in cost_matrix.iterrows():
        if (row['veh_id'], row['req_id']) in selected_set:
            match = {
                'veh_id': row['veh_id'],
                'req_id': row['req_id'],
                'pickup_node': row['pickup_node'],
                'dropoff_node': row['dropoff_node'],
                'savings': row['savings'],
                'cost_d2d': row['cost_d2d'],
                'cost_pudo': row['cost_pudo'],
            }
            match['ranking_score'] = row['ranking_score']
            match['rider_side_cost'] = row['rider_side_cost']
            matches.append(match)
    return matches


def _log_milp(decision_log, cost_matrix, M, selected_vq_pairs,
              n_vars, n_veh_constraints, n_req_constraints,
              solver_status, solver_time_s):
    """Shared MILP logging for all solver backends."""
    selected_set = set(selected_vq_pairs)
    obj_coefficients = []
    selected_pairs = []
    unselected_pairs = []
    for _, row in cost_matrix.iterrows():
        v, q = row['veh_id'], row['req_id']
        coeff = float(M + row['ranking_score'])
        entry = {
            'veh_id': int(v), 'req_id': int(q),
            'ranking_score': float(row['ranking_score']), 'coeff': coeff,
        }
        obj_coefficients.append(entry)
        if (v, q) in selected_set:
            selected_pairs.append({
                'veh_id': int(v), 'req_id': int(q), 'x_value': 1.0,
            })
        else:
            unselected_pairs.append(entry)

    decision_log.log_milp(
        n_variables=n_vars,
        n_vehicle_constraints=n_veh_constraints,
        n_request_constraints=n_req_constraints,
        big_M=M,
        objective_coefficients=obj_coefficients,
        solver_status=solver_status,
        solver_time_s=solver_time_s,
        selected_pairs=selected_pairs,
        unselected_pairs=unselected_pairs,
    )


def solve_milp_matching(vehicles, requests, cost_matrix, decision_log=None):
    """
    Solve bipartite matching via LP relaxation to minimize total operational cost.

    The constraint matrix is totally unimodular with integer RHS,
    so the LP relaxation produces exact integer solutions — no MIP needed.

    Formulation (big-M ensures maximum matching):
    Maximize: Σ (M + ranking_score[v,q]) · x_v,q
    Subject to:
        Σ_q x_v,q <= 1  for all vehicles v
        Σ_v x_v,q <= 1  for all requests q
        x_v,q ∈ [0, 1]
    """
    model = gp.Model("PUDO_Matching")
    model.setParam('OutputFlag', 0)

    pairs = list(zip(cost_matrix['veh_id'], cost_matrix['req_id']))
    score_lookup = dict(zip(pairs, cost_matrix['ranking_score']))

    # Continuous variables (LP relaxation yields integer solutions for bipartite matching)
    x = model.addVars(pairs, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x")

    M = cost_matrix['ranking_score'].abs().max() * 2 + 1
    model.setObjective(
        gp.quicksum((M + score_lookup[p]) * x[p] for p in pairs),
        gp.GRB.MAXIMIZE
    )

    # Each vehicle serves at most one request
    veh_pairs = {}
    for v, q in pairs:
        veh_pairs.setdefault(v, []).append((v, q))
    n_vc = 0
    for v in vehicles:
        if v in veh_pairs:
            model.addConstr(gp.quicksum(x[p] for p in veh_pairs[v]) <= 1)
            n_vc += 1

    # Each request served by at most one vehicle
    req_pairs = {}
    for v, q in pairs:
        req_pairs.setdefault(q, []).append((v, q))
    n_rc = 0
    for q in requests:
        if q in req_pairs:
            model.addConstr(gp.quicksum(x[p] for p in req_pairs[q]) <= 1)
            n_rc += 1

    t0 = _time.perf_counter()
    model.optimize()
    t_solve = _time.perf_counter() - t0

    selected = []
    if model.status == gp.GRB.OPTIMAL:
        for p in pairs:
            if x[p].X > 0.5:
                selected.append(p)
        solver_status = 'Optimal'
    else:
        solver_status = f'Gurobi LP status {model.status}'

    matches = _extract_matches(cost_matrix, selected)
    if matches:
        total_cost = sum(m['cost_pudo'] for m in matches)
        total_savings = sum(m['savings'] for m in matches)
        logging.info(f"LP solved: {len(matches)} matches, "
                     f"cost {total_cost:.2f}, savings {total_savings:.2f}, "
                     f"time {t_solve:.4f}s")
    else:
        logging.warning(f"LP solver status: {solver_status}")

    if decision_log is not None:
        _log_milp(decision_log, cost_matrix, M, selected,
                  len(pairs), n_vc, n_rc, solver_status, t_solve)

    return matches


def optimize_pudo_matching(sim, platform, params, decision_log=None):
    """
    Main entry point for PUDO optimization called from decisions.py.

    Args:
        sim: Simulator object
        platform: Platform object with vehQ and reqQ queues
        params: Configuration parameters
        decision_log: Optional PudoDecisionLog for recording decisions

    Returns:
        matches: List of optimal vehicle-request-PUDO matches
    """
    timing = {}

    # Get current queues
    vehQ = platform.vehQ
    reqQ = platform.reqQ

    if len(vehQ) == 0 or len(reqQ) == 0:
        logging.info("Empty vehicle or request queue, no matches possible")
        return [], timing

    vehicles = sim.vehicles.loc[vehQ]
    requests = sim.inData.requests.loc[reqQ]
    timing['n_vehicles'] = len(vehQ)
    timing['n_requests'] = len(reqQ)

    logging.info(f"PUDO optimization: {len(vehQ)} vehicles, {len(reqQ)} requests")

    # Phase A: Find feasible PUDO nodes for each request
    t_a = _time.perf_counter()
    feasible_origins = {}
    feasible_destinations = {}
    max_walk_dist = params.pudo.max_walking_distance
    walking_speed = params.pudo.walking_speed

    safe_nodes = getattr(sim.inData, 'safe_nodes', None)

    for req_id in reqQ:
        req = requests.loc[req_id]

        # Find feasible pickup nodes (undirected: pedestrians ignore one-way streets)
        origin_nodes, origin_dists, origin_times = get_feasible_pudo_nodes(
            req.origin, max_walk_dist,
            sim.skims.walk, sim.skims.walk_dist, walking_speed,
            safe_nodes=safe_nodes
        )
        feasible_origins[req_id] = origin_nodes

        # Find feasible dropoff nodes (walk_dist is symmetric, no .T needed)
        dest_nodes, dest_dists, dest_times = get_feasible_pudo_nodes(
            req.destination, max_walk_dist,
            sim.skims.walk, sim.skims.walk_dist, walking_speed,
            safe_nodes=safe_nodes
        )
        feasible_destinations[req_id] = dest_nodes

        # Log feasibility
        if decision_log is not None:
            decision_log.log_feasibility(
                req_id=req_id, side='origin', anchor_node=req.origin,
                feasible_nodes=origin_nodes, walk_distances=origin_dists,
                walk_times=origin_times, max_walk_distance=max_walk_dist,
            )
            decision_log.log_feasibility(
                req_id=req_id, side='destination', anchor_node=req.destination,
                feasible_nodes=dest_nodes, walk_distances=dest_dists,
                walk_times=dest_times, max_walk_distance=max_walk_dist,
            )

        logging.debug(f"Request {req_id}: {len(origin_nodes)} pickup nodes, {len(dest_nodes)} dropoff nodes")

    timing['phase_a_feasibility_s'] = _time.perf_counter() - t_a

    # Phase B: Calculate costs for all combinations (pre-select best PUDO per v-q pair)
    t_b = _time.perf_counter()
    platform_fare = sim.inData.platforms.iloc[0].fare
    platform_fare_per_min = getattr(sim.inData.platforms.iloc[0], 'fare_per_min', 0.0)
    cost_matrix = calculate_pudo_costs(
        vehicles, requests, sim.skims.dist, sim.skims.ride,
        feasible_origins, feasible_destinations, params,
        fare_per_km=platform_fare, fare_per_min=platform_fare_per_min,
        decision_log=decision_log,
        skim_walk_dist=sim.skims.walk_dist,
    )
    timing['phase_b_cost_matrix_s'] = _time.perf_counter() - t_b
    timing['n_cost_matrix_rows'] = len(cost_matrix)

    if len(cost_matrix) == 0:
        logging.warning("No feasible matches found")
        return [], timing

    logging.info(f"Cost matrix computed: {len(cost_matrix)} feasible (vehicle, request) pairs")

    # Phase C: Solve matching
    t_c = _time.perf_counter()
    matches = solve_milp_matching(vehQ, reqQ, cost_matrix,
                                  decision_log=decision_log)
    timing['phase_c_solve_s'] = _time.perf_counter() - t_c
    timing['solver'] = 'lp'
    timing['n_matches'] = len(matches)

    return matches, timing


def optimize_single_pair_pudo(sim, veh_id, req_id, params, decision_log=None):
    """
    Optimize PUDO nodes for a single (vehicle, request) pair.

    This is called AFTER greedy matching has selected the pair.
    No MILP solver needed - deterministic enumeration of feasible nodes.

    Args:
        sim: Simulator object
        veh_id: Vehicle ID (from greedy match)
        req_id: Request ID (from greedy match)
        params: Configuration parameters
        decision_log: Optional PudoDecisionLog for recording decisions

    Returns:
        dict: {veh_id, req_id, pickup_node, dropoff_node, savings,
               cost_d2d, cost_pudo, walk_to_pickup_dist, walk_from_dropoff_dist}
    """
    veh = sim.vehicles.loc[veh_id]
    req = sim.inData.requests.loc[req_id]

    # Find feasible PUDO nodes
    max_walk_dist = params.pudo.max_walking_distance
    walking_speed = params.pudo.walking_speed
    safe_nodes = getattr(sim.inData, 'safe_nodes', None)

    origin_nodes, origin_dists, origin_times = get_feasible_pudo_nodes(
        req.origin, max_walk_dist,
        sim.skims.walk, sim.skims.walk_dist, walking_speed,
        safe_nodes=safe_nodes
    )

    dest_nodes, dest_dists, dest_times = get_feasible_pudo_nodes(
        req.destination, max_walk_dist,
        sim.skims.walk, sim.skims.walk_dist, walking_speed,
        safe_nodes=safe_nodes
    )

    # Log feasibility
    if decision_log is not None:
        decision_log.log_feasibility(
            req_id=req_id, side='origin', anchor_node=req.origin,
            feasible_nodes=origin_nodes, walk_distances=origin_dists,
            walk_times=origin_times, max_walk_distance=max_walk_dist,
        )
        decision_log.log_feasibility(
            req_id=req_id, side='destination', anchor_node=req.destination,
            feasible_nodes=dest_nodes, walk_distances=dest_dists,
            walk_times=dest_times, max_walk_distance=max_walk_dist,
        )

    # Reuse existing calculate_pudo_costs() with single pair
    vehicles_df = sim.vehicles.loc[[veh_id]]
    requests_df = sim.inData.requests.loc[[req_id]]
    platform_fare = sim.inData.platforms.iloc[0].fare
    platform_fare_per_min = getattr(sim.inData.platforms.iloc[0], 'fare_per_min', 0.0)

    cost_matrix = calculate_pudo_costs(
        vehicles=vehicles_df,
        requests=requests_df,
        skim_dist=sim.skims.dist,
        skim_ride=sim.skims.ride,
        feasible_origins={req_id: origin_nodes},
        feasible_destinations={req_id: dest_nodes},
        params=params,
        fare_per_km=platform_fare, fare_per_min=platform_fare_per_min,
        decision_log=decision_log,
        skim_walk_dist=sim.skims.walk_dist,
    )

    # Return best match or D2D fallback
    if len(cost_matrix) > 0:
        match = cost_matrix.iloc[0].to_dict()

        # Add walking distances (undirected — pedestrians ignore one-way streets)
        match['walk_to_pickup_dist'] = sim.skims.walk_dist[req.origin][match['pickup_node']]
        match['walk_from_dropoff_dist'] = sim.skims.walk_dist[match['dropoff_node']][req.destination]

        logging.debug(f"Single-pair PUDO: veh={veh_id}, req={req_id}, savings={match['savings']:.2f}")

        return match
    else:
        # Fallback to D2D
        logging.debug(f"Single-pair D2D fallback: veh={veh_id}, req={req_id} (no PUDO savings)")
        return {
            'veh_id': veh_id,
            'req_id': req_id,
            'pickup_node': req.origin,
            'dropoff_node': req.destination,
            'savings': 0,
            'cost_d2d': 0,
            'cost_pudo': 0,
            'walk_to_pickup_dist': 0,
            'walk_from_dropoff_dist': 0
        }
