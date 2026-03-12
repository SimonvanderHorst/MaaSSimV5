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


class _SkimRow:
    """Row accessor for _SkimLookup — returned by __getitem__."""
    __slots__ = ('_arr', '_row_idx', '_col_pos')

    def __init__(self, arr, row_idx, col_pos):
        self._arr = arr
        self._row_idx = row_idx
        self._col_pos = col_pos

    def __getitem__(self, row):
        return float(self._arr[self._row_idx[row], self._col_pos])


class _SkimLookup:
    """Fast O(1) lookup wrapper around a skim DataFrame.

    Preserves ``df[col][row]`` semantics using NumPy array + two index maps.
    Handles transposed skims correctly (ride, walk use .T).
    """
    __slots__ = ('_arr', '_col_idx', '_row_idx')

    def __init__(self, df):
        self._arr = df.values
        self._col_idx = {node: i for i, node in enumerate(df.columns)}
        self._row_idx = {node: i for i, node in enumerate(df.index)}

    def __getitem__(self, col):
        return _SkimRow(self._arr, self._row_idx, self._col_idx[col])


def _compute_d2d_baseline(veh_pos, req_origin, req_destination,
                          skim_dist, skim_ride, cost_per_meter, cost_per_second):
    """Compute D2D baseline cost for a single (vehicle, request) pair.

    Returns dict with cost_d2d_total, dist_to_origin, dist_origin_to_dest,
    and direct_vehicle_path, or None if pre-filters fail.
    """
    dist_to_origin = skim_dist[req_origin][veh_pos]
    dist_origin_to_dest = skim_dist[req_destination][req_origin]

    time_to_origin = skim_ride[veh_pos][req_origin]
    time_origin_to_dest = skim_ride[req_origin][req_destination]
    cost_d2d_total = ((dist_to_origin + dist_origin_to_dest) * cost_per_meter +
                      (time_to_origin + time_origin_to_dest) * cost_per_second)

    direct_vehicle_path = dist_to_origin + dist_origin_to_dest

    return {
        'cost_d2d_total': cost_d2d_total,
        'dist_to_origin': dist_to_origin,
        'dist_origin_to_dest': dist_origin_to_dest,
        'direct_vehicle_path': direct_vehicle_path,
    }


def _evaluate_pudo_combo(veh_pos, pickup_node, dropoff_node, req_origin, req_destination,
                         skim_dist, skim_ride, _walk_dist,
                         cost_per_meter, cost_per_second, walking_cost_per_meter,
                         friction_cost, direct_vehicle_path, cost_d2d_total,
                         rider_aware, ra_params=None):
    """Evaluate a single PUDO pickup/dropoff combination.

    Args:
        ra_params: Dict with rider-aware params (walk_speed, beta_walk, beta_wait, beta_zero, alpha)
                   Required only when rider_aware=True.

    Returns dict with savings, ranking_score, rider_side_cost, cost components, and node details.
    """
    # Vehicle cost (distance + time)
    dist_to_pickup = skim_dist[pickup_node][veh_pos]
    dist_pickup_to_dropoff = skim_dist[dropoff_node][pickup_node]
    time_to_pickup = skim_ride[veh_pos][pickup_node]
    time_pickup_to_dropoff = skim_ride[pickup_node][dropoff_node]
    cost_pudo_vehicle = ((dist_to_pickup + dist_pickup_to_dropoff) * cost_per_meter +
                         (time_to_pickup + time_pickup_to_dropoff) * cost_per_second)

    # Walking cost (undirected — pedestrians ignore one-way streets)
    walk_to_pickup = _walk_dist[req_origin][pickup_node]
    walk_from_dropoff = _walk_dist[dropoff_node][req_destination]
    cost_pudo_walking = (walk_to_pickup + walk_from_dropoff) * walking_cost_per_meter

    # Total PUDO cost (vehicle operational only; walking gated by max_walking_distance)
    cost_pudo_total = cost_pudo_vehicle
    savings = cost_d2d_total - cost_pudo_total

    # rider-aware ranking — inspired by Ding et al. 2024,
    # "Incorporating walking into ride-hailing: flexible pick-up and drop-off"
    # alpha=0: minimize driving distance only; alpha=1: minimize walking distance
    if rider_aware and savings > 0 and ra_params is not None:
        t_walk_min = (walk_to_pickup + walk_from_dropoff) / ra_params['walk_speed'] / 60.0
        t_wait_savings_min = min(walk_to_pickup / ra_params['walk_speed'], time_to_pickup) / 60.0
        rider_side_cost = (ra_params['beta_walk'] * t_walk_min
                           - ra_params['beta_wait'] * t_wait_savings_min
                           + ra_params['beta_zero'])
        ranking_score = savings - ra_params['alpha'] * rider_side_cost
    else:
        rider_side_cost = 0.0
        ranking_score = savings

    return {
        'savings': savings,
        'ranking_score': ranking_score,
        'rider_side_cost': rider_side_cost,
        'cost_pudo_vehicle': cost_pudo_vehicle,
        'cost_pudo_walking': cost_pudo_walking,
        'cost_pudo_total': cost_pudo_total,
        'dist_to_pickup': dist_to_pickup,
        'dist_pickup_to_dropoff': dist_pickup_to_dropoff,
        'walk_to_pickup': walk_to_pickup,
        'walk_from_dropoff': walk_from_dropoff,
        'time_to_pickup': time_to_pickup,
    }


def calculate_pudo_costs(vehicles, requests, skim_dist, skim_ride,
                         feasible_origins, feasible_destinations, params,
                         fare_per_km, fare_per_min=0.0, decision_log=None,
                         skim_walk_dist=None):
    """
    Calculate HOLISTIC operational costs for all vehicle-request-PUDO combinations.
    Pre-selects the best PUDO pair for each (vehicle, request) combination to reduce MILP complexity.

    Accounts for vehicle cost (distance + time) + walking disutility

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
    walking_cost_per_meter = cost_per_meter * params.pudo.beta_walk
    friction_cost = params.pudo.friction_cost

    # Rider-aware optimization parameters
    rider_aware = params.pudo.get('rider_aware_optimization', False)
    ra_params = None
    if rider_aware:
        behavioral = params.pudo.get('behavioral', {})
        ra_params = {
            'walk_speed': params.pudo.walking_speed,
            'beta_walk': behavioral.get('rider_beta_walk_time', 0.24),
            'beta_wait': behavioral.get('rider_beta_wait', 0.22),
            'beta_zero': behavioral.get('rider_beta_zero', 0.0),
            'alpha': params.pudo.get('rider_cost_weight', 1.0),
        }

    max_dispatch_ratio = params.pudo.get('max_dispatch_ratio', float('inf'))
    max_dispatch_m = params.pudo.get('max_dispatch_distance', float('inf'))

    # Wrap skim DataFrames for fast NumPy-backed lookup in inner loops
    _skim_dist_d = _SkimLookup(skim_dist)
    _skim_ride_d = _SkimLookup(skim_ride)
    _walk_dist_d = _SkimLookup(_walk_dist)

    # raw arrays + index maps for vectorized lookups
    sd_a, sd_ri, sd_ci = _skim_dist_d._arr, _skim_dist_d._row_idx, _skim_dist_d._col_idx
    sr_a, sr_ri, sr_ci = _skim_ride_d._arr, _skim_ride_d._row_idx, _skim_ride_d._col_idx
    sw_a, sw_ri, sw_ci = _walk_dist_d._arr, _walk_dist_d._row_idx, _walk_dist_d._col_idx

    # pre-cache request-only D2D values + numpy index arrays for feasible nodes
    use_vec = decision_log is None
    _req_cache = {}
    for req_id, req in requests.iterrows():
        dist_od = _skim_dist_d[req.destination][req.origin]
        time_od = _skim_ride_d[req.origin][req.destination]
        pickups = feasible_origins.get(req_id, [req.origin])
        dropoffs = feasible_destinations.get(req_id, [req.destination])
        entry = (dist_od, time_od, pickups, dropoffs)
        if use_vec:
            entry += (
                np.array([sd_ri[n] for n in pickups]),   # p_sd_r
                np.array([sd_ci[n] for n in pickups]),   # p_sd_c
                np.array([sr_ri[n] for n in pickups]),   # p_sr_r
                np.array([sr_ci[n] for n in pickups]),   # p_sr_c
                np.array([sw_ri[n] for n in pickups]),   # p_sw_r
                np.array([sd_ci[n] for n in dropoffs]),  # d_sd_c
                np.array([sr_ri[n] for n in dropoffs]),  # d_sr_r
                np.array([sw_ci[n] for n in dropoffs]),  # d_sw_c
            )
        _req_cache[req_id] = entry

    for veh_id, veh in vehicles.iterrows():
        veh_pos = veh.pos
        for req_id, req in requests.iterrows():
            rc = _req_cache[req_id]
            dist_od, time_od = rc[0], rc[1]

            # D2D baseline (vehicle-dependent parts only)
            dist_to_origin = _skim_dist_d[req.origin][veh_pos]
            time_to_origin = _skim_ride_d[veh_pos][req.origin]
            cost_d2d = ((dist_to_origin + dist_od) * cost_per_meter +
                        (time_to_origin + time_od) * cost_per_second)

            # pre-filters
            if dist_od > 0 and dist_to_origin / dist_od > max_dispatch_ratio:
                continue
            if dist_to_origin > max_dispatch_m:
                continue

            pickups, dropoffs = rc[2], rc[3]

            if use_vec:
                # vectorized inner loop
                p_sd_r, p_sd_c, p_sr_r, p_sr_c, p_sw_r = rc[4:9] 
                d_sd_c, d_sr_r, d_sw_c = rc[9:12]
                D = len(dropoffs)

                veh_sd_r = sd_ri[veh_pos]
                veh_sr_c = sr_ci[veh_pos]
                orig_sw_c = sw_ci[req.origin]
                dest_sw_r = sw_ri[req.destination]

                # per-pickup (P,)
                dtp = sd_a[veh_sd_r, p_sd_c]
                ttp = sr_a[p_sr_r, veh_sr_c]
                wtp = sw_a[p_sw_r, orig_sw_c]

                # per-dropoff (D,)
                wfd = sw_a[dest_sw_r, d_sw_c]

                # per-combo (P, D)
                dp2d = sd_a[p_sd_r[:, None], d_sd_c[None, :]]
                tp2d = sr_a[d_sr_r[None, :], p_sr_c[:, None]]

                cost_pudo_arr = ((dtp[:, None] + dp2d) * cost_per_meter +
                                 (ttp[:, None] + tp2d) * cost_per_second)
                savings_arr = cost_d2d - cost_pudo_arr 

                if rider_aware and ra_params is not None: 
                    tw = wtp[:, None] + wfd[None, :] 
                    t_walk_min = tw / ra_params['walk_speed'] / 60.0
                    t_wait_min = np.minimum(
                        wtp[:, None] / ra_params['walk_speed'],
                        ttp[:, None]) / 60.0
                    rider_cost_arr = (ra_params['beta_walk'] * t_walk_min
                                      - ra_params['beta_wait'] * t_wait_min
                                      + ra_params['beta_zero'])
                    ranking_arr = np.where(savings_arr > 0,
                                           savings_arr - ra_params['alpha'] * rider_cost_arr,
                                           savings_arr)
                else:
                    ranking_arr = savings_arr

                best_flat = int(np.argmax(ranking_arr))
                best_p, best_d = divmod(best_flat, D)
                best_ranking = float(ranking_arr.flat[best_flat])

                if best_ranking > 0:
                    best_savings = float(savings_arr.flat[best_flat])
                    best_pickup = pickups[best_p]
                    best_dropoff = dropoffs[best_d]
                    best_cost_pudo = float(cost_pudo_arr.flat[best_flat])
                    if rider_aware and ra_params is not None and best_savings > 0:
                        best_rider_side_cost = float(rider_cost_arr.flat[best_flat])
                    else:
                        best_rider_side_cost = 0.0
                else:
                    best_savings = 0.0
                    best_ranking = 0.0
                    best_rider_side_cost = 0.0
                    best_pickup = req.origin
                    best_dropoff = req.destination
                    best_cost_pudo = cost_d2d

                results.append({
                    'veh_id': veh_id,
                    'req_id': req_id,
                    'savings': best_savings,
                    'pickup_node': best_pickup,
                    'dropoff_node': best_dropoff,
                    'cost_d2d': cost_d2d,
                    'cost_pudo': best_cost_pudo,
                    'rider_side_cost': best_rider_side_cost,
                    'ranking_score': best_ranking,
                })

            else:
                # --- scalar fallback with per-combo decision logging ---
                direct_vehicle_path = dist_to_origin + dist_od

                best_savings = 0
                best_ranking = 0
                best_rider_side_cost = 0.0
                best_pickup = req.origin
                best_dropoff = req.destination
                best_cost_pudo = cost_d2d
                best_components = None
                n_combos = 0

                for pickup_node in pickups:
                    for dropoff_node in dropoffs:
                        n_combos += 1
                        combo = _evaluate_pudo_combo(
                            veh_pos, pickup_node, dropoff_node,
                            req.origin, req.destination,
                            _skim_dist_d, _skim_ride_d, _walk_dist_d,
                            cost_per_meter, cost_per_second, walking_cost_per_meter,
                            friction_cost, direct_vehicle_path, cost_d2d,
                            rider_aware, ra_params)

                        log_kwargs = dict(
                            veh_id=veh_id, req_id=req_id,
                            pickup_node=pickup_node, dropoff_node=dropoff_node,
                            dist_veh_to_origin=dist_to_origin,
                            dist_origin_to_dest=dist_od,
                            cost_d2d=cost_d2d,
                            dist_veh_to_pickup=combo['dist_to_pickup'],
                            dist_pickup_to_dropoff=combo['dist_pickup_to_dropoff'],
                            cost_pudo_vehicle=combo['cost_pudo_vehicle'],
                            walk_to_pickup_m=combo['walk_to_pickup'],
                            walk_from_dropoff_m=combo['walk_from_dropoff'],
                            cost_pudo_walking=combo['cost_pudo_walking'],
                            cost_pudo_total=combo['cost_pudo_total'],
                            savings=combo['savings'],
                        )
                        if rider_aware and combo['savings'] > 0:
                            log_kwargs['rider_side_cost'] = combo['rider_side_cost']
                            log_kwargs['ranking_score'] = combo['ranking_score']
                        decision_log.log_cost_detail(**log_kwargs)

                        if combo['ranking_score'] > best_ranking:
                            best_ranking = combo['ranking_score']
                            best_savings = combo['savings']
                            best_rider_side_cost = combo['rider_side_cost']
                            best_pickup = pickup_node
                            best_dropoff = dropoff_node
                            best_cost_pudo = combo['cost_pudo_total']
                            best_components = {
                                'cost_vehicle_driving': combo['cost_pudo_vehicle'],
                                'cost_walking': combo['cost_pudo_walking'],
                                'walk_to_pickup_m': combo['walk_to_pickup'],
                                'walk_from_dropoff_m': combo['walk_from_dropoff'],
                                'dist_veh_to_pickup_m': combo['dist_to_pickup'],
                                'dist_pickup_to_dropoff_m': combo['dist_pickup_to_dropoff'],
                            }
                            if rider_aware:
                                best_components['rider_side_cost'] = combo['rider_side_cost']
                                best_components['ranking_score'] = combo['ranking_score']

                if best_savings >= 0:
                    results.append({
                        'veh_id': veh_id,
                        'req_id': req_id,
                        'savings': best_savings,
                        'pickup_node': best_pickup,
                        'dropoff_node': best_dropoff,
                        'cost_d2d': cost_d2d,
                        'cost_pudo': best_cost_pudo,
                        'rider_side_cost': best_rider_side_cost,
                        'ranking_score': best_ranking,
                    })

                    decision_log.log_best_edge(
                        veh_id=veh_id, req_id=req_id,
                        best_pickup=best_pickup, best_dropoff=best_dropoff,
                        cost_d2d=cost_d2d, cost_pudo=best_cost_pudo,
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


