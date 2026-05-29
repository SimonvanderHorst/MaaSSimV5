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


def get_feasible_pudo_nodes(node_id, max_walk_distance, walk_lookup, walk_dist_lookup,
                            walking_speed, safe_mask=None):
    """Find all nodes within walking distance of a given node."""
    col = walk_dist_lookup._col_idx[node_id]
    dists = walk_dist_lookup._arr[:, col]
    mask = dists <= max_walk_distance
    if safe_mask is not None:
        mask = mask & safe_mask

    rows = np.where(mask)[0]
    node_list = walk_dist_lookup._row_nodes[rows].tolist()
    dist_dict = {int(n): float(dists[r]) for n, r in zip(node_list, rows)}

    walk_col = walk_lookup._col_idx[node_id]
    times = walk_lookup._arr[:, walk_col]
    time_dict = {int(n): float(times[r]) for n, r in zip(node_list, rows)}

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
    __slots__ = ('_arr', '_col_idx', '_row_idx', '_row_nodes')

    def __init__(self, df):
        self._arr = df.values
        self._col_idx = {node: i for i, node in enumerate(df.columns)}
        self._row_idx = {node: i for i, node in enumerate(df.index)}
        self._row_nodes = np.array(list(df.index))

    def __getitem__(self, col):
        return _SkimRow(self._arr, self._row_idx, self._col_idx[col])


def _compute_d2d_baseline(veh_pos, req_origin, req_destination,
                          skim_dist, skim_ride, cost_per_meter, cost_per_second):
    """Compute D2D baseline cost for a single (vehicle, request) pair.

    Returns dict with cost_d2d_total, dist_to_origin, dist_origin_to_dest,
    and direct_vehicle_path, or None if pre-filters fail.
    """
    dist_to_origin = skim_dist[veh_pos][req_origin]
    dist_origin_to_dest = skim_dist[req_origin][req_destination]

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
                         friction_cost, direct_vehicle_path, cost_d2d_total):
    """Evaluate a single PUDO pickup/dropoff combination."""
    # Vehicle cost (distance + time)
    dist_to_pickup = skim_dist[veh_pos][pickup_node]
    dist_pickup_to_dropoff = skim_dist[pickup_node][dropoff_node]
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

    return {
        'savings': savings,
        'ranking_score': savings,
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
                         skim_walk_dist=None,
                         _lookup_dist=None, _lookup_ride=None, _lookup_walk=None,
                         feasible_origin_times=None, feasible_dest_times=None,
                         ride_speed=None):
    """
    Calculate HOLISTIC operational costs for all vehicle-request-PUDO combinations.
    Pre-selects the best PUDO pair for each (vehicle, request) combination to reduce MILP complexity.

    Accounts for vehicle cost (distance + time) + walking disutility

    Args:
        vehicles: DataFrame of available vehicles with 'pos' column
        requests: DataFrame of requests with 'origin', 'destination'
        skim_dist: Distance matrix (meters); dist[A][B] = distance from A to B
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
    from MaaSSim.decisions._helpers import WAIT_DIST_LIMIT_M, MIN_FARE_EUR

    results = []
    _walk_dist = skim_walk_dist if skim_walk_dist is not None else skim_dist  # backward compat
    cost_per_meter = fare_per_km / 1000.0
    cost_per_second = fare_per_min / 60.0  # EUR/s from EUR/min
    walking_cost_per_meter = cost_per_meter * params.pudo.beta_walk
    friction_cost = params.pudo.friction_cost

    max_dispatch_ratio = params.pudo.get('max_dispatch_ratio', float('inf'))
    max_dispatch_m = params.pudo.get('max_dispatch_distance', float('inf'))

    # S1 thresholds — same delta as _d2d_ratio_check, same caps as _driver_s1_check
    delta_s1 = params.pudo.get('rider_s1_delta', 0.8)
    wait_limit_s = WAIT_DIST_LIMIT_M / ride_speed if ride_speed else float('inf')
    patience_s = params.times.driver_pickup_patience
    transaction_s = params.times.transaction
    s1_stats = {
        's1_cells_total': 0,
        's1_cells_rider_fail': 0,
        's1_cells_driver_wait_fail': 0,
        's1_cells_fare_fail': 0,
        's1_cells_walk_patience_fail': 0,
        's1_cells_feasible': 0,
    }

    # Reuse pre-built lookups if provided, otherwise wrap DataFrames
    _skim_dist_d = _lookup_dist or _SkimLookup(skim_dist)
    _skim_ride_d = _lookup_ride or _SkimLookup(skim_ride)
    _walk_dist_d = _lookup_walk or _SkimLookup(_walk_dist)

    # raw arrays + index maps for vectorized lookups
    sd_a, sd_ri, sd_ci = _skim_dist_d._arr, _skim_dist_d._row_idx, _skim_dist_d._col_idx
    sr_a, sr_ri, sr_ci = _skim_ride_d._arr, _skim_ride_d._row_idx, _skim_ride_d._col_idx
    sw_a, sw_ri, sw_ci = _walk_dist_d._arr, _walk_dist_d._row_idx, _walk_dist_d._col_idx

    # pre-cache request-only D2D values + numpy index arrays for feasible nodes
    # all four skims share [row=dest, col=orig] numpy layout: arr[ri[dest], ci[orig]] = cost(orig→dest)
    use_vec = decision_log is None
    _req_cache = {}
    _req_walk = {}  # req_id -> (wt_pickup (P,), wt_dropoff (D,)) walk times in seconds
    _req_ttrav = {}  # req_id -> request.ttrav in seconds (dist/ride_speed); matches _d2d_ratio_check
    for req_id, req in requests.iterrows():
        dist_od = _skim_dist_d[req.origin][req.destination]
        time_od = _skim_ride_d[req.origin][req.destination]
        # request.ttrav is dist / speeds.ride (uniform), matches _d2d_ratio_check denominator
        ttrav_s = (dist_od / ride_speed) if ride_speed else time_od
        pickups = feasible_origins.get(req_id, [req.origin])
        dropoffs = feasible_destinations.get(req_id, [req.destination])
        entry = (dist_od, time_od, pickups, dropoffs)
        if use_vec:
            entry += (
                np.array([sd_ri[n] for n in pickups]),   # p_sd_r (pickup as dest)
                np.array([sd_ci[n] for n in pickups]),   # p_sd_c (pickup as orig)
                np.array([sr_ri[n] for n in pickups]),   # p_sr_r
                np.array([sr_ci[n] for n in pickups]),   # p_sr_c
                np.array([sw_ri[n] for n in pickups]),   # p_sw_r
                np.array([sd_ri[n] for n in dropoffs]),  # d_sd_r (dropoff as dest)
                np.array([sr_ri[n] for n in dropoffs]),  # d_sr_r
                np.array([sw_ci[n] for n in dropoffs]),  # d_sw_c
            )
        _req_cache[req_id] = entry

        # walk times for S1 mask; origin==pickup → 0, else from Phase A dict
        ot = (feasible_origin_times or {}).get(req_id, {})
        dt = (feasible_dest_times or {}).get(req_id, {})
        wt_pickup = np.array([ot.get(int(n), 0.0) for n in pickups], dtype=float)
        wt_dropoff = np.array([dt.get(int(n), 0.0) for n in dropoffs], dtype=float)
        _req_walk[req_id] = (wt_pickup, wt_dropoff)
        _req_ttrav[req_id] = ttrav_s

    if use_vec:
        # vectorized path — loop over requests, all vehicles at once
        veh_ids = vehicles.index.tolist()
        veh_positions = vehicles['pos'].values
        V = len(veh_ids)
        veh_sd_c = np.array([sd_ci[p] for p in veh_positions])  # (V,) veh as orig
        veh_sr_c = np.array([sr_ci[p] for p in veh_positions])  # (V,) veh as orig

        for req_id, req in requests.iterrows():
            rc = _req_cache[req_id]
            dist_od, time_od = rc[0], rc[1]
            pickups, dropoffs = rc[2], rc[3]
            p_sd_r, p_sd_c, p_sr_r, p_sr_c, p_sw_r = rc[4:9]
            d_sd_r, d_sr_r, d_sw_c = rc[9:12]
            D = len(dropoffs)

            # request-only arrays — computed once, not V times
            orig_sw_c = sw_ci[req.origin]
            dest_sw_r = sw_ri[req.destination]
            dp2d = sd_a[d_sd_r[None, :], p_sd_c[:, None]]        # (P, D) dist(pickup→dropoff)
            tp2d = sr_a[d_sr_r[None, :], p_sr_c[:, None]]        # (P, D) time(pickup→dropoff)

            # D2D baseline for all vehicles
            orig_sd_r = sd_ri[req.origin]
            orig_sr_r = sr_ri[req.origin]
            dist_to_origin = sd_a[orig_sd_r, veh_sd_c]           # (V,) dist(veh→origin)
            time_to_origin = sr_a[orig_sr_r, veh_sr_c]           # (V,) time(veh→origin)
            cost_d2d = ((dist_to_origin + dist_od) * cost_per_meter +
                        (time_to_origin + time_od) * cost_per_second)  # (V,)

            # pre-filter mask
            mask = dist_to_origin <= max_dispatch_m               # (V,)
            if dist_od > 0:
                mask &= (dist_to_origin / dist_od) <= max_dispatch_ratio
            valid_idx = np.where(mask)[0]
            if len(valid_idx) == 0:
                continue

            # vehicle-to-pickup for valid vehicles only
            dtp = sd_a[p_sd_r[None, :], veh_sd_c[valid_idx, None]]  # (V_v, P) dist(veh→pickup)
            ttp = sr_a[p_sr_r[None, :], veh_sr_c[valid_idx, None]]  # (V_v, P) time(veh→pickup)

            # PUDO cost + savings via broadcasting
            cost_pudo = ((dtp[:, :, None] + dp2d[None, :, :]) * cost_per_meter +
                         (ttp[:, :, None] + tp2d[None, :, :]) * cost_per_second)  # (V_v, P, D)
            savings = cost_d2d[valid_idx, None, None] - cost_pudo                  # (V_v, P, D)

            # ── S1 mask: rider ratio + driver wait cap + fare floor ──
            wt_pickup, wt_dropoff = _req_walk[req_id]               # (P,), (D,)
            # parallel overhead: max(walk_to, dispatch) + walk_from
            overhead = np.maximum(wt_pickup[None, :, None], ttp[:, :, None]) \
                       + wt_dropoff[None, None, :]                  # (V_v, P, D)
            # denominator matches _d2d_ratio_check: request.ttrav = dist / speeds.ride
            ttrav_s = _req_ttrav[req_id]
            rider_s1_fail = overhead > delta_s1 * ttrav_s  # (V_v, P, D)
            driver_wait_fail = (ttp >= wait_limit_s)[:, :, None]    # (V_v, P, 1)
            # fare floor on baseline (pre-DQN) fare: dist*cpm + ttrav*cps; ttrav is
            # the d2d ride time, not pickup→dropoff. Apply per (v,r), not per (p,d).
            fare_baseline = ((dist_to_origin[valid_idx] + dist_od) * cost_per_meter +
                             (time_to_origin[valid_idx] + time_od) * cost_per_second)
            fare_fail = (fare_baseline < MIN_FARE_EUR)[:, None, None]  # (V_v, 1, 1)
            # walk-patience: pax must reach pickup before driver gives up
            # pax_arrival = transaction + walk_to, driver_deadline = drive_to + patience
            walk_patience_fail = (wt_pickup[None, :, None] + transaction_s
                                  > ttp[:, :, None] + patience_s)        # (V_v, P, 1)

            infeasible = rider_s1_fail | driver_wait_fail | fare_fail | walk_patience_fail
            s1_stats['s1_cells_total'] += int(infeasible.size)
            s1_stats['s1_cells_rider_fail'] += int(rider_s1_fail.sum())
            s1_stats['s1_cells_driver_wait_fail'] += int(driver_wait_fail.sum() * infeasible.shape[2])
            s1_stats['s1_cells_fare_fail'] += int(fare_fail.sum() * infeasible.shape[1] * infeasible.shape[2])
            s1_stats['s1_cells_walk_patience_fail'] += int(walk_patience_fail.sum() * infeasible.shape[2])
            s1_stats['s1_cells_feasible'] += int((~infeasible).sum())
            savings = np.where(infeasible, -np.inf, savings)

            # best combo per vehicle
            V_v = len(valid_idx)
            flat = savings.reshape(V_v, -1)                       # (V_v, P*D)
            best_flat = np.argmax(flat, axis=1)                   # (V_v,)
            arange_v = np.arange(V_v)
            best_savings = flat[arange_v, best_flat]              # (V_v,)
            best_p, best_d = np.divmod(best_flat, D)
            best_cost_pudo = cost_pudo.reshape(V_v, -1)[arange_v, best_flat]

            # D2D-baseline S1 per vehicle (no walking; overhead = dispatch time)
            ttp_d2d = time_to_origin[valid_idx]                                # (V_v,)
            d2d_rider_fail = ttp_d2d > delta_s1 * ttrav_s if ttrav_s > 0 else np.zeros_like(ttp_d2d, dtype=bool)
            d2d_driver_wait_fail = ttp_d2d >= wait_limit_s
            d2d_fare_fail = fare_baseline < MIN_FARE_EUR
            d2d_feasible = ~(d2d_rider_fail | d2d_driver_wait_fail | d2d_fare_fail)

            for j in range(V_v):
                vi = valid_idx[j]
                if best_savings[j] > 0:
                    results.append({
                        'veh_id': veh_ids[vi],
                        'req_id': req_id,
                        'savings': float(best_savings[j]),
                        'pickup_node': pickups[int(best_p[j])],
                        'dropoff_node': dropoffs[int(best_d[j])],
                        'cost_d2d': float(cost_d2d[vi]),
                        'cost_pudo': float(best_cost_pudo[j]),
                        'rider_side_cost': 0.0,
                        'ranking_score': float(best_savings[j]),
                        'time_to_origin': float(time_to_origin[vi]),
                    })
                elif d2d_feasible[j]:
                    # PUDO not profitable (or all infeasible) but D2D itself is S1-feasible
                    results.append({
                        'veh_id': veh_ids[vi],
                        'req_id': req_id,
                        'savings': 0.0,
                        'pickup_node': req.origin,
                        'dropoff_node': req.destination,
                        'cost_d2d': float(cost_d2d[vi]),
                        'cost_pudo': float(cost_d2d[vi]),
                        'rider_side_cost': 0.0,
                        'ranking_score': 0.0,
                        'time_to_origin': float(time_to_origin[vi]),
                    })
                # else: drop entirely — neither PUDO nor D2D is S1-feasible for this (v,r)

    else:
        # scalar fallback with per-combo decision logging
        for veh_id, veh in vehicles.iterrows():
            veh_pos = veh.pos
            for req_id, req in requests.iterrows():
                rc = _req_cache[req_id]
                dist_od, time_od = rc[0], rc[1]

                dist_to_origin = _skim_dist_d[veh_pos][req.origin]
                time_to_origin = _skim_ride_d[veh_pos][req.origin]
                cost_d2d = ((dist_to_origin + dist_od) * cost_per_meter +
                            (time_to_origin + time_od) * cost_per_second)

                if dist_od > 0 and dist_to_origin / dist_od > max_dispatch_ratio:
                    continue
                if dist_to_origin > max_dispatch_m:
                    continue

                pickups, dropoffs = rc[2], rc[3]
                direct_vehicle_path = dist_to_origin + dist_od
                wt_pickup_d, wt_dropoff_d = _req_walk[req_id]

                # baseline fare floor for this (v,r); applies to PUDO and D2D alike
                fare_baseline_s = ((dist_to_origin + dist_od) * cost_per_meter +
                                   (time_to_origin + time_od) * cost_per_second)
                fare_fail_vr = fare_baseline_s < MIN_FARE_EUR
                driver_wait_fail_vr = time_to_origin >= wait_limit_s

                best_savings = 0
                best_ranking = 0
                best_pickup = req.origin
                best_dropoff = req.destination
                best_cost_pudo = cost_d2d
                best_components = None
                n_combos = 0

                for p_idx, pickup_node in enumerate(pickups):
                    for d_idx, dropoff_node in enumerate(dropoffs):
                        n_combos += 1
                        combo = _evaluate_pudo_combo(
                            veh_pos, pickup_node, dropoff_node,
                            req.origin, req.destination,
                            _skim_dist_d, _skim_ride_d, _walk_dist_d,
                            cost_per_meter, cost_per_second, walking_cost_per_meter,
                            friction_cost, direct_vehicle_path, cost_d2d)

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
                        decision_log.log_cost_detail(**log_kwargs)

                        # ── S1 mask: skip combo if infeasible ──
                        s1_stats['s1_cells_total'] += 1
                        # parallel overhead: max(walk_to, dispatch) + walk_from
                        # denominator matches _d2d_ratio_check: request.ttrav = dist / speeds.ride
                        overhead = max(wt_pickup_d[p_idx], combo['time_to_pickup']) + wt_dropoff_d[d_idx]
                        ttrav_s = _req_ttrav[req_id]
                        if ttrav_s > 0 and overhead > delta_s1 * ttrav_s:
                            s1_stats['s1_cells_rider_fail'] += 1
                            continue
                        if combo['time_to_pickup'] >= wait_limit_s:
                            s1_stats['s1_cells_driver_wait_fail'] += 1
                            continue
                        if fare_fail_vr:
                            s1_stats['s1_cells_fare_fail'] += 1
                            continue
                        if wt_pickup_d[p_idx] + transaction_s > combo['time_to_pickup'] + patience_s:
                            s1_stats['s1_cells_walk_patience_fail'] += 1
                            continue
                        s1_stats['s1_cells_feasible'] += 1

                        if combo['ranking_score'] > best_ranking:
                            best_ranking = combo['ranking_score']
                            best_savings = combo['savings']
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

                # D2D-baseline S1 (no walking; overhead = dispatch time)
                ttrav_s = _req_ttrav[req_id]
                d2d_rider_fail_vr = (ttrav_s > 0 and time_to_origin > delta_s1 * ttrav_s)
                d2d_feasible_vr = not (d2d_rider_fail_vr or driver_wait_fail_vr or fare_fail_vr)

                if best_savings > 0:
                    results.append({
                        'veh_id': veh_id,
                        'req_id': req_id,
                        'savings': best_savings,
                        'pickup_node': best_pickup,
                        'dropoff_node': best_dropoff,
                        'cost_d2d': cost_d2d,
                        'cost_pudo': best_cost_pudo,
                        'rider_side_cost': 0.0,
                        'ranking_score': best_ranking,
                        'time_to_origin': time_to_origin,
                    })

                    decision_log.log_best_edge(
                        veh_id=veh_id, req_id=req_id,
                        best_pickup=best_pickup, best_dropoff=best_dropoff,
                        cost_d2d=cost_d2d, cost_pudo=best_cost_pudo,
                        savings=best_savings, n_combos_evaluated=n_combos,
                        cost_components=best_components,
                    )
                elif d2d_feasible_vr:
                    results.append({
                        'veh_id': veh_id,
                        'req_id': req_id,
                        'savings': 0.0,
                        'pickup_node': req.origin,
                        'dropoff_node': req.destination,
                        'cost_d2d': cost_d2d,
                        'cost_pudo': cost_d2d,
                        'rider_side_cost': 0.0,
                        'ranking_score': 0.0,
                        'time_to_origin': time_to_origin,
                    })

    return pd.DataFrame(results), s1_stats


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
            matches.append(match)
    return matches


def _log_milp(decision_log, cost_matrix, M, selected_vq_pairs,
              n_vars, n_veh_constraints, n_req_constraints,
              solver_status, solver_time_s,
              coeff_lookup=None, tie_break_lookup=None):
    """Shared MILP logging for all solver backends."""
    selected_set = set(selected_vq_pairs)
    obj_coefficients = []
    selected_pairs = []
    unselected_pairs = []
    for _, row in cost_matrix.iterrows():
        v, q = row['veh_id'], row['req_id']
        key = (v, q)
        if coeff_lookup is not None:
            coeff = float(coeff_lookup[key])
            tie_break = float(tie_break_lookup[key]) if tie_break_lookup is not None else 0.0
        else:
            coeff = float(M + row['ranking_score'])
            tie_break = 0.0
        entry = {
            'veh_id': int(v), 'req_id': int(q),
            'ranking_score': float(row['ranking_score']),
            'tie_break': tie_break, 'coeff': coeff,
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
    Maximize: Σ (M - cost_pudo[v,q]) · x_v,q
    Pure min-cost within max cardinality. cost_pudo already includes vehicle
    dist + time, so this directly minimizes total vehicle-occupied resource.
    Subject to:
        Σ_q x_v,q <= 1  for all vehicles v
        Σ_v x_v,q <= 1  for all requests q
        x_v,q ∈ [0, 1]
    """
    pairs = list(zip(cost_matrix['veh_id'], cost_matrix['req_id']))

    # pure min-cost: maximize -cost_pudo (cardinality dominates via big-M)
    cost_col = cost_matrix['cost_pudo'].to_numpy()
    M = float(cost_col.max() * 2 + 1)
    coeffs = M - cost_col
    coeff_lookup = dict(zip(pairs, coeffs))
    tie_break_lookup = None

    for attempt in range(4):
        try:
            model = gp.Model("PUDO_Matching")
            model.setParam('OutputFlag', 0)

            x = model.addVars(pairs, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x")

            model.setObjective(
                gp.quicksum(coeff_lookup[p] * x[p] for p in pairs),
                gp.GRB.MAXIMIZE
            )

            # each vehicle serves at most one request
            veh_pairs = {}
            for v, q in pairs:
                veh_pairs.setdefault(v, []).append((v, q))
            n_vc = 0
            for v in vehicles:
                if v in veh_pairs:
                    model.addConstr(gp.quicksum(x[p] for p in veh_pairs[v]) <= 1)
                    n_vc += 1

            # each request served by at most one vehicle
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
            break
        except gp.GurobiError as e:
            if attempt < 3:
                wait = 5 * (2 ** attempt)
                logging.warning(f"gurobi error (attempt {attempt+1}/4): {e}, retry in {wait}s")
                _time.sleep(wait)
            else:
                logging.error(f"gurobi failed after 4 attempts: {e}")
                return []

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
                  len(pairs), n_vc, n_rc, solver_status, t_solve,
                  coeff_lookup=coeff_lookup, tie_break_lookup=tie_break_lookup)

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

    # Phase A: Find feasible PUDO nodes for each request (cached by node_id)
    t_a = _time.perf_counter()
    feasible_origins = {}
    feasible_destinations = {}
    feasible_origin_times = {}   # req_id -> {node: walk_time_s}
    feasible_dest_times = {}     # req_id -> {node: walk_time_s}
    max_walk_dist = params.pudo.max_walking_distance
    walking_speed = params.pudo.walking_speed

    safe_nodes = getattr(sim.inData, 'safe_nodes', None)
    walk_lookup = sim.skims._walk
    walk_dist_lookup = sim.skims._walk_dist

    # pre-compute boolean mask once for safe_nodes filtering
    if safe_nodes is not None:
        safe_mask = np.array([n in safe_nodes for n in walk_dist_lookup._row_nodes])
    else:
        safe_mask = None

    # cache feasible nodes by node_id (walk network doesn't change mid-episode)
    _feas_cache = getattr(sim, '_feasible_cache', None)
    if _feas_cache is None:
        _feas_cache = {}
        sim._feasible_cache = _feas_cache

    for req_id in reqQ:
        req = requests.loc[req_id]

        if req.origin in _feas_cache:
            origin_nodes, origin_dists, origin_times = _feas_cache[req.origin]
        else:
            origin_nodes, origin_dists, origin_times = get_feasible_pudo_nodes(
                req.origin, max_walk_dist,
                walk_lookup, walk_dist_lookup, walking_speed,
                safe_mask=safe_mask
            )
            _feas_cache[req.origin] = (origin_nodes, origin_dists, origin_times)
        feasible_origins[req_id] = origin_nodes
        feasible_origin_times[req_id] = origin_times

        if req.destination in _feas_cache:
            dest_nodes, dest_dists, dest_times = _feas_cache[req.destination]
        else:
            dest_nodes, dest_dists, dest_times = get_feasible_pudo_nodes(
                req.destination, max_walk_dist,
                walk_lookup, walk_dist_lookup, walking_speed,
                safe_mask=safe_mask
            )
            _feas_cache[req.destination] = (dest_nodes, dest_dists, dest_times)
        feasible_destinations[req_id] = dest_nodes
        feasible_dest_times[req_id] = dest_times

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
    cost_matrix, s1_stats = calculate_pudo_costs(
        vehicles, requests, sim.skims.dist, sim.skims.ride,
        feasible_origins, feasible_destinations, params,
        fare_per_km=platform_fare, fare_per_min=platform_fare_per_min,
        decision_log=decision_log,
        skim_walk_dist=sim.skims.walk_dist,
        _lookup_dist=sim.skims._dist,
        _lookup_ride=sim.skims._ride,
        _lookup_walk=sim.skims._walk_dist,
        feasible_origin_times=feasible_origin_times,
        feasible_dest_times=feasible_dest_times,
        ride_speed=params.speeds.ride,
    )
    timing.update(s1_stats)
    timing['phase_b_cost_matrix_s'] = _time.perf_counter() - t_b
    timing['n_cost_matrix_rows'] = len(cost_matrix)

    # opt-in connectivity stats (off by default; runner sets sim._log_connectivity=True)
    if getattr(sim, '_log_connectivity', False) and len(cost_matrix) > 0:
        epv = cost_matrix.groupby('veh_id').size()
        epr = cost_matrix.groupby('req_id').size()
        timing['n_feasible_vehs'] = int(len(epv))
        timing['n_feasible_reqs'] = int(len(epr))
        timing['n_vehs_with_choice'] = int((epv >= 2).sum())
        timing['n_reqs_with_choice'] = int((epr >= 2).sum())
        timing['median_edges_per_veh'] = float(epv.median())
        timing['median_edges_per_req'] = float(epr.median())
        timing['max_edges_per_veh'] = int(epv.max())
        timing['max_edges_per_req'] = int(epr.max())

    # per-batch req-without-any-cost-matrix-row tally
    if len(cost_matrix) == 0:
        reqs_without_row = set(reqQ)
    else:
        reqs_without_row = set(reqQ) - set(cost_matrix['req_id'].unique())
    sim.unserved_by_reason['no_feasible_candidate_in_batch'] += len(reqs_without_row)

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

    # cost_matrix had feasible rows but solver returned nothing → treat as milp failure
    if len(cost_matrix) > 0 and len(matches) == 0:
        sim.unserved_by_reason['milp_failed'] += len(set(cost_matrix['req_id'].unique()))

    return matches, timing


