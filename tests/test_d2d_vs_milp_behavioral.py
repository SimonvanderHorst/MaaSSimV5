#!/usr/bin/env python
"""
Condensed same-seed comparison: D2D baseline vs Batch MILP PUDO
with behavioral acceptance model enabled.

Single run, shared demand, text summary to stdout. Also extracts per-request outcomes and merges decision logs for visualization.

Usage:
    python tests/test_d2d_vs_milp_behavioral.py
"""
import sys
import os
import copy
import time
import numpy as np
import networkx as nx
import pandas as pd
from dotmap import DotMap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from MaaSSim.data_structures import structures
from MaaSSim.maassim import Simulator
from MaaSSim.simulators import _prep_rides
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df
from MaaSSim.performance import calculate_vkt, calculate_pudo_metrics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_pudo_test.json')


# ---------------------------------------------------------------------------
# Shared demand generation
# ---------------------------------------------------------------------------

def generate_shared_demand():
    """Generate demand once for both strategies to share."""
    params = get_config(CONFIG_PATH)
    inData = structures.copy()
    inData = load_G(inData, params, stats=True)
    inData = generate_demand(inData, params, avg_speed=True)
    inData.vehicles = generate_vehicles(inData, params.nV)

    if len(inData.platforms) == 0:
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0, 'fare'] = 1.20
        inData.platforms.loc[0, 'fare_per_min'] = 0.42
        inData.platforms.loc[0, 'name'] = 'Platform'
        inData.platforms.loc[0, 'batch_time'] = getattr(params, 'batch_time', 60)

    inData = _prep_rides(inData)
    return inData, params


def copy_indata(inData_base):
    """Deep-copy mutable DataFrames; share read-only graph/skim."""
    c = DotMap()
    c.G = inData_base.G
    c.nodes = inData_base.nodes
    c.skim = inData_base.skim
    c.stats = inData_base.stats
    c.passengers = inData_base.passengers.copy()
    c.requests = inData_base.requests.copy()
    c.vehicles = inData_base.vehicles.copy()
    c.platforms = inData_base.platforms.copy()
    if 'sim_schedule' in c.requests.columns:
        c.requests['sim_schedule'] = c.requests['sim_schedule'].apply(
            lambda x: x.copy() if isinstance(x, pd.DataFrame) else x
        )
    for attr in ('walk_dist', 'ride_time'):
        if isinstance(getattr(inData_base, attr, None), pd.DataFrame):
            setattr(c, attr, getattr(inData_base, attr))
    if hasattr(inData_base, 'safe_nodes'):
        c.safe_nodes = inData_base.safe_nodes
    return c


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_d2d(inData_base, params_base):
    inData = copy_indata(inData_base)
    params = copy.deepcopy(params_base)
    params.pudo.enabled = False
    sim = Simulator(inData, params=params, logger_level='WARNING')
    sim.make_and_run(run_id=0)
    sim.output()
    return sim


def run_milp_pudo(inData_base, params_base):
    inData = copy_indata(inData_base)
    params = copy.deepcopy(params_base)
    params.pudo.enabled = True
    params.pudo.greedy_first = False
    params.pudo.behavioral.enabled = True
    params.pudo.d2d_fallback = True
    params.pudo.decision_log_level = 'summary'
    sim = Simulator(inData, params=params, logger_level='WARNING')
    sim.make_and_run(run_id=0)
    sim.output()
    return sim


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

def _scalar(x):
    if isinstance(x, np.ndarray):
        return float(x.flat[0])
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)


def collect_kpis(sim, label):
    vkt = _scalar(calculate_vkt(sim, run_id=0))
    pax_km = _scalar(sim.res[0]['veh_exp']['PAX_KM'].sum())
    cruise_km = max(0.0, vkt - pax_km)
    n_rides = _scalar(sim.res[0]['veh_kpi'].loc['sum', 'nRIDES'])
    avg_wait = _scalar(sim.res[0]['pax_kpi'].loc['mean', 'WAIT'])
    revenue = _scalar(sim.res[0]['all_kpi']['total_revenue'])

    kpis = dict(label=label, vkt=vkt, pax_km=pax_km, cruise_km=cruise_km,
                n_rides=n_rides, avg_wait=avg_wait, revenue=revenue)

    if sim.params.get('pudo', {}).get('enabled', False):
        pm = calculate_pudo_metrics(sim, run_id=0)
        kpis.update(
            pudo_trips=pm['num_pudo_trips'],
            total_trips=pm['num_total_trips'],
            pudo_acceptance=pm['pudo_acceptance_rate'],
            avg_walk_pickup=pm['avg_walk_to_pickup'],
            avg_walk_dropoff=pm['avg_walk_from_dropoff'],
            total_savings=pm['total_savings_offered'],
        )
    return kpis


# ---------------------------------------------------------------------------
# Per-request outcomes
# ---------------------------------------------------------------------------

def extract_outcomes(sim, label):
    requests = sim.inData.requests
    trips = sim.runs[0]['trips']
    rows = []
    for req_id, req in requests.iterrows():
        pax_id = req.pax_id if 'pax_id' in req.index else req_id
        pax_trips = trips[trips.pax == pax_id]
        veh_ids = pax_trips['veh_id'].dropna().unique()
        completed = len(veh_ids) > 0

        row = dict(req_id=req_id, dist=float(req.dist), completed=completed)

        pickup_node = req.get('pudo_pickup_node', None)
        if pd.notna(pickup_node):
            row['used_pudo'] = int(pickup_node) != int(req.origin)
            row['walk_pickup'] = float(req.walk_to_pickup_dist)
            row['walk_dropoff'] = float(req.walk_from_dropoff_dist)
            row['savings'] = float(req.pudo_savings)
            fallback = req.get('pudo_d2d_fallback', False)
            row['d2d_fallback'] = bool(fallback) if pd.notna(fallback) else False
        else:
            row.update(used_pudo=False, walk_pickup=0.0, walk_dropoff=0.0,
                       savings=0.0, d2d_fallback=False)
        rows.append(row)

    df = pd.DataFrame(rows)
    df['strategy'] = label
    return df


# ---------------------------------------------------------------------------
# Decision log merging (for visualization)
# ---------------------------------------------------------------------------

def merge_decision_logs(sim):
    """Merge decision_log dicts from all batches into one."""
    if not hasattr(sim, 'plats') or not sim.plats:
        return None
    all_assignments = []
    all_best_edges = []
    all_offers = []
    meta = {}
    for batch in sim.plats[0].batch_history:
        log = batch.get('decision_log')
        if log is None:
            continue
        all_assignments.extend(log.get('assignments', []))
        all_best_edges.extend(log.get('best_per_edge', []))
        all_offers.extend(log.get('offers', []))
        if not meta:
            meta = log.get('meta', {})
    if not all_assignments and not all_best_edges:
        return None
    return {
        'meta': meta,
        'assignments': all_assignments,
        'best_per_edge': all_best_edges,
        'offers': all_offers,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(d, m, outcomes_milp):
    W = 60
    print('\n' + '=' * W)
    print('  D2D vs Batch MILP PUDO  (behavioral, same seed)')
    print('=' * W)

    print(f"\n  {'Metric':<25} {'D2D':>12} {'MILP':>12} {'Delta':>10}")
    print('  ' + '-' * 52)

    vkt_d = (m['vkt'] - d['vkt']) / d['vkt'] * 100 if d['vkt'] > 0 else 0
    print(f"  {'VKT (km)':<25} {d['vkt']:>12.1f} {m['vkt']:>12.1f} {vkt_d:>+9.1f}%")
    print(f"  {'PAX KM':<25} {d['pax_km']:>12.1f} {m['pax_km']:>12.1f}")
    print(f"  {'Cruise KM':<25} {d['cruise_km']:>12.1f} {m['cruise_km']:>12.1f}")
    print(f"  {'Rides Completed':<25} {d['n_rides']:>12.0f} {m['n_rides']:>12.0f}")
    print(f"  {'Avg Wait (s)':<25} {d['avg_wait']:>12.0f} {m['avg_wait']:>12.0f}")
    print(f"  {'Revenue ($)':<25} {d['revenue']:>12.0f} {m['revenue']:>12.0f}")

    if 'pudo_trips' in m:
        print(f"\n  {'PUDO Trips':<25} {'n/a':>12} {m['pudo_trips']:>12.0f}")
        print(f"  {'Total Trips':<25} {d['n_rides']:>12.0f} {m['total_trips']:>12.0f}")
        print(f"  {'PUDO Acceptance':<25} {'n/a':>12} {m['pudo_acceptance']:>11.0%}")
        print(f"  {'Avg Walk Pickup (m)':<25} {'n/a':>12} {m['avg_walk_pickup']:>12.1f}")
        print(f"  {'Avg Walk Dropoff (m)':<25} {'n/a':>12} {m['avg_walk_dropoff']:>12.1f}")
        print(f"  {'Total Savings ($)':<25} {'n/a':>12} {m['total_savings']:>12.2f}")

    n_pudo = int(outcomes_milp['used_pudo'].sum())
    n_fb = int(outcomes_milp['d2d_fallback'].sum())
    print(f"\n  {'PUDO Redirect Used':<25} {'n/a':>12} {n_pudo:>12}")
    print(f"  {'D2D Fallback Used':<25} {'n/a':>12} {n_fb:>12}")

    print('\n' + '=' * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('\n  Generating shared demand...')
    inData_base, params_base = generate_shared_demand()
    print(f'  {len(inData_base.passengers)} passengers, '
          f'{len(inData_base.vehicles)} vehicles, '
          f'{len(inData_base.requests)} requests')

    cf = getattr(params_base.speeds, 'ride_congestion', 1.0)
    print(f'  Congestion factor: {cf:.1f}x (~{50.0 / cf:.0f} km/h)')

    t0 = time.perf_counter()
    print('\n  Running D2D baseline...')
    sim_d2d = run_d2d(inData_base, params_base)
    t1 = time.perf_counter()
    print(f'  D2D done ({t1 - t0:.1f}s)')

    print('  Running Batch MILP PUDO...')
    sim_milp = run_milp_pudo(inData_base, params_base)
    t2 = time.perf_counter()
    print(f'  MILP done ({t2 - t1:.1f}s)')

    kpis_d2d = collect_kpis(sim_d2d, 'D2D')
    kpis_milp = collect_kpis(sim_milp, 'Batch MILP')
    outcomes_milp = extract_outcomes(sim_milp, 'MILP')

    print_summary(kpis_d2d, kpis_milp, outcomes_milp)

    # --- Route comparison visualizations (3 successful + 3 unsuccessful) ---
    print('\n  Generating route comparison plots...')
    merged_log = merge_decision_logs(sim_milp)
    if merged_log:
        # Build outcome labels from offers
        outcome_labels = {}
        for o in merged_log.get('offers', []):
            rid = int(o['req_id'])
            outcome = o.get('outcome', 'unknown')
            # Map raw outcomes to visualization outcome keys
            if outcome == 'accepted' and not o.get('d2d_fallback', False):
                outcome_labels[rid] = 'pudo_accepted'
            elif outcome == 'd2d_fallback_driver':
                outcome_labels[rid] = 'd2d_fallback_driver'
            elif outcome == 'd2d_fallback_rider':
                outcome_labels[rid] = 'd2d_fallback_rider'
            elif outcome in ('driver_declined',):
                outcome_labels[rid] = 'driver_s2_rejected'
            elif outcome == 'rider_s1_declined':
                outcome_labels[rid] = 'rider_s1_rejected'
            else:
                outcome_labels[rid] = outcome

        # Collect all req_ids that have route data
        all_req_ids = set()
        for a in merged_log.get('assignments', []):
            all_req_ids.add(int(a['req_id']))
        for e in merged_log.get('best_per_edge', []):
            all_req_ids.add(int(e['req_id']))

        # Split into successful PUDO and unsuccessful
        successful = [r for r in sorted(all_req_ids)
                      if outcome_labels.get(r) == 'pudo_accepted']
        unsuccessful = [r for r in sorted(all_req_ids)
                        if outcome_labels.get(r, '') != 'pudo_accepted']
        req_ids = successful[:3] + unsuccessful[:3]

        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'results', 'route_comparison')
        from MaaSSim.visualizations import plot_route_comparison
        generated = plot_route_comparison(merged_log, sim_milp, req_ids,
                                         out_dir,
                                         outcome_labels=outcome_labels)
        print(f'  {len(generated)} route comparison plots saved to {out_dir}')

        # --- Find accepted PUDO trip with highest vehicle savings (meters) ---
        G = sim_milp.inData.G
        best_saving, best_rid = 0, None
        for a in merged_log.get('assignments', []):
            rid = int(a['req_id'])
            if outcome_labels.get(rid) != 'pudo_accepted':
                continue
            try:
                d2d = (nx.shortest_path_length(G, int(a['veh_pos']), int(a['origin']), weight='length')
                       + nx.shortest_path_length(G, int(a['origin']), int(a['destination']), weight='length'))
                pudo = (nx.shortest_path_length(G, int(a['veh_pos']), int(a['pickup_node']), weight='length')
                        + nx.shortest_path_length(G, int(a['pickup_node']), int(a['dropoff_node']), weight='length'))
                saving = d2d - pudo
                if saving > best_saving:
                    best_saving, best_rid = saving, rid
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if best_rid is not None:
            print(f'\n  Best accepted PUDO vehicle savings: R{best_rid} saves {best_saving:.0f}m')
            if best_rid not in req_ids:
                extra = plot_route_comparison(merged_log, sim_milp,
                                             [best_rid], out_dir,
                                             outcome_labels=outcome_labels)
                generated.extend(extra)
                print(f'  Best-savings plot: {extra[0] if extra else "failed"}')
    else:
        print('  No decision logs available (decision_log_level may be off)')


if __name__ == '__main__':
    main()
