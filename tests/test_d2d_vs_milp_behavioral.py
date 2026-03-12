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
import io
from datetime import datetime
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
from MaaSSim.decisions import f_pudo_driver_decline

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_pudo_test.json')


def f_d2d_ratio_check(*args, **kwargs):
    """Stage 1 rider ratio check adapted for f_trav_mode calling convention.

    Rejects if (max(walk_time, dispatch_time) + matching_delay) / trip_time > 0.5
    """
    traveller = kwargs.get('traveller')
    sim = traveller.sim
    offer = list(traveller.offers.values())[0]
    pass_walk_time = sim.skims.walk[traveller.pax.pos][traveller.request.origin]
    veh_pickup_time = offer.get('wait_time', 0)
    pass_matching_time = sim.env.now - traveller.t_matching
    tt = traveller.request.ttrav
    tt_seconds = tt.total_seconds() if hasattr(tt, 'total_seconds') else float(tt)
    if tt_seconds <= 0:
        return False
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt_seconds > 0.5


# ---------------------------------------------------------------------------
# Shared demand generation
# ---------------------------------------------------------------------------

def generate_shared_demand():
    """Generate demand once for both strategies to share."""
    params = get_config(CONFIG_PATH)
    inData = structures.copy()
    inData = load_G(inData, params, stats=True)
    inData = generate_demand(inData, params, avg_speed=True)
    inData.vehicles = generate_vehicles(inData, params.simulation.nV)

    if len(inData.platforms) == 0:
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0, 'fare'] = params.platform.fare_per_km
        inData.platforms.loc[0, 'fare_per_min'] = params.platform.fare_per_min
        inData.platforms.loc[0, 'name'] = 'Platform'
        inData.platforms.loc[0, 'batch_time'] = params.platform.batch_time

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
    sim = Simulator(inData, params=params, logger_level='WARNING',
                    f_driver_decline=f_pudo_driver_decline, f_trav_mode=f_d2d_ratio_check)
    sim.make_and_run(run_id=0)
    sim.output()
    return sim


def run_milp_pudo(inData_base, params_base):
    inData = copy_indata(inData_base)
    params = copy.deepcopy(params_base)
    params.pudo.enabled = True
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


def print_milp_timing(sim):
    """Aggregate and print per-phase timing from batch_history."""
    batches = sim.plats[0].batch_history
    if not batches:
        print('  No MILP batches recorded.')
        return

    phases = [
        ('phase_a_feasibility_s', 'Feasibility'),
        ('phase_b_cost_matrix_s', 'Cost matrix'),
        ('phase_c_solve_s', 'LP solve'),
        ('phase_d_offers_s', 'Offer creation'),
    ]
    totals = {k: 0.0 for k, _ in phases}
    total_matches = 0
    total_vehs = 0
    total_reqs = 0

    for b in batches:
        t = b.get('timing', {})
        for k, _ in phases:
            totals[k] += t.get(k, 0.0)
        total_matches += t.get('n_matches', 0)
        total_vehs += t.get('n_vehicles', 0)
        total_reqs += t.get('n_requests', 0)

    grand = sum(totals.values())
    n = len(batches)

    print(f'\n  MILP Timing Breakdown ({n} batches, {total_matches} matches)')
    print('  ' + '-' * 44)
    for k, label in phases:
        pct = totals[k] / grand * 100 if grand > 0 else 0
        print(f'  Phase {k[6].upper()}  {label:<18} {totals[k]:>6.2f}s  ({pct:4.1f}%)')
    print('  ' + '-' * 44)
    print(f'  {"Total optimizer":<27} {grand:>6.2f}s')
    print(f'  Avg queue sizes: {total_vehs / n:.0f} vehs, {total_reqs / n:.0f} reqs')


def collect_kpis(sim, label):
    vkt = _scalar(calculate_vkt(sim, run_id=0))
    pax_km = _scalar(sim.res[0]['veh_exp']['PAX_KM'].sum())
    cruise_km = max(0.0, vkt - pax_km)
    n_rides = _scalar(sim.res[0]['veh_kpi'].loc['sum', 'nRIDES'])
    avg_wait = _scalar(sim.res[0]['pax_kpi'].loc['mean', 'WAIT'])
    revenue = _scalar(sim.res[0]['all_kpi']['total_revenue'])

    total_requests = len(sim.inData.requests)
    kpis = dict(label=label, vkt=vkt, pax_km=pax_km, cruise_km=cruise_km,
                n_rides=n_rides, total_requests=total_requests,
                avg_wait=avg_wait, revenue=revenue)

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

def format_summary(d, m, outcomes_milp, wall_d2d=None, wall_milp=None):
    """Build summary text and return it as a string (also prints to stdout)."""
    buf = io.StringIO()

    def _print(msg=''):
        print(msg)
        buf.write(msg + '\n')

    W = 60
    _print('\n' + '=' * W)
    _print('  D2D vs Batch MILP PUDO  (behavioral, same seed)')
    _print('=' * W)

    _print(f"\n  {'Metric':<25} {'D2D':>12} {'MILP':>12} {'Delta':>10}")
    _print('  ' + '-' * 52)

    vkt_d = (m['vkt'] - d['vkt']) / d['vkt'] * 100 if d['vkt'] > 0 else 0
    _print(f"  {'VKT (km)':<25} {d['vkt']:>12.1f} {m['vkt']:>12.1f} {vkt_d:>+9.1f}%")
    _print(f"  {'PAX KM':<25} {d['pax_km']:>12.1f} {m['pax_km']:>12.1f}")
    _print(f"  {'Cruise KM':<25} {d['cruise_km']:>12.1f} {m['cruise_km']:>12.1f}")
    _print(f"  {'Requests':<25} {d['total_requests']:>12.0f} {m['total_requests']:>12.0f}")
    _print(f"  {'Rides Completed':<25} {d['n_rides']:>12.0f} {m['n_rides']:>12.0f}")
    d_rate = d['n_rides'] / d['total_requests'] * 100 if d['total_requests'] > 0 else 0
    m_rate = m['n_rides'] / m['total_requests'] * 100 if m['total_requests'] > 0 else 0
    _print(f"  {'Completion Rate':<25} {d_rate:>11.0f}% {m_rate:>11.0f}%")
    _print(f"  {'Avg Wait (s)':<25} {d['avg_wait']:>12.0f} {m['avg_wait']:>12.0f}")
    _print(f"  {'Revenue ($)':<25} {d['revenue']:>12.0f} {m['revenue']:>12.0f}")

    if 'pudo_trips' in m:
        _print(f"\n  {'PUDO Trips':<25} {'n/a':>12} {m['pudo_trips']:>12.0f}")
        _print(f"  {'PUDO Acceptance':<25} {'n/a':>12} {m['pudo_acceptance']:>11.0%}")
        _print(f"  {'Avg Walk Pickup (m)':<25} {'n/a':>12} {m['avg_walk_pickup']:>12.1f}")
        _print(f"  {'Avg Walk Dropoff (m)':<25} {'n/a':>12} {m['avg_walk_dropoff']:>12.1f}")
        _print(f"  {'Total Savings ($)':<25} {'n/a':>12} {m['total_savings']:>12.2f}")

    n_pudo = int(outcomes_milp['used_pudo'].sum())
    n_fb = int(outcomes_milp['d2d_fallback'].sum())
    _print(f"\n  {'PUDO Redirect Used':<25} {'n/a':>12} {n_pudo:>12}")
    _print(f"  {'D2D Fallback Used':<25} {'n/a':>12} {n_fb:>12}")

    if wall_d2d is not None and wall_milp is not None:
        _print(f"\n  {'Wall Clock (s)':<25} {wall_d2d:>12.1f} {wall_milp:>12.1f}")

    _print('\n' + '=' * W)
    return buf.getvalue()


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
    print_milp_timing(sim_milp)

    kpis_d2d = collect_kpis(sim_d2d, 'D2D')
    kpis_milp = collect_kpis(sim_milp, 'Batch MILP')
    outcomes_milp = extract_outcomes(sim_milp, 'MILP')

    # --- Create timestamped output folder ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results', 'd2d_vs_milp_behavioral', timestamp)
    os.makedirs(out_dir, exist_ok=True)

    summary_text = format_summary(kpis_d2d, kpis_milp, outcomes_milp,
                                   wall_d2d=t1 - t0, wall_milp=t2 - t1)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)

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

    print(f'\n  All results saved to: {out_dir}')


if __name__ == '__main__':
    main()
