#!/usr/bin/env python
"""
Compare D2D baseline vs PUDO Gurobi batch matching.
Reports aggregate VKT, PUDO metrics, and solver timing.

Usage:
    python tests/run_pudo_comparison.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from MaaSSim.simulators import simulate
from MaaSSim.performance import calculate_vkt, calculate_pudo_metrics
from MaaSSim.utils import get_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_pudo_test.json')


def run_scenario(label, params):
    """Run a single simulation scenario and return stats dict."""
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")

    t0 = time.perf_counter()
    sim = simulate(params=params, logger_level='WARNING')
    wall_time = time.perf_counter() - t0

    stats = {'label': label, 'wall_time_s': wall_time}

    # VKT breakdown
    stats['vkt'] = calculate_vkt(sim, run_id=0)
    if 0 in sim.res and 'veh_exp' in sim.res[0]:
        veh_exp = sim.res[0]['veh_exp']
        stats['pax_km'] = veh_exp['PAX_KM'].sum()
        cruise_s = veh_exp['CRUISE'].sum()
        stats['cruise_km'] = cruise_s * sim.params.speeds.ride / 1000
    else:
        stats['pax_km'] = 0
        stats['cruise_km'] = 0

    # PUDO metrics (only meaningful when PUDO enabled)
    if params.pudo.enabled:
        pm = calculate_pudo_metrics(sim, run_id=0)
        stats['pudo_trips'] = pm.get('num_pudo_trips', 0)
        stats['total_trips'] = pm.get('num_total_trips', 0)
        stats['avg_walk_pickup_m'] = pm.get('avg_walk_to_pickup', 0)
        stats['avg_walk_dropoff_m'] = pm.get('avg_walk_from_dropoff', 0)

    # Print summary
    print(f"  VKT:       {stats['vkt']:.2f} km")
    print(f"  PAX_KM:    {stats['pax_km']:.2f} km")
    print(f"  CRUISE_KM: {stats['cruise_km']:.2f} km")
    if params.pudo.enabled:
        print(f"  PUDO trips: {stats.get('pudo_trips', 0)} / {stats.get('total_trips', 0)}")
        print(f"  Avg walk pickup:  {stats.get('avg_walk_pickup_m', 0):.1f} m")
        print(f"  Avg walk dropoff: {stats.get('avg_walk_dropoff_m', 0):.1f} m")
    print(f"  Wall time: {wall_time:.1f} s")

    return stats


def main():
    params = get_config(CONFIG_PATH)

    print("=" * 60)
    print("  PUDO Gurobi Batch vs D2D Comparison")
    print(f"  City: {params.city}  |  nP={params.nP}  nV={params.nV}  simTime={params.simTime}h")
    print("=" * 60)

    # --- D2D baseline ---
    params.pudo.enabled = False
    d2d = run_scenario("D2D Baseline", params)

    # --- PUDO Gurobi batch ---
    params.pudo.enabled = True
    params.pudo.greedy_first = False
    params.pudo.solver = 'gurobi_lp'
    pudo = run_scenario("PUDO Gurobi Batch (LP relax)", params)

    # --- Comparison ---
    vkt_delta = pudo['vkt'] - d2d['vkt']
    vkt_pct = (vkt_delta / d2d['vkt'] * 100) if d2d['vkt'] > 0 else 0

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'D2D':>10} {'PUDO':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    print(f"  {'VKT (km)':<25} {d2d['vkt']:>10.2f} {pudo['vkt']:>10.2f} {vkt_delta:>+10.2f}")
    print(f"  {'PAX_KM':<25} {d2d['pax_km']:>10.2f} {pudo['pax_km']:>10.2f}")
    print(f"  {'CRUISE_KM':<25} {d2d['cruise_km']:>10.2f} {pudo['cruise_km']:>10.2f}")
    print(f"  {'Wall time (s)':<25} {d2d['wall_time_s']:>10.1f} {pudo['wall_time_s']:>10.1f}")
    print(f"\n  VKT change: {vkt_pct:+.1f}%")


if __name__ == '__main__':
    main()
