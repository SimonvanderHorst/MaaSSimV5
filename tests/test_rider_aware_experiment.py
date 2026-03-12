"""
5-rep experiment: rider-aware MILP ranking ON vs OFF.
Same demand per rep, different seeds across reps.
Compares operational KPIs and behavioral utility distributions.

Usage:
    python tests/test_rider_aware_experiment.py
"""
import sys
import os
import copy
import time
import random
import io
from datetime import datetime
import numpy as np
import pandas as pd
from dotmap import DotMap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from MaaSSim.data_structures import structures
from MaaSSim.maassim import Simulator
from MaaSSim.simulators import _prep_rides
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df
from MaaSSim.performance import calculate_vkt, calculate_pudo_metrics

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_pudo_test.json')

SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Shared demand generation
# ---------------------------------------------------------------------------

def generate_shared_demand(seed):
    np.random.seed(seed)
    random.seed(seed)
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
# Strategy runner
# ---------------------------------------------------------------------------

def run_milp_pudo(inData_base, params_base, rider_aware, seed):
    inData = copy_indata(inData_base)
    params = copy.deepcopy(params_base)
    params.pudo.enabled = True
    params.pudo.behavioral.enabled = True
    params.pudo.d2d_fallback = True
    params.pudo.decision_log_level = 'summary'
    params.pudo.rider_aware_optimization = rider_aware
    params.pudo.rider_cost_weight = 1.0

    random.seed(seed)
    np.random.seed(seed)
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
    total_requests = len(sim.inData.requests)

    kpis = dict(label=label, vkt=vkt, pax_km=pax_km, cruise_km=cruise_km,
                n_rides=n_rides, total_requests=total_requests,
                avg_wait=avg_wait, revenue=revenue)

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
# Behavioral data extraction
# ---------------------------------------------------------------------------

def collect_behavioral_data(sim, label):
    # rider data from sim-level log
    rider_df = pd.DataFrame()
    if hasattr(sim, '_rider_behavioral_log') and sim._rider_behavioral_log:
        rider_df = pd.DataFrame(sim._rider_behavioral_log)
        rider_df['strategy'] = label

    # driver data from decision log offers
    driver_rows = []
    if hasattr(sim, 'plats') and sim.plats:
        for batch in sim.plats[0].batch_history:
            log = batch.get('decision_log')
            if log is None:
                continue
            for o in log.get('offers', []):
                beh = o.get('behavioral', {})
                if 'driver_utility' in beh:
                    driver_rows.append({
                        'veh_id': o.get('veh_id'),
                        'req_id': o.get('req_id'),
                        'driver_utility': beh['driver_utility'],
                        'driver_alpha': beh['driver_alpha'],
                        'driver_accepted': beh['driver_accepted'],
                        'outcome': o.get('outcome', ''),
                    })
    driver_df = pd.DataFrame(driver_rows) if driver_rows else pd.DataFrame()
    if len(driver_df) > 0:
        driver_df['strategy'] = label

    return rider_df, driver_df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

KPI_COLS = ['vkt', 'pax_km', 'cruise_km', 'n_rides', 'avg_wait', 'revenue',
            'pudo_trips', 'pudo_acceptance', 'avg_walk_pickup', 'avg_walk_dropoff',
            'total_savings']


def aggregate_kpis(all_kpis):
    df = pd.DataFrame(all_kpis)
    agg = df.groupby('label')[KPI_COLS].agg(['mean', 'std'])
    return df, agg


def aggregate_behavioral(rider_dfs, driver_dfs):
    rider_all = pd.concat(rider_dfs, ignore_index=True) if rider_dfs else pd.DataFrame()
    driver_all = pd.concat(driver_dfs, ignore_index=True) if driver_dfs else pd.DataFrame()

    summary = {}
    if len(rider_all) > 0:
        for strat, g in rider_all.groupby('strategy'):
            summary[strat + '_rider'] = {
                'utility_mean': g['rider_utility'].mean(),
                'utility_std': g['rider_utility'].std(),
                'utility_median': g['rider_utility'].median(),
                'alpha_mean': g['rider_alpha'].mean(),
                'acceptance_rate': g['rider_accepted'].mean(),
                'n_offers': len(g),
                'fallback_rate': g['d2d_fallback_triggered'].mean(),
                # utility components
                'u_fare_mean': g['u_fare_component'].mean(),
                'u_wait_mean': g['u_wait_component'].mean(),
                'u_walk_mean': g['u_walk_component'].mean(),
                'u_zero_mean': g['u_zero_component'].mean(),
            }

    if len(driver_all) > 0:
        for strat, g in driver_all.groupby('strategy'):
            summary[strat + '_driver'] = {
                'utility_mean': g['driver_utility'].mean(),
                'utility_std': g['driver_utility'].std(),
                'utility_median': g['driver_utility'].median(),
                'alpha_mean': g['driver_alpha'].mean(),
                'acceptance_rate': g['driver_accepted'].mean(),
                'n_offers': len(g),
            }

    return rider_all, driver_all, summary


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary(kpi_raw, kpi_agg, behavioral_summary):
    buf = io.StringIO()

    def _p(msg=''):
        print(msg)
        buf.write(msg + '\n')

    W = 68
    _p('\n' + '=' * W)
    _p('  Rider-Aware MILP Ranking: ON vs OFF  ({} reps)'.format(len(SEEDS)))
    _p('=' * W)

    # operational KPIs
    _p('\n  {:<22} {:>18} {:>18}'.format('Metric', 'ON (mean +/- std)', 'OFF (mean +/- std)'))
    _p('  ' + '-' * 60)

    on_label = 'rider_aware_ON'
    off_label = 'rider_aware_OFF'

    for col in KPI_COLS:
        on_m = kpi_agg.loc[on_label, (col, 'mean')]
        on_s = kpi_agg.loc[on_label, (col, 'std')]
        off_m = kpi_agg.loc[off_label, (col, 'mean')]
        off_s = kpi_agg.loc[off_label, (col, 'std')]

        if col in ('pudo_acceptance',):
            _p('  {:<22} {:>7.1%} +/- {:>4.1%}  {:>7.1%} +/- {:>4.1%}'.format(
                col, on_m, on_s, off_m, off_s))
        elif col in ('n_rides', 'pudo_trips', 'total_savings'):
            _p('  {:<22} {:>7.1f} +/- {:>5.1f}  {:>7.1f} +/- {:>5.1f}'.format(
                col, on_m, on_s, off_m, off_s))
        else:
            _p('  {:<22} {:>7.1f} +/- {:>5.1f}  {:>7.1f} +/- {:>5.1f}'.format(
                col, on_m, on_s, off_m, off_s))

    # paired deltas
    _p('\n  Paired deltas (ON - OFF, per rep):')
    _p('  ' + '-' * 60)
    for col in KPI_COLS:
        on_vals = kpi_raw[kpi_raw.label == on_label].sort_values('rep')[col].values
        off_vals = kpi_raw[kpi_raw.label == off_label].sort_values('rep')[col].values
        deltas = on_vals - off_vals
        _p('  {:<22} {:>+8.3f} +/- {:>6.3f}'.format(col, deltas.mean(), deltas.std()))

    # behavioral
    _p('\n  Behavioral Summary:')
    _p('  ' + '-' * 60)
    for key in sorted(behavioral_summary.keys()):
        _p('  [{}]'.format(key))
        for metric, val in behavioral_summary[key].items():
            if isinstance(val, float):
                _p('    {:<24} {:>10.4f}'.format(metric, val))
            else:
                _p('    {:<24} {:>10}'.format(metric, val))

    _p('\n' + '=' * W)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_kpis = []
    all_rider = []
    all_driver = []

    for rep, seed in enumerate(SEEDS):
        print('\n  === Rep {}/{} (seed={}) ==='.format(rep + 1, len(SEEDS), seed))

        inData_base, params_base = generate_shared_demand(seed)
        print('  {} passengers, {} vehicles, {} requests'.format(
            len(inData_base.passengers), len(inData_base.vehicles),
            len(inData_base.requests)))

        # rider_aware ON
        t0 = time.perf_counter()
        sim_on = run_milp_pudo(inData_base, params_base, rider_aware=True, seed=seed)
        t1 = time.perf_counter()
        kpis_on = collect_kpis(sim_on, 'rider_aware_ON')
        kpis_on['rep'] = rep
        kpis_on['seed'] = seed
        kpis_on['wall_time'] = t1 - t0
        rider_on, driver_on = collect_behavioral_data(sim_on, 'rider_aware_ON')
        rider_on['rep'] = rep
        driver_on['rep'] = rep if len(driver_on) > 0 else []
        print('  ON done ({:.1f}s) — {} rides, {:.0f} pudo trips'.format(
            t1 - t0, int(kpis_on['n_rides']), kpis_on['pudo_trips']))
        del sim_on

        # rider_aware OFF
        t2 = time.perf_counter()
        sim_off = run_milp_pudo(inData_base, params_base, rider_aware=False, seed=seed)
        t3 = time.perf_counter()
        kpis_off = collect_kpis(sim_off, 'rider_aware_OFF')
        kpis_off['rep'] = rep
        kpis_off['seed'] = seed
        kpis_off['wall_time'] = t3 - t2
        rider_off, driver_off = collect_behavioral_data(sim_off, 'rider_aware_OFF')
        rider_off['rep'] = rep
        driver_off['rep'] = rep if len(driver_off) > 0 else []
        print('  OFF done ({:.1f}s) — {} rides, {:.0f} pudo trips'.format(
            t3 - t2, int(kpis_off['n_rides']), kpis_off['pudo_trips']))
        del sim_off

        all_kpis.extend([kpis_on, kpis_off])
        if len(rider_on) > 0:
            all_rider.append(rider_on)
        if len(rider_off) > 0:
            all_rider.append(rider_off)
        if len(driver_on) > 0:
            all_driver.append(driver_on)
        if len(driver_off) > 0:
            all_driver.append(driver_off)

    # aggregate
    kpi_raw, kpi_agg = aggregate_kpis(all_kpis)
    rider_all, driver_all, behavioral_summary = aggregate_behavioral(all_rider, all_driver)

    # output
    summary_text = format_summary(kpi_raw, kpi_agg, behavioral_summary)

    # save
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results', 'rider_aware_experiment', timestamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)

    kpi_raw.to_csv(os.path.join(out_dir, 'kpis_per_rep.csv'), index=False)
    kpi_agg.to_csv(os.path.join(out_dir, 'kpis_aggregated.csv'))

    if len(rider_all) > 0:
        rider_all.to_csv(os.path.join(out_dir, 'rider_behavioral.csv'), index=False)
    if len(driver_all) > 0:
        driver_all.to_csv(os.path.join(out_dir, 'driver_behavioral.csv'), index=False)

    if behavioral_summary:
        pd.DataFrame(behavioral_summary).T.to_csv(
            os.path.join(out_dir, 'behavioral_summary.csv'))

    print('\n  All results saved to: {}'.format(out_dir))


if __name__ == '__main__':
    main()
