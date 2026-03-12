# MaaSSim package layout

## Core simulation

- `simulators.py`: main entry point. `simulate()` runs one sim from a config.
- `maassim.py`: the `Simulator` class, sets up SimPy processes, runs the event loop, collects output.
- `traveller.py`: `PassengerAgent`. Requests ride, waits for offer, walks/rides/walks.
- `driver.py`: `VehicleAgent`. Idles, gets dispatched, picks up, drives, drops off, repositions.
- `platform.py`: `PlatformAgent`. Queues requests, matches supply/demand, builds offers.
- `data_structures.py`: column definitions for the passenger/vehicle/platform DataFrames.
- `performance.py`: KPI calculation after a run (wait times, detours, vehicle-km, PUDO metrics).
- `animations.py`: animated map replays.
- `__main__.py`: so you can do `python -m MaaSSim`.

## Decision functions (`decisions/`)

The behavioral functions that control agent choices. You can swap any of them by passing your own function as a kwarg to `simulate()`.

- `matching.py`: `f_match()` for standard matching, `f_match_pudo()` for PUDO batch matching with MILP.
- `rider.py`: `f_out()` (leave system?), `f_mode()` / `f_pudo_rider_mode()` (accept/reject offer).
- `driver.py`: `f_driver_out()` (leave?), `f_repos()` (reposition), `f_pudo_driver_decline()` (reject dispatch?).
- `_helpers.py`: shared bits like `dummy_False`, `dummy_True`, `_compute_baseline_fare()`, `_safe_sigmoid()`.

## PUDO optimization

- `pudo_optimizer.py`: MILP that batch-assigns riders to pickup/dropoff nodes.
- `pudo_logger.py`: logs per-request PUDO decisions, preps data for visualization.

## DQN incentive learning (`dqn/`) - still very much WIP

Deep Q-network that learns how to split incentives between rider and driver after matching.

- `agent.py`: `DQNAgent`, `QNetwork` (PyTorch), `ReplayBuffer`.
- `policy.py`: `DQNIncentivePolicy`, hooks into sim to collect transitions.
- `state.py`: normalizes state features.
- `actions.py`: builds the discrete action table (incentive splits).
- `rewards.py`: reward computation from match outcomes.
- `config.json`: default hyperparameters.

## Utilities (`utils/`)

- `config.py`: `get_config()` loads a JSON config, `save_config()` writes one back.
- `network.py`: `load_G()` loads a road graph, `download_G()` grabs one from OSM.
- `demand.py`: `generate_demand()`, `generate_vehicles()`, `prep_supply_and_demand()`.
- `helpers.py`: small stuff: `rand_node()`, `empty_series()`, `initialize_df()`.
- `experiment.py`: `test_space()` and `collect_results()` for parameter sweeps.

## Visualizations (`visualizations/`)

- `core.py`: general plots (waiting times, ride stats, spatial heatmaps).
- `pudo.py`: PUDO-specific (walking distances, node usage, acceptance rates).

---

# Running your own experiment

```python
from MaaSSim.simulators import simulate
from MaaSSim.utils import get_config

params = get_config('tests/config_pudo_test.json')
sim = simulate(config=params)

# results
trips = sim.runs[0].trips       # raw trip log
pax_kpi = sim.res[0].pax_exp   # per-passenger KPIs
veh_kpi = sim.res[0].veh_exp   # per-vehicle KPIs
```

Tweak parameters in the config JSON before loading, or copy `tests/config_pudo_test.json` as a starting point (200 pax, 20 vehicles, 2h in Delft, PUDO on). Key fields: `nP`/`nV` (agent counts), `simTime` (hours), `city` (for network download), `pudo` (walking thresholds, behavioral params), `paths` (graph and skim locations). More defaults in `data/config/`.
