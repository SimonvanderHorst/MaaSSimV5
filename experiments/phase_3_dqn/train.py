"""
DQN training loop for PUDO incentive allocation.

Each episode = one simulation day with fresh demand.
Transitions are collected during the sim, then used to fill the replay buffer.

Usage:
    python experiments/phase_3_dqn/train.py
"""
import sys
import os
import copy
import random
import time
from datetime import datetime
import psutil

import json
import shutil
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from MaaSSim.maassim import Simulator
from MaaSSim.utils import generate_shared_demand, copy_indata

from MaaSSim.dqn.agent import DQNAgent
from MaaSSim.dqn.policy import DQNIncentivePolicy
from MaaSSim.dqn import DQN_DEFAULTS

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'tests', 'final_calibration', 'configs',
                           'pudo_milp_15s_2x_rijswijk.json')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_epsilon(global_step, cfg):
    frac = min(global_step / max(cfg['eps_decay_steps'], 1), 1.0)
    return cfg['eps_start'] + frac * (cfg['eps_end'] - cfg['eps_start'])


def compute_beta(global_step, cfg):
    beta_start = cfg.get('per_beta_start', 0.4)
    frac = min(global_step / max(cfg['eps_decay_steps'], 1), 1.0)
    return beta_start + frac * (1.0 - beta_start)


def train(n_episodes=500, n_demand_seeds=5, base_seed=42, config_overrides=None,
          resume_dir=None, model_filename='final_model.pt', revenue_model=None,
          out_dir=None):

    start_episode = 0

    if resume_dir:
        # restore config and meta from previous run
        with open(os.path.join(resume_dir, 'training_meta.json')) as f:
            meta = json.load(f)
        n_demand_seeds = meta['n_demand_seeds']
        base_seed = meta['base_seed']

        with open(os.path.join(resume_dir, 'dqn_config.json')) as f:
            cfg = json.load(f)

        agent = DQNAgent(cfg)
        agent.load(os.path.join(resume_dir, model_filename))
        policy = DQNIncentivePolicy(agent)

        # figure out where we left off (checkpoint's global_step is ground truth)
        log_df = pd.read_csv(os.path.join(resume_dir, 'training_log.csv'))
        # trim rows beyond checkpoint (crash may have logged episodes the checkpoint missed)
        log_df = log_df[log_df['global_step'] <= agent.global_step]
        start_episode = int(log_df['episode'].max()) + 1
        best_reward = float(log_df['episode_reward'].max())

        run_dir = resume_dir
        # rewrite trimmed log (drops rows from crashed episodes beyond checkpoint)
        log_df.to_csv(os.path.join(run_dir, 'training_log.csv'), index=False)
        log_header_written = True
        match_header_written = True
        match_log_path = os.path.join(run_dir, 'match_log.csv')

        print(f"resuming from episode {start_episode}, global_step {agent.global_step}, "
              f"best_reward {best_reward:.1f}, buffer {len(agent.buffer)}")
    else:
        cfg = {**DQN_DEFAULTS, **(config_overrides or {})}
        agent = DQNAgent(cfg)
        policy = DQNIncentivePolicy(agent)

        best_reward = -float('inf')
        log_header_written = False
        match_header_written = False

        # output dir
        run_dir = out_dir or os.path.join('results', 'dqn', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)

        # save configs for reproducibility
        with open(os.path.join(run_dir, 'dqn_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        shutil.copy(CONFIG_PATH, os.path.join(run_dir, 'sim_config.json'))
        with open(os.path.join(run_dir, 'training_meta.json'), 'w') as f:
            json.dump({
                'n_episodes': n_episodes,
                'n_demand_seeds': n_demand_seeds,
                'base_seed': base_seed,
                'config_overrides': config_overrides,
                'config_path': CONFIG_PATH,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)

        match_log_path = os.path.join(run_dir, 'match_log.csv')

    # pre-generate demand pool (load network once, share across seeds)
    print(f"generating {n_demand_seeds} demand instances...")
    demand_pool = []
    shared_net = None
    for i in range(n_demand_seeds):
        t0 = time.perf_counter()
        inData, params = generate_shared_demand(CONFIG_PATH, seed=base_seed + i,
                                                shared_network=shared_net)
        if shared_net is None:
            shared_net = inData  # first load becomes the shared network
        demand_pool.append((inData, params))
        print(f"  seed {base_seed + i}: {time.perf_counter() - t0:.1f}s")

    # resolve revenue model from sim platform registry, wire onto policy
    if revenue_model is not None:
        for _, _params in demand_pool:
            _params.platform.revenue_model = revenue_model
    _plat = demand_pool[0][1].platform
    _rm_name = _plat.get('revenue_model', 'commission')
    if _rm_name not in _plat.revenue_models:
        raise KeyError(f"unknown revenue_model {_rm_name!r}; known: {list(_plat.revenue_models)}")
    _rm = _plat.revenue_models[_rm_name]
    policy.reward_type = _rm['reward_type']
    policy.booking_fee = _rm['booking_fee']
    print(f"revenue_model: {_rm_name} "
          f"(reward_type={policy.reward_type}, booking_fee={policy.booking_fee})")

    for ep in range(start_episode, start_episode + n_episodes):
        eps = compute_epsilon(agent.global_step, cfg)
        policy.reset(eps)

        # pick demand (cycle through pool)
        inData_base, params_base = demand_pool[ep % n_demand_seeds]
        inData = copy_indata(inData_base)
        params = copy.deepcopy(params_base)
        params.pudo.enabled = True
        params.pudo.behavioral.enabled = True
        params.pudo.d2d_fallback = True

        # seed for stochastic behavioral models
        ep_seed = base_seed + 1000 + ep
        random.seed(ep_seed)
        np.random.seed(ep_seed)

        # run simulation with DQN policy — instrument sub-phases
        t0 = time.perf_counter()
        sim = Simulator(inData, params=params, logger_level='CRITICAL')
        sim._dqn_policy = policy
        sim._lightweight_history = True  # skip batch_history match dicts
        t_init = time.perf_counter()
        sim.myinit()
        t_generate = time.perf_counter()
        sim.generate()
        t_simulate = time.perf_counter()
        sim.simulate(run_id=0)
        t_done = time.perf_counter()
        ep_time = t_done - t0
        sim_init_s = t_init - t0
        sim_myinit_s = t_generate - t_init
        sim_generate_s = t_simulate - t_generate
        sim_simulate_s = t_done - t_simulate
        sim_envrun_s = sim.sim_end - sim.sim_start
        sim_makeres_s = sim.make_res_time
        sim_assertme_s = sim.assert_me_time

        # sum PUDO phase timings across all batches
        phase_keys = ['phase_a_feasibility_s', 'phase_b_cost_matrix_s',
                      'phase_c_solve_s', 'phase_d_offers_s']
        phase_sums = {k: 0.0 for k in phase_keys}
        for batch in sim.plats[0].batch_history:
            for k in phase_keys:
                phase_sums[k] += batch['timing'].get(k, 0.0)

        # harvest transitions then free sim memory
        transitions = policy.get_episode_transitions()
        agent.store_transitions(transitions)
        if policy._mc_min != float('inf'):
            agent.observe_mc_range(policy._mc_min, policy._mc_max)
        sim.cleanup()
        del sim
        gc.collect()

        # freeze normalizer after warmup, re-normalize buffered transitions
        loss = None
        if agent.global_step >= cfg['warmup_steps'] and not agent.normalizer.frozen:
            agent.normalizer.freeze()
            if hasattr(agent.buffer, 'iter_data'):
                for data_idx, (s, a, r, ns, d) in agent.buffer.iter_data():
                    agent.buffer.replace_data(data_idx, (
                        agent.normalizer.normalize(s), a, r,
                        agent.normalizer.normalize(ns), d))
            else:
                for i, (s, a, r, ns, d) in enumerate(agent.buffer.buf):
                    agent.buffer.buf[i] = (
                        agent.normalizer.normalize(s), a, r,
                        agent.normalizer.normalize(ns), d)
            print(f"  normalizer frozen (n={agent.normalizer.n})")

        # freeze c51 support after enough episodes to observe real return range
        support_freeze = cfg.get('support_freeze_steps', cfg['warmup_steps'])
        if agent.global_step >= support_freeze and not agent.reward_tracker.frozen:
            agent.freeze_support()
            rt = agent.reward_tracker
            print(f"  c51 support frozen: [{agent.v_min}, {agent.v_max}] (observed [{rt.min:.3f}, {rt.max:.3f}])")

        # refreeze with trained-policy MC returns
        refreeze = cfg.get('support_refreeze_steps')
        if refreeze and agent.global_step >= refreeze and not getattr(agent, '_refrozen', False):
            old = (agent.v_min, agent.v_max)
            agent.freeze_support()
            agent._refrozen = True
            print(f"  c51 support re-frozen: [{agent.v_min}, {agent.v_max}] (was [{old[0]}, {old[1]}])")

        # train from buffer
        if agent.global_step >= cfg['warmup_steps']:
            n_train = len(transitions)
            beta = compute_beta(agent.global_step, cfg) if agent.use_per else None
            for _ in range(n_train):
                loss = agent.train_step(beta=beta)
                agent.sync_target()

        # per-match log (append to file, don't keep in memory)
        ep_match_rows = policy.get_match_rows(ep)
        if ep_match_rows:
            df_matches = pd.DataFrame(ep_match_rows)
            df_matches.to_csv(match_log_path, index=False, float_format='%.4f',
                              mode='a', header=not match_header_written)
            match_header_written = True

        # metrics
        metrics = policy.get_episode_metrics()
        ep_reward = metrics['episode_reward']

        log_row = {
            'episode': ep,
            'epsilon': eps,
            'global_step': agent.global_step,
            'num_matches': metrics['num_matches'],
            'episode_reward': ep_reward,
            'acceptance_rate': metrics['acceptance_rate'],
            'avg_pi_r': metrics['avg_pi_r'],
            'avg_pi_d': metrics['avg_pi_d'],
            'vkt_savings_m': metrics['vkt_savings_m'],
            'margin_capture': metrics['margin_capture'],
            'avg_q_chosen': metrics['avg_q_chosen'],
            'avg_q_max': metrics['avg_q_max'],
            'avg_q_mean': metrics['avg_q_mean'],
            'avg_q_std_chosen': metrics['avg_q_std_chosen'],
            'loss': loss,
            'buffer_size': len(agent.buffer),
            'ep_time': ep_time,
            'sim_init_s': sim_init_s,
            'sim_myinit_s': sim_myinit_s,
            'sim_generate_s': sim_generate_s,
            'sim_simulate_s': sim_simulate_s,
            'phase_a_s': phase_sums['phase_a_feasibility_s'],
            'phase_b_s': phase_sums['phase_b_cost_matrix_s'],
            'phase_c_s': phase_sums['phase_c_solve_s'],
            'phase_d_s': phase_sums['phase_d_offers_s'],
            'sim_envrun_s': sim_envrun_s,
            'sim_makeres_s': sim_makeres_s,
            'sim_assertme_s': sim_assertme_s,
            'optimizer_total_s': sum(v for k, v in phase_sums.items() if k in phase_keys),
            'rss_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        }
        # save best
        if ep_reward > best_reward and agent.global_step >= cfg['warmup_steps']:
            best_reward = ep_reward
            agent.save(os.path.join(run_dir, 'best_model.pt'), include_buffer=False)

        print(f"ep {ep:4d} | eps {eps:.3f} | matches {metrics['num_matches']:3d} | "
              f"reward {ep_reward:8.1f} | acc {metrics['acceptance_rate']:.2f} | "
              f"loss {loss if loss else 'n/a':>8} | buf {len(agent.buffer)} | {ep_time:.1f}s")

        # append training log + checkpoint save
        log_path = os.path.join(run_dir, 'training_log.csv')
        pd.DataFrame([log_row]).to_csv(log_path, index=False, mode='a',
                                        header=not log_header_written)
        log_header_written = True
        agent.save(os.path.join(run_dir, 'final_model.pt'))
    print(f"\ndone. results saved to {run_dir}")

    return agent, run_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--resume', type=str, default=None, help='path to run dir to resume')
    parser.add_argument('--model', type=str, default='final_model.pt', help='checkpoint filename to load on resume')
    parser.add_argument('--revenue-model', type=str, default=None,
                        help='override sim config platform.revenue_model selector')
    args = parser.parse_args()
    train(n_episodes=args.episodes, resume_dir=args.resume, model_filename=args.model,
          revenue_model=args.revenue_model)
