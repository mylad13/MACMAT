"""Shared test fixtures/helpers for the MACMAT smoke + import tests.

Everything here runs on CPU with tiny settings so the suite is CI-friendly.
The single-process (``n_rollout_threads == 1``) vectorised-env path has latent
shape bugs and was never exercised by the original training runs, so the helpers
default to the supported multi-process (subproc) path with 2 threads.
"""
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "hetmarl" / "scripts" / "config" / "hydra_train_config.yaml"


def make_all_args(**overrides):
    """Load the real training config and apply CPU/test overrides."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(CONFIG)
    all_args = cfg.all_args
    all_args.cuda = False
    all_args.use_wandb = False
    all_args.n_training_threads = 1
    for key, value in overrides.items():
        OmegaConf.update(all_args, key, value)
    return all_args


def build_envs(all_args, base_seed=0):
    from hetmarl.envs.gridworld.GridWorld_Env import GridWorldEnv
    from hetmarl.envs.env_wrappers import InfoDummyVecEnv, InfoSubprocVecEnv

    def get_env_fn(rank):
        def init_env():
            env = GridWorldEnv(all_args)
            env.seed(base_seed + rank * 1000)
            return env

        return init_env

    n = all_args.n_rollout_threads
    if n == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    return InfoSubprocVecEnv([get_env_fn(i) for i in range(n)])


def build_runner(run_dir, **overrides):
    """Construct a GridWorldRunner exactly as train_gridworld.main would, on CPU."""
    overrides.setdefault("n_rollout_threads", 2)
    overrides.setdefault("seed", 0)
    all_args = make_all_args(**overrides)

    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.set_num_threads(all_args.n_training_threads)

    if all_args.asynch:
        all_args.episode_length = all_args.max_steps
    else:
        all_args.episode_length = all_args.max_steps // all_args.local_step_num

    envs = build_envs(all_args, all_args.seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": torch.device("cpu"),
        "run_dir": run_dir,
    }
    from hetmarl.runner.shared.gridworld_runner import GridWorldRunner

    return GridWorldRunner(config), envs
