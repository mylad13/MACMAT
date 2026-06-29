#!/usr/bin/env python
"""Capture a deterministic, bit-for-bit training digest for refactor verification.

Loads the real training config, applies the same overrides as the Tier-4 refactor
baseline (CPU, fixed seed, single BLAS thread, supported 2-thread subproc path),
runs training, and records every logged train/env metric to JSON. Run it before
a refactor and again after; the JSON files must be byte-identical.

    # capture a baseline before refactoring
    python tools/bitwise_baseline.py /tmp/before.json
    # ... refactor ...
    python tools/bitwise_baseline.py /tmp/after.json
    diff /tmp/before.json /tmp/after.json   # must be empty

Cover more branches by overriding config fields, e.g.:
    python tools/bitwise_baseline.py /tmp/cfc.json --set rnn_type=CfC
    python tools/bitwise_baseline.py /tmp/noattn.json --set use_graph_attention=false use_graph_attention_eval=false

Note: the single-process (n_rollout_threads=1) vec-env path has latent shape bugs
and is unused here; this harness uses the supported subproc path with 2 threads.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIG = REPO / "hetmarl" / "scripts" / "config" / "hydra_train_config.yaml"


def build(num_env_steps, seed, n_threads, overrides):
    cfg = OmegaConf.load(CONFIG)
    a = cfg.all_args
    a.cuda = False
    a.use_wandb = False
    a.n_training_threads = 1
    a.n_rollout_threads = n_threads
    a.num_env_steps = num_env_steps
    a.seed = seed
    for tok in overrides:
        k, _, v = tok.partition("=")
        OmegaConf.update(a, k, OmegaConf.create(f"v: {v}").v)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    a = build(args.steps, args.seed, args.threads, args.overrides)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.episode_length = a.max_steps if a.asynch else a.max_steps // a.local_step_num

    from hetmarl.envs.gridworld.GridWorld_Env import GridWorldEnv
    from hetmarl.envs.env_wrappers import InfoSubprocVecEnv

    def env_fn(rank):
        def init():
            env = GridWorldEnv(a)
            env.seed(a.seed + rank * 1000)
            return env
        return init

    envs = InfoSubprocVecEnv([env_fn(i) for i in range(a.n_rollout_threads)])
    run_dir = REPO / ".bitwise_runs" / f"seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from hetmarl.runner.shared.gridworld_runner import GridWorldRunner
    runner = GridWorldRunner({"all_args": a, "envs": envs, "eval_envs": None,
                              "num_agents": a.num_agents, "device": torch.device("cpu"),
                              "run_dir": run_dir})

    rec = {"train": [], "env": []}

    def scalar(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(np.mean(v))
            except Exception:
                return str(v)

    orig_t, orig_e = runner.log_train, runner.log_env

    def cap_t(info, step):
        rec["train"].append({"step": int(step), **{k: scalar(v) for k, v in sorted(info.items())}})
        return orig_t(info, step)

    def cap_e(info, step):
        rec["env"].append({"step": int(step),
                           **{k: scalar(np.mean(v)) for k, v in sorted(info.items()) if hasattr(v, "__len__") and len(v) > 0}})
        return orig_e(info, step)

    runner.log_train, runner.log_env = cap_t, cap_e
    runner.run()
    envs.close()

    Path(args.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(f"WROTE {args.out}: {len(rec['train'])} train / {len(rec['env'])} env records")


if __name__ == "__main__":
    main()
