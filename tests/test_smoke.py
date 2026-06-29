"""End-to-end CPU smoke test: env + MACMAT policy + one PPO update.

Runs the real GridWorldRunner for a single short episode and asserts that the
training update produces the expected metrics with finite values. Exercises the
env step/reset, observation conversion, action selection, replay buffer, and the
PPO trainer together.
"""
import math

import numpy as np
import torch
import pytest

from conftest import build_runner


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().mean().item())
    return float(np.mean(np.asarray(value, dtype=float)))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("macmat_smoke")
    # 600 env steps / 300 max_steps / 2 threads == 1 episode == 1 PPO update.
    runner, envs = build_runner(run_dir, num_env_steps=600, n_rollout_threads=2)
    captured = {}
    orig_log_train = runner.log_train

    def cap(train_infos, total_num_steps):
        captured.update(train_infos)
        return orig_log_train(train_infos, total_num_steps)

    runner.log_train = cap
    try:
        runner.run()
    finally:
        envs.close()
    return runner, captured


def test_policy_built_on_cpu(trained):
    runner, _ = trained
    n_params = sum(p.numel() for p in runner.policy.transformer.parameters())
    assert n_params > 0
    assert str(next(runner.policy.transformer.parameters()).device) == "cpu"


def test_obs_and_action_spaces(trained):
    runner, _ = trained
    assert runner.num_agents == 4
    act_space = runner.envs.action_space
    assert len(act_space) == runner.num_agents
    # macmat action space == (2*action_size+1)^2 + 2
    assert act_space[0].n == (2 * 3 + 1) ** 2 + 2
    obs_space = runner.envs.observation_space[0].spaces
    assert "local_agent_view" in obs_space
    assert "agent_pose" in obs_space


def test_ppo_update_metrics_finite(trained):
    _, captured = trained
    assert captured, "no train_infos were logged"
    expected = {"value_loss", "policy_loss", "dist_entropy", "actor_grad_norm",
                "critic_grad_norm", "ratio"}
    assert expected.issubset(captured.keys()), sorted(captured.keys())
    for key, value in captured.items():
        scalar = _scalar(value)
        assert math.isfinite(scalar), f"{key} is not finite: {scalar}"
