# Tier 4 refactor — status

Progress against [`TIER4_REFACTOR_PLAN.md`](TIER4_REFACTOR_PLAN.md). Every change
below was made in a small, revertable commit and verified to preserve behaviour
**bit-for-bit** (see "Verification" below). Guardrail #1 (no numerical change) was
treated as hard: anything that could not be proven equivalent was not changed.

## Prerequisite fixes (the repo did not run as-is)

Under the pinned deps the training/eval entry points could not even start, so
these were needed before any baseline could be captured:

- **gym ≥0.26 env-checker** rejected the env's list-valued multi-agent
  `action_space`. `gym.make(..., disable_env_checker=True)` in `GridWorld_Env`.
- **Dead MiniGrid scripts** (`benchmark.py`, `manual_control.py`, `run_tests.py`)
  imported a nonexistent top-level `gym_minigrid` package — removed.
- Stale tracked `.pyc` files untracked; build/cache artifacts gitignored.

## Item status

| # | Item | Status | Notes |
|---|------|--------|-------|
| 8 | Tests + CI safety net | ✅ done | `tests/` (import + smoke + semantics + ncps), `.github/workflows/ci.yml` (compileall + pytest on CPU torch). |
| 10 | SyntaxWarnings + LICENSE | ✅ done | `roomgrid.py` `is`→`==`; Apache-2.0 `LICENSE`/`NOTICE`; README license section. |
| 4 | Centralize entity codes | ✅ done | `semantics.CellCode` IntEnum replaces all cell-code literals in `searchandrescue.py` + the runner channel builder. Agent-*class* codes (0/1/2/10/210) are a separate namespace, left as-is. |
| 7 | Strip commented scaffolding | ✅ done | NCP wiring presets + A/B/C memory-fusion + NoClass fusion variants moved to `docs/architecture_notes.md`; macmat_transformer 866→806 lines. |
| 9 | Lazy `ncps` | ✅ done | `torchncp` exposes `LTCCell`/`CfC`/`LTC` lazily (PEP 562); a default MACMAT run needs no external `ncps`. |
| 6 | Encapsulate RNN-type handling | ✅ done (core) | The `rnn_type == 'LSTM' or 'Mixed' in rnn_type` test (46 sites: runner 32, buffer 11, transformer 3) → a single `_rnn_uses_tuple_state` property per class. |
| 5 | Algorithm dispatch | ✅ done | `algorithms/registry.py` (`AlgorithmSpec` + `is_frontier_algorithm`); `base_runner`/`transformer_policy` resolve classes through it. Env obs/action-space shape logic left as env config. |
| 2 | Decompose `compute_global_goal` / `eval_compute_global_goal` | 🟡 partial | Extracted `_goal_to_macro_action`, `_rnn_states_to_numpy` (shared by the training and eval action-selection closures, removing the duplication). The full sync/async sub-method split is **deferred** (see below). |
| 1 | Unify `eval()`/`render()` | 🟡 partial | Shared the duplicated `AsynchControl` setup via `_make_asynch_control`. The full single-`_run_eval_rollouts` body is **deferred** — eval/render are far less similar than the plan assumed (parallel `eval_*` vs plain state families, and eval has a large run-name/logging preamble render lacks). |
| 3 | Split env `step()`/`reset()` | 🟡 partial | Extracted `reset()`'s observation/tracking-state init into `_init_observation_state()`. The remaining `step()`/`reset()` phase split is **deferred**. |

## Verification

`tools/bitwise_baseline.py` captures a deterministic JSON digest of every logged
train/env metric. Six configs were locked as baselines and re-checked after each
behaviour-touching commit; all stayed **byte-identical**:

| config | branch it covers |
|--------|------------------|
| `lstm` (default) | LSTM = `(h, c)` tuple recurrent state |
| `--set rnn_type=CfC` | flat single-array recurrent state (liquid cell) |
| `--set use_graph_attention=false use_graph_attention_eval=false` | the whole no-attention path in `compute_global_goal` |
| `--set use_classbased_action=false` | NoClass ablation (class embedding) |
| `--set use_partial_comm=true use_full_comm=false` | partial-comm activation logic |
| `--set dynamic_targets=true dynamic_obstacles=true` | moving targets/obstacles in `step()` |

A separate fixed-seed **eval** baseline (the `eval_data_dict` minus wall-clock
timing fields) covers `eval()` / `eval_compute_global_goal` / `_make_asynch_control`.
`python -m pytest` and `python -m compileall hetmarl` pass.

## Why the deep decompositions were deferred

The remaining splits (items 1/2/3 "full") restructure 300–630-line hot-path
functions whose branches the available baselines cannot all reach:

- `spawn_hazards`, `use_localization=false`, the `GRU` rnn_type, and `amat` each
  hit **pre-existing bugs** and cannot run, so those branches are unverifiable.
- `ft_*` frontier methods and the `NCP`/`MixedNCP`/`MixedCfC` recurrent types are
  untested here.

Per guardrail #1, restructuring control flow through branches that cannot be
verified risks silently changing a paper's results. The safe, mechanical
extractions that *are* covered were done; the deeper splits should follow only
once those configs are fixed/covered. `tools/bitwise_baseline.py` is the harness
to do that — capture before, refactor, diff after.
