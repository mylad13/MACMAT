# Tier 4 — Core Refactoring Plan (hand-off)

This document is a hand-off plan for the **structural refactors** of the MACMAT
codebase. Tiers 1–3 (docs/packaging, dead-code removal, config/device portability) are
already applied. Tier 4 is the highest-value-but-highest-risk work: decomposing very
large functions, de-duplicating the eval/render paths, and replacing scattered string/
magic-number branching. None of it should change numerical behavior.

## Guardrails (read first)

1. **Behavior must be bit-for-bit preservable.** This is a research repo tied to a paper.
   Before/after every change, run a fixed-seed smoke comparison and confirm identical
   trajectories, losses, and metrics.
   - Suggested baseline: `cd hetmarl/scripts && python train/train_gridworld.py
     all_args.cuda=false all_args.n_rollout_threads=1 all_args.num_env_steps=6000
     all_args.use_wandb=false all_args.seed=0` and capture the first few logged
     loss/value numbers. Repeat after each refactor; they must match.
2. **Refactor in small, independently-revertable commits.** One function per commit.
3. **No logic changes inside a move.** Extract verbatim first (pure cut/paste into a
   helper), verify identical behavior, *then* simplify in a separate commit.
4. **Keep public entry points stable:** `GridWorldRunner.run/eval/render`,
   `TransformerPolicy.{get_actions,evaluate_actions,act,save,restore}`, and the env's
   `step/reset/get_short_term_action/ft_get_short_term_goals`.
5. Add a `tests/` smoke test (see item 8) *before* the big refactors so you have a safety
   net.

## Inventory of oversized functions (the core targets)

| File | Function | Approx. size | Notes |
|------|----------|--------------|-------|
| `hetmarl/runner/shared/gridworld_runner.py` | `compute_global_goal()` | ~571 lines | sync/async × RNN-type × graph-attention branches |
| `hetmarl/runner/shared/gridworld_runner.py` | `eval_compute_global_goal()` | ~625 lines | near-duplicate of `compute_global_goal` for eval + FT |
| `hetmarl/runner/shared/gridworld_runner.py` | `eval()` | ~445 lines | ~90% shared with `render()` |
| `hetmarl/runner/shared/gridworld_runner.py` | `render()` | ~336 lines | adds GIF capture on top of `eval()` |
| `hetmarl/runner/shared/gridworld_runner.py` | `_convert()` / obs conversion | ~350 lines | separate MACMAT vs AMAT paths |
| `hetmarl/runner/shared/gridworld_runner.py` | `run()` | ~240 lines | training loop + async control + insert |
| `hetmarl/envs/.../searchandrescue.py` | `reset()` | ~518 lines | grid gen + agent setup + obs build |
| `hetmarl/envs/.../searchandrescue.py` | `step()` | ~631 lines | dynamics + exploration + comm + rewards |
| `hetmarl/envs/.../searchandrescue.py` | `get_short_term_action()` | ~285 lines | macmat/amat/ft A* decoding mixed |

## Work items (prioritized)

### 1. Unify `eval()` and `render()` (highest value, contained)
They differ almost only in GIF capture and which env handle is used.
- Extract a single `_run_eval_rollouts(envs, *, capture_frames: bool, num_episodes)` that
  returns the metrics dict; `render()` = `_run_eval_rollouts(..., capture_frames=True)`
  plus the `imageio.mimsave` block, `eval()` = `capture_frames=False`.
- Fold the duplicated `AsynchControl` setup (currently copy-pasted in `run`/`eval`/
  `render`) into one `_make_asynch_control(num_envs)` helper.
- **Verification:** identical eval metrics for a fixed seed/checkpoint; GIFs still written.

### 2. Decompose `compute_global_goal()` / `eval_compute_global_goal()`
- Extract the RNN-state (un)packing (LSTM/Mixed tuple vs flat array) into
  `_pack_rnn_states()` / `_unpack_rnn_states()` — this branch repeats 30+ times across the
  file and is the single biggest readability win.
- Split into `_policy_step_sync(...)` and `_policy_step_async(...)`; have eval reuse the
  same helpers with `deterministic=True` instead of maintaining a parallel copy.
- Consider merging `eval_compute_global_goal` into `compute_global_goal` parameterized by
  `deterministic` and `use_ft`.
- **Risk:** high (touches the action-selection hot path). Extract verbatim first.

### 3. Split env `step()` and `reset()`
- `step()` → `_advance_dynamics()` (targets/obstacles), `_update_exploration_and_comm()`,
  `_compute_rewards()`, `_build_observations()`.
- `reset()` → `_generate_grid()`, `_place_agents_targets_obstacles()`,
  `_init_observation_state()`.
- These methods mutate a lot of `self.*` state; document the state each helper reads/writes
  in its docstring. Keep helpers private and order-dependent (don't reorder calls).

### 4. Centralize environment semantic codes
Magic numbers for cell/entity types are scattered across the env and runner
(`10/11/12` agents, `20` self, `40/160` obstacles, `60`, `80` door, `180` target, `200`
hazard).
- Add `hetmarl/envs/gridworld/semantics.py` with an `IntEnum` (e.g. `CellCode`).
- Replace literal comparisons in `searchandrescue.py`, `GridWorld_Env.py`, and
  `gridworld_runner.py` (the channel-builder around lines 539–560).
- **Verification:** values must be unchanged; this is a pure rename.

### 5. Replace `algorithm_name` string matching with a dispatch table
`algorithm_name` (`macmat`/`amat`/`ft_*`) is string-matched in `GridWorld_Env.py`,
`searchandrescue.py`, `transformer_policy.py`, and the runner. Introduce a small registry/
enum mapping name → (policy class, obs-space builder, action-decoder) to remove the
spread-out `if name == ...` chains.

### 6. Encapsulate RNN-type handling
The `if self.rnn_type == 'LSTM' or 'Mixed' in self.rnn_type` pattern appears in the
runner, policy, and network. Move all state-shape logic behind the RNN wrapper so callers
treat the recurrent state opaquely (a single object with `.detach()/.to()/.zeros_like()`).

### 7. Strip remaining commented research scaffolding
Move design-rationale comments (e.g. the NCP wiring presets for different `n_hidden_r` in
`macmat_transformer.py`, and the "Option A/B/C" memory-fusion alternatives) into a short
`docs/architecture_notes.md`, then delete the commented code blocks.

### 8. Add a minimal test/CI safety net
- `tests/test_smoke.py`: build the env, build the MACMAT policy on CPU, run a handful of
  env steps + one PPO update; assert shapes and finite losses.
- `tests/test_import.py`: import every live module (guards against future dead-import
  regressions like the one removed in Tier 2).
- Add a GitHub Actions workflow running `python -m pytest` + `python -m compileall hetmarl`.

### 9. Optional: drop the hard `ncps` dependency
Only `torchncp/__init__.py` (`LTCCell` import), `cfc.py`, and `ltc.py` pull in `ncps`.
MACMAT itself uses only the vendored `CfCCell`/`WiredCfCCell`/local wirings. If desired,
make the `LTC`/`LTCCell` exports lazy (import inside the function) so a default MACMAT run
needs no external `ncps`. Keep `ncps` in `requirements.txt` either way unless fully
decoupled.

### 10. Housekeeping
- Fix vendored MiniGrid `is`-with-literal `SyntaxWarning`s (e.g.
  `roomgrid.py:302 front_cell.type is 'wall'` → `==`).
- Add type hints + docstrings to the public runner/policy/env methods.
- Add a root `LICENSE` and confirm vendored license headers are retained.

## Suggested sequencing
1. Item 8 (tests) → safety net.
2. Item 4 (semantics enum) and Item 7 (comment cleanup) → low risk, immediate clarity.
3. Item 1 (eval/render unify) → big readability win, contained.
4. Item 2 (compute_global_goal) and Item 6 (RNN encapsulation) → highest risk, do last
   with the test net in place.
5. Items 3, 5, 9, 10 as capacity allows.

## Definition of done
- No function over ~150 lines in the runner/env hot paths.
- `eval()`/`render()` share a single rollout implementation.
- Zero magic entity codes outside `semantics.py`.
- Fixed-seed training/eval metrics identical to pre-refactor baseline.
- `pytest` smoke + import tests pass in CI.
