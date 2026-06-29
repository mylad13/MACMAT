# Pre-trained checkpoints

The **10 learned models** from the MACMAT paper: 6 memory architectures
(NoRNN, LSTM, NCP, MixedNCP, CfC, MixedCfC) and 4 structural ablations
(LSTM/MixedNCP × NoAttn/NoClass). Each `<Model>/` directory contains:

- `transformer.pt` — the trained weights (the only file the policy loads; see
  `hetmarl/algorithms/transformer_policy.py: restore`).
- `config.yaml` — the original Weights & Biases run config, kept for provenance.

The architecture is **rebuilt from the YAML args at load time**, so the flags in
`hetmarl/scripts/config/hydra_render_config.yaml` must match the checkpoint or
`load_state_dict` fails. The required flags per model are listed below.

## Global flags (already set in `hydra_render_config.yaml`)

| Flag | Value |
|---|---|
| `algorithm_name` | `macmat` |
| `encode_state` | `false` |
| `max_num_targets` | `5` |
| `num_agents` | `4` |
| `n_agent_types` | `2` |
| `n_block` / `n_embd` / `n_head` | `1` / `128` / `2` |
| `hidden_size` / `recurrent_hidden_size` / `n_embd_vit` | `128` / `128` / `48` |

## The 10 models

`params`/`keys` are the saved `transformer.pt` tensor count. All 10 were verified to
**strict-load** (0 missing / 0 unexpected keys) into the architecture built from this
repo's code.

| Model | params | keys | Per-model flags (in addition to globals) |
|---|---:|---:|---|
| **LSTM** (default) | 944,295 | 128 | `rnn_type: LSTM`, `use_rnn: true` |
| NoRNN             | 762,023 | 116 | `use_rnn: false` |
| NCP               | 926,631 | 142 | `rnn_type: NCP`, `use_rnn: true` |
| MixedNCP          | 1,058,727 | 146 | `rnn_type: MixedNCP`, `use_rnn: true` |
| CfC               | 911,143 | 134 | `rnn_type: CfC`, `use_rnn: true` |
| MixedCfC          | 1,043,239 | 138 | `rnn_type: MixedCfC`, `use_rnn: true` |
| LSTM-NoAttn       | 678,823 | 86  | `rnn_type: LSTM`, `use_graph_attention: false` |
| MixedNCP-NoAttn   | 793,255 | 104 | `rnn_type: MixedNCP`, `use_graph_attention: false` |
| LSTM-NoClass      | 938,356 | 126 | `rnn_type: LSTM`, `use_classbased_action: false` |
| MixedNCP-NoClass  | 1,052,788 | 144 | `rnn_type: MixedNCP`, `use_classbased_action: false` |

## How to render a model

1. In `hetmarl/scripts/config/hydra_render_config.yaml`, uncomment exactly one
   `model_dir` line and set the per-model flags from the table above (LSTM is the default).
2. From `hetmarl/scripts/`: `python render/render_gridworld.py`.
