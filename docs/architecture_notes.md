# MACMAT architecture notes

Design-rationale notes for alternatives that were explored while developing the
MACMAT encoder/decoder ([`macmat_transformer.py`](../hetmarl/algorithms/macmat/algorithm/macmat_transformer.py)).
These were previously kept as commented-out code in the source; they are recorded
here so the source stays readable while the reasoning is preserved.

## Recurrent memory cell: NCP / WiredCfC wiring presets

For the `NCP` and `MixedNCP` recurrent types the memory cell is a
`WiredCfCCell` wired with `NCP_No_Motor`. The wiring is sized to the recurrent
hidden width `n_hidden_r`. The shipped configuration uses `n_hidden_r = 128`;
the other presets below were used during sweeps over the memory width. The ratios
are roughly: `command_neurons ≈ inter_neurons / 2`,
`sensory_fanout ≈ inter_neurons / 4`, `inter_fanout ≈ command_neurons / 4`,
`recurrent_command_synapses ≈ command_neurons` (≈1 per command neuron).

| `n_hidden_r` | inter_neurons | command_neurons | sensory_fanout | inter_fanout | recurrent_command_synapses |
|-------------:|--------------:|----------------:|---------------:|-------------:|---------------------------:|
| 32           | 20            | 12              | 5              | 3            | 12                         |
| 64           | 40            | 24              | 10             | 6            | 24                         |
| 96           | 64            | 32              | 16             | 8            | 32                         |
| **128 (shipped)** | **80**   | **48**          | **20**         | **12**       | **48**                     |
| 144          | 96            | 48              | 24             | 12           | 48                         |
| 192          | 128           | 64              | 32             | 16           | 64                         |

For `MixedCfC`, an `AutoNCP(n_hidden_r, 1)`-wired `WiredCfCCell` was also tried in
place of the plain `CfCCell`; the shipped model uses the plain `CfCCell`
alongside the `LSTMCell` (the "Mixed" path runs both and fuses the CfC/NCP
hidden state).

## Memory–observation fusion (CfC / NCP encoder path)

After the recurrent cell updates the per-agent memory, the memory is fused with
the current observation embedding. Three variants were considered:

- **Option A — context-aware memory, updated**: run the memory through the
  graph-attention blocks first (`rnn_states = blocks(ln(rnn_states))`), feed that
  updated context-aware memory forward to the next macro-step, and fuse it with
  the observation.
- **Option B — context-aware memory, not propagated**: same graph attention over
  the memory, but the context-aware version is used only for the current step and
  not carried forward.
- **Option C — merge then attend (shipped)**: fuse the raw memory with the
  observation embedding first (`fusion_layer(cat(obs, memory))`), and let the
  downstream graph-attention blocks make the fused representation context-aware.

The shipped model uses **Option C**.

## Agent-class conditioning (NoClass ablation only)

When `use_classbased_action=False` (the NoClass ablation) the agent's class is
injected as a learned embedding instead of via class-specific heads. Two fusion
points were explored, each with two variants:

- **Observation encoder** — either (1) concatenate the class encoding with the
  observation embeddings and pass through an extra linear layer, or (2) add the
  class encoding to the observation embeddings.
- **Action encoder** — either (1) concatenate the class encoding with the action
  embeddings and pass through `action_encoder`, or (2) add the class encoding to
  the action embeddings.

The shipped NoClass ablation **adds** the class encoding in both places
(Option 2).
