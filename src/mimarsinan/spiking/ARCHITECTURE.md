# spiking/ — Spike-train encoding, the unified segment-aware NF forward, and deployed-cascade calibration

This module is the torch-side single source of truth for how spiking models
execute and how values cross host/chip boundaries. `SegmentForwardDriver`
partitions a model's mapper exec graph into host value nodes and neural spike
segments and runs one decode→compute→re-encode walk for every mode; per-mode
spike dynamics (cycle-accurate LIF, single-spike TTFS, pointwise analytical)
live in pluggable segment policies. The same boundary encode/decode helpers
are consumed by `SpikingHybridCoreFlow` and the SANA-FE/Lava/Nevresim runners,
so NF↔SCM parity holds by construction; the calibration helpers (scale-aware
boundaries, DFQ bias matching, gain correction, theta co-training) close the
remaining ANN→SNN conversion gap on the deployed cascade.

## Key files
| File | Purpose |
|---|---|
| `spike_trains.py` | Spike-train constructors: uniform, cycle-accurate signed-IF (`lif_spike_train`), materialized, and the legacy rate fallback. |
| `boundary_config.py` | `BoundaryConfig` dataclass — runtime knobs for boundary encode/decode (T, spiking mode, cycle accuracy, dtype, negative shift). |
| `segment_boundary.py` | SSOT boundary encode/decode: `encode_segment_input` (cached trains take precedence; missing non-raw slices are a hard error), `decode_segment_output(_torch)` (counts / T), and `normalize_ttfs_boundary_value` (the TTFS wire-domain transcode: spike time encodes value / boundary scale). |
| `compute_boundary.py` | `encode_compute_boundary` — cycle-accurate spike emission for subsumed plain-LIF-Perceptron ComputeOp boundaries; wrapper mappers and non-LIF ops stay rate-mode. |
| `segment_partition.py` | Exec-graph classification (spike producer vs host value boundary) and union-find partition into maximal spike segments; encoding perceptrons start fresh segments. |
| `segment_forward.py` | `SegmentForwardDriver` — the mode-agnostic walk: host ComputeOps run once on decoded values (with min recording and `_negative_shift`), spike segments delegate to `policy.run_segment`. |
| `segment_policies.py` | `LifSegmentPolicy` (per-cycle signed-IF cascade, uniform re-encode at entries, `node_value_recorder` side-channel) and `AnalyticalSegmentPolicy` (every node once on ideal values); re-exports `TtfsSegmentPolicy`. |
| `segment_policy_ttfs.py` | `TtfsSegmentPolicy` — latency-windowed single-spike sim (arrival latch, ramp decode, window-relative wire-normalized boundary trains per consumer input scale), drive-time effective-bias install, optional offload-boundary STE (`boundary_surrogate_temp`). |
| `chip_aligned_nf.py` | `chip_aligned_segment_forward` — thin LIF wrapper (driver + `LifSegmentPolicy`, `run_cycle_accurate` fallback); the torch mirror of HCM `_forward_rate`. |
| `lif_utils.py` | Unwrap wrapped `LIFActivation`s; toggle `use_cycle_accurate_trains` model-wide. |
| `scale_aware_boundaries.py` | Set per-block `activation_scale` (theta_out, encoding layer pinned) and forward-propagate `input_activation_scale` via the polymorphic mapper walk. |
| `dfq_bias_correction.py` | Mode-agnostic DFQ core: teacher channel-mean capture (forward hooks) reduced on the perceptron's declared channel axis, the mask-aware per-neuron `bias += eta*(ann − cascade)` loop, and keep-best/early-stop over an injected deployed-behavior probe. |
| `distribution_matching.py` | TTFS distribution matching: quantile scale-aware boundaries + the DFQ loop over the deployed single-spike cascade; returns gap/dead-fraction stats. |
| `lif_distribution_matching.py` | LIF DFQ bias correction over the deployed cycle-accurate cascade (read via the `node_value_recorder` side-channel); no boundary retune. |
| `gain_correction.py` | Per-cascade-depth theta trim (`gamma^d`, encoding/entry pinned) inverting the TTFS ramp-decode death cascade; `apply_gain_at_rate` for ramped tuning. |
| `theta_cotrain.py` | Promote non-encoding perceptrons' `activation_scale` to trainable per-output-channel Parameters (rebound on every referencing node) for cascade fine-tuning. |

## Dependencies
- `mapping` — IR types (`ComputeOp`, `IRSource`) and `HybridHardCoreMapping`/`HybridStage` for boundary encode; mapper classes (`InputMapper`, `ComputeOpMapper`, `Conv1D/2DPerceptronMapper`) for node classification; `scale_propagation.walk_out_scales` for boundary-scale propagation.
- `models` — `LIFActivation`, `run_cycle_accurate`, `TTFSActivation`, `TransformedActivation` (neuron dynamics + unwrap); `effective_preactivation_bias` (norm-folded TTFS bias); `activation_channel_axis` (owner-declared DFQ channel-axis ground truth); lazily `TTFSSegmentForward` as the DFQ cascade readout.
- `transformations` — `pruning.committed_masks.commit_perceptron_pruning`: DFQ starts from the committed-pruning raw-parameter state (the deployed executor never fires enforcement hooks).
- `chip_simulation` — `spike_modes` spike-timing encoders (Uniform / TTFS), shared with the simulators so encode timing cannot drift.
- `tuning` — lazily `LIFBlendActivation` in `lif_utils` (unwrap/toggle during the blend ramp; deferred import avoids a cycle).

## Dependents
- `chip_simulation` — SANA-FE neural-stage recording (`decode_segment_output`); `spiking_mode_policy` installs `chip_aligned_segment_forward` as the NF probe.
- `mapping` — `support/bias_compensation` drives `SegmentForwardDriver` + `TtfsSegmentPolicy`.
- `models` — `nn/activations/lif.py` builds spike trains; `perceptron_mixer/perceptron.py` unwraps LIF activations.
- `pipelining` — the simulation factory toggles cycle-accurate trains; the SCM mapping and TTFS adaptation steps calibrate/propagate boundary scales.
- `tuning` — the LIF/TTFS adaptation tuners (chip-aligned forward, DFQ matching, gain correction) and the KD blend tuner (theta co-training).

## Exported API
`__init__.py` re-exports:
- `BoundaryConfig`, `encode_segment_input`, `encode_compute_boundary`, `decode_segment_output`, `decode_segment_output_torch` — the boundary contract.
- `SegmentForwardDriver`, `LifSegmentPolicy`, `TtfsSegmentPolicy` — the unified segment forward.
- `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` — spike-train constructors.
- `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model` — LIF helpers.
- `calibrate_scale_aware_boundaries`, `propagate_boundary_input_scales` — boundary-scale calibration.
- `match_activation_distributions`, `match_lif_activation_distributions` — DFQ distribution matching.

`AnalyticalSegmentPolicy`, `chip_aligned_segment_forward`, gain correction,
theta co-training, and the partition helpers are imported from their modules
directly.
