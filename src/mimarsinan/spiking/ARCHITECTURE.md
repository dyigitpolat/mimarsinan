# `mimarsinan/spiking/`

Host-side spike-train construction and boundary encoding shared by training
(NF cycle-accurate), SCM/HCM (`SpikingHybridCoreFlow`), and deployment-side
runners (SANA-FE, Lava, Nevresim).

## Files

| File | Exports | Role |
|---|---|---|
| `spike_trains.py` | `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` | Low-level spike-train constructors (signed IF — no relu on the membrane). |
| `lif_utils.py` | `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model` | Walk-and-unwrap activation helpers. |
| `boundary_config.py` | `BoundaryConfig` | Runtime config for boundary encode/decode (re-exported by `segment_boundary`). |
| `segment_boundary.py` | `BoundaryConfig`, `encode_segment_input`, `encode_compute_boundary`, `decode_segment_output`(`_torch`) | **Single source of truth** for boundary encode/decode (rates+cached trains → `(T,B,in)` spike train; spike counts → `counts/T` rates). Consumed identically by `SpikingHybridCoreFlow._forward_rate`, SANA-FE/Lava/Nevresim runners, and the segment-forward driver. |
| `compute_boundary.py` | `encode_compute_boundary` | Cycle-accurate spike emission for subsumed **plain LIF-Perceptron** ComputeOp boundaries (wrappers stay rate-mode). The input gather (`_gather_op_input_train`) applies the producer's `node_output_shifts` entry before its [0,1] clamp (Case B) and warns once per op on unshifted negatives. |
| `segment_partition.py` | `classify_spike_producers`, `partition_spike_segments`, `partition_perceptron_segments`, `perceptron_of`, `is_encoding_perceptron`, `is_value_boundary` | Mode-agnostic exec-graph classification. **Value boundaries** are the raw input and *every* host `ComputeOpMapper` (matching HCM, where every host ComputeOp runs once on decoded values); perceptrons produce spikes; structural nodes inherit. Segments = maximal connected spike regions; an `is_encoding_layer` perceptron starts a fresh segment (it consumes a decoded value). |
| `scale_aware_boundaries.py` | `calibrate_scale_aware_boundaries`, `propagate_boundary_input_scales` | Sets each block's `activation_scale` (theta_out) and forward-propagates `input_activation_scale` (= upstream theta_out) so the fixed-window TTFS [0,1] train normalizes/un-normalizes each boundary. |
| `distribution_matching.py` | `match_activation_distributions` | **Distribution-matching calibration** for the deployed single-spike TTFS cascade. (1) scale-aware boundaries from the teacher's per-perceptron activation `quantile` (theta_out); (2) `bias_iters` rounds of DFQ per-neuron bias correction — match each perceptron's cascade channel-mean to the teacher ANN's via `bias += eta*(ann_mean − cascade_mean)`, reviving death-cascade-starved deep neurons. Model must already be in deployed-TTFS state. Returns a stats dict (before/after mean `|gap|` + %dead over the DFQ window). Generic; no model-specific logic. |
| `segment_forward.py` | `SegmentForwardDriver` | **Unified segment-aware NF driver** — ONE walk for every spiking mode. Iterates the exec graph; host ComputeOps run **once on decoded values** (recording per-channel minima into `compute_min_recorder` and applying `_negative_shift` before downstream re-encode); spike segments are delegated to `policy.run_segment`. Fails loud (`NotImplementedError`) on host ops interleaved inside a neural segment (a segment runs atomically at its first member). |
| `segment_policies.py` | `LifSegmentPolicy`, `AnalyticalSegmentPolicy` (+ re-export `TtfsSegmentPolicy`) | Per-mode segment dynamics. **LIF**: perceptrons cascade per-cycle signed-IF off upstream trains; a subsumed (encoding) perceptron mirrors HCM's two outputs — host-op rate value for ComputeOp consumers + cycle-emitted train (plain LIF only; Conv wrappers stay uniform); region exits decode `count/T`. **Analytical** (`ttfs`/`ttfs_quantized`): every node runs once on ideal values (the pointwise-analytical NF). Top-level picklable classes (tuner installs pickle the forwards). |
| `segment_policy_ttfs.py` | `TtfsSegmentPolicy` | **TTFS** (`ttfs_cycle_based`): latency-windowed single-spike sim (1-cycle delay per perceptron hop, arrival latch, window `[depth, depth+T)`); non-encoding entries consume boundary values as single-spike TTFS trains (the rising edge of HCM's latched encode — ramp reconstruction makes the dynamics equal); a **subsumed encoding** entry charges the ideal decoded value directly (it mirrors HCM's host ComputeOp, so it is bias-mode-agnostic); region exits decode `count/T * activation_scale`. The per-neuron bias ramp is identical for `on_chip` and `param_encoded` (both give cumulative `bias·(t_local+1)`), so the NF forward does not branch on bias mode. **Drive-time bias install**: `prepare()` recomputes each perceptron's `effective_preactivation_bias` (norm-folded; differentiable) fresh on every drive and installs it on the `TTFSActivation`s; `finalize()` restores the raw `layer.bias` reference (the picklable stored contract). Subtracting the raw bias under a non-identity normalization poured `fused_b − b` into the ramp (the 2026-06-08 cascaded+offload fusion cliff); the drive-time recompute also closes the stale-reference class of bugs structurally. |
| `chip_aligned_nf.py` | `chip_aligned_segment_forward` | Thin LIF wrapper: `SegmentForwardDriver` + `LifSegmentPolicy` (falls back to `run_cycle_accurate` when the model has no mapper graph). The torch-side mirror of HCM `_forward_rate`; per-neuron parity reference (`test_nf_hcm_per_node_spike_parity_mmixcore.py`, `test_nf_hcm_multisegment_parity.py`). Installed as the LIF tuner's NF probe. |

## Boundary contract

`encode_compute_boundary(op, …)` returns a `(T, B, D)` binary spike train
when the op is a **plain LIF-Perceptron** boundary in cycle-accurate mode;
otherwise `None`. "Plain LIF-Perceptron" means
`op.params["module"]` exposes an `activation` that unwraps to `LIFActivation`
and is *not* a wrapper (e.g. `Conv2DPerceptronMapper`, where
`module.perceptron is not module`). Wrappers, non-LIF perceptrons, and
structural ops (mean/flatten/add/etc.) stay rate-only — the chip's
calibrated weights expect uniform encoding at those boundaries.

## Signed integrate-and-fire

`LIFActivation` and the spike-train builders charge the membrane with the
**signed** normalized input (`x / scale`, no `relu`): negative weighted input
lowers the membrane and may recover, matching the deployed chip / HCM
(`memb += W@s + b`). This is what makes per-neuron NF↔HCM spike-count parity
exact; the old relu-on-membrane diverged whenever a cycle's input went negative.
See `tests/unit/models/test_lif_step_vs_activation_parity.py`.

## `encode_segment_input` invariants

- A cached spike train in `state_buffer_spikes[node_id]` is consumed verbatim — never re-encoded.
- A missing non-raw input slice in cycle-accurate mode while *other* slices ARE cached is a hard error (`ValueError`): silently uniform-encoding a missing slice hides upstream emission bugs.
- "Raw input only" stages uniform-encode the gathered rate.
- In legacy rate mode, missing slices fall back to `rates_to_spike_train` with the configured `spike_mode`.

## `chip_aligned_segment_forward`

Installed by `LIFAdaptationTuner._finalize_forward` as `model.forward` at
**finalize** (`rate == 1.0`). The blend ramp runs in the value domain (golden,
non-destructive); this forward is installed only at finalize. All
downstream pipeline steps (WQ, NormFusion, SCM accuracy probes) then validate
against the same forward that Nevresim / SANA-FE / Lava run, closing the NF→chip
gap by construction. Falls back to `run_cycle_accurate` only when the model has
no mapper graph.

## Unified segment-forward model

`SegmentForwardDriver` is the single decode→compute→re-encode walk for all
modes; only the segment-internal spike dynamics differ per mode and live in the
policy's `run_segment`. LIF is the degenerate policy (depth-0, multi-spike,
no latch, uniform encode); TTFS is the latency-windowed single-spike policy.
`TTFSSegmentForward` (`models/spiking/training/ttfs_segment_forward.py`) and
`chip_aligned_segment_forward` are thin wrappers over the same driver, so the
boundary semantics (every ComputeOp = host value boundary) cannot drift between
modes.

## Dependencies

- **Internal**: `mimarsinan.mapping.ir` (`ComputeOp`, `IRSource`), `mimarsinan.mapping.packing.hybrid_hardcore_mapping` (`HybridStage`, `HybridHardCoreMapping`), `mimarsinan.models.activations` (`LIFActivation`, `run_cycle_accurate`), `mimarsinan.models.spiking.training.ttfs_segment_forward` (`TTFSSegmentForward` — `distribution_matching` drives the cascade for DFQ bias correction).
- **External**: `torch`, `spikingjelly.activation_based` (IFNode + surrogates).

## Dependents

- `mimarsinan.models.hybrid_core_flow.SpikingHybridCoreFlow` — calls `encode_compute_boundary` / `encode_segment_input` in `_forward_rate`.
- `mimarsinan.tuning.tuners.lif_adaptation_tuner` — installs `chip_aligned_segment_forward` as `model.forward` at finalize (post-ramp).
