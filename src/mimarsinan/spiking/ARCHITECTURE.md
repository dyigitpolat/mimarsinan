# `mimarsinan/spiking/`

Host-side spike-train construction and boundary encoding shared by training
(NF cycle-accurate), SCM/HCM (`SpikingHybridCoreFlow`), and deployment-side
runners (SANA-FE, Lava, Nevresim).

## Files

| File | Exports | Role |
|---|---|---|
| `spike_trains.py` | `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` | Low-level spike-train constructors (signed IF — no relu on the membrane). |
| `lif_utils.py` | `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model` | Walk-and-unwrap activation helpers. |
| `segment_boundary.py` | `SegmentBoundary`, `BoundaryConfig`, `encode_segment_input`, `encode_compute_boundary`, `decode_segment_output`(`_torch`) | **Single source of truth** for boundary encode/decode (rates+cached trains → `(T,B,in)` spike train; spike counts → `counts/T` rates). Consumed identically by `SpikingHybridCoreFlow._forward_rate`, SANA-FE/Lava/Nevresim runners, and `chip_aligned_segment_forward`. `SegmentBoundary` carries inert Round-2 seams (`shift`, `placement`, `spike_generation_mode`). |
| `chip_aligned_nf.py` | `chip_aligned_segment_forward` | **Segment-aware** chip-aligned NF forward — the torch-side mirror of HCM `_forward_rate`. Walks the mapper exec graph keeping a per-cycle spike `train` (intra-segment perceptron cascade) and a `rate` (`count/T`) per node; perceptrons run single-step **signed-IF**, and **each ComputeOp runs once on the decoded rate** (not per-cycle on spikes) with downstream perceptrons re-encoding — the decode→compute→re-encode HCM does at each boundary. Encoding layers stay subsumed (rate mode + uniform encode). Matches HCM for multi-segment models incl. non-linear ComputeOps (LayerNorm); per-neuron parity reference (`test_nf_hcm_per_node_spike_parity_mmixcore.py`, `test_nf_hcm_multisegment_parity.py`). Installed as the LIF tuner's NF probe. |

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

Installed by `LIFAdaptationTuner._after_run` as `model.forward` once the
blend ramp completes (`rate == 1.0`). All downstream pipeline steps (WQ,
NormFusion, SCM accuracy probes) then validate against the same forward
that Nevresim / SANA-FE / Lava run, closing the NF→chip gap by
construction. Falls back to `run_cycle_accurate` only when the model has no
mapper graph.

## Dependencies

- **Internal**: `mimarsinan.mapping.ir` (`ComputeOp`, `IRSource`), `mimarsinan.mapping.packing.hybrid_hardcore_mapping` (`HybridStage`, `HybridHardCoreMapping`), `mimarsinan.models.activations` (`LIFActivation`, `run_cycle_accurate`).
- **External**: `torch`, `spikingjelly.activation_based` (IFNode + surrogates).

## Dependents

- `mimarsinan.models.hybrid_core_flow.SpikingHybridCoreFlow` — calls `encode_compute_boundary` / `encode_segment_input` in `_forward_rate`.
- `mimarsinan.tuning.tuners.lif_adaptation_tuner` — installs `chip_aligned_segment_forward` as `model.forward` post-blend.
