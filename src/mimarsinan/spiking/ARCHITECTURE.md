# `mimarsinan/spiking/`

Host-side spike-train construction and boundary encoding shared by training
(NF cycle-accurate), SCM/HCM (`SpikingHybridCoreFlow`), and deployment-side
runners (SANA-FE, Lava, Nevresim).

## Files

| File | Exports | Role |
|---|---|---|
| `spike_trains.py` | `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` | Low-level spike-train constructors. |
| `lif_utils.py` | `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model` | Walk-and-unwrap activation helpers. |
| `segment_encoding.py` | `SegmentEncodingConfig`, `emit_compute_spike_train`, `build_segment_input_spike_train` | Cycle-accurate boundary emission and segment-input assembly used by `SpikingHybridCoreFlow._forward_rate`. |
| `chip_aligned_nf.py` | `chip_aligned_nf_forward` | NF forward that mirrors the chip's encoding semantics — encoding-layer perceptrons run once in rate mode, their outputs are uniform-encoded per cycle, and the rest of the graph runs single-step LIF for `T` cycles. |

## Boundary contract

`emit_compute_spike_train(op, …)` returns a `(T, B, D)` binary spike train
when the op is a **plain LIF-Perceptron** boundary in cycle-accurate mode;
otherwise `None`. "Plain LIF-Perceptron" means
`op.params["module"]` exposes an `activation` that unwraps to `LIFActivation`
and is *not* a wrapper (e.g. `Conv2DPerceptronMapper`, where
`module.perceptron is not module`). Wrappers, non-LIF perceptrons, and
structural ops (mean/flatten/add/etc.) stay rate-only — the chip's
calibrated weights expect uniform encoding at those boundaries.

## `build_segment_input_spike_train` invariants

- A cached spike train in `state_buffer_spikes[node_id]` is consumed verbatim — never re-encoded.
- A missing non-raw input slice in cycle-accurate mode while *other* slices ARE cached is a hard error (`ValueError`): silently uniform-encoding a missing slice hides upstream emission bugs.
- "Raw input only" stages uniform-encode the gathered rate.
- In legacy rate mode, missing slices fall back to `rates_to_spike_train` with the configured `spike_mode`.

## `chip_aligned_nf_forward`

Installed by `LIFAdaptationTuner._after_run` as `model.forward` once the
blend ramp completes (`rate == 1.0`). All downstream pipeline steps (WQ,
NormFusion, SCM accuracy probes) then validate against the same forward
that Nevresim / SANA-FE / Lava run, closing the NF→chip gap by
construction. Falls back to `run_cycle_accurate` when the model has no
mapper graph or no encoding-layer perceptron.

## Dependencies

- **Internal**: `mimarsinan.mapping.ir` (`ComputeOp`, `IRSource`), `mimarsinan.mapping.hybrid_hardcore_mapping` (`HybridStage`, `HybridHardCoreMapping`), `mimarsinan.models.activations` (`LIFActivation`, `run_cycle_accurate`).
- **External**: `torch`, `spikingjelly.activation_based` (IFNode + surrogates).

## Dependents

- `mimarsinan.models.hybrid_core_flow.SpikingHybridCoreFlow` — calls `emit_compute_spike_train` / `build_segment_input_spike_train` in `_forward_rate`.
- `mimarsinan.tuning.tuners.lif_adaptation_tuner` — installs `chip_aligned_nf_forward` as `model.forward` post-blend.
