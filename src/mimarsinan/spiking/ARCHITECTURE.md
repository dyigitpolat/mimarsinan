# `mimarsinan/spiking/`

Host-side spike-train construction and boundary encoding shared by training
(NF cycle-accurate), SCM/HCM (`SpikingHybridCoreFlow`), and deployment-side
runners (SANA-FE, Lava, Nevresim).

## Files

| File | Exports | Role |
|---|---|---|
| `spike_trains.py` | `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` | Low-level spike-train constructors. |
| `lif_utils.py` | `unwrap_lif_activation`, `apply_cycle_accurate_trains_to_model`, `boundary_lif_activation` | Walk-and-unwrap helpers; ephemeral LIF factory. |
| `segment_encoding.py` | `SegmentEncodingConfig`, `BoundaryKind`, `BoundaryLifCache`, `classify_encoding_boundary`, `emit_compute_spike_train`, `build_segment_input_spike_train` | Boundary spike-train classification + cycle-accurate emission. **Single source of truth** for how compute-op outputs become per-cycle spike trains for downstream neural segments. |

## Boundary contract

Every IR ComputeOp that feeds a neural segment has a **boundary kind** that
determines whether/how its output emits a `(T, B, D)` spike train:

| `BoundaryKind` | Condition | Emission |
|---|---|---|
| `ENCODING_LIF_PERCEPTRON` | `op_type == "module"` and wraps a `Perceptron` whose `activation` resolves to `LIFActivation` | Run perceptron `T` times in single-step mode on the gathered input train (uniform-encoded if upstream is raw). Matches NF semantics exactly. |
| `ENCODING_SPLIT_HOST` | `op_type == "module"` and wraps a bare `Conv*` / `Linear` / `Sequential(Conv*, …)` | Run module `T` times on per-cycle input, wrap output through an ephemeral `LIFActivation` from `BoundaryLifCache` (scale from `hybrid_mapping.node_activation_scales[op.id]`). |
| `STRUCTURAL_PASSTHROUGH` | mean/flatten/add/etc. | None — rate alone suffices; consumer uniform-encodes if needed. |
| `LEGACY_RATE` | `config.use_cycle_accurate_trains == False` | None — caller falls back to legacy `rates_to_spike_train`. |
| `RAW_INPUT` | The very first stage reading `node_id == -2` directly | Uniform-encode raw input (handled inside `build_segment_input_spike_train`). |

## `build_segment_input_spike_train` invariants

- A cached spike train in `state_buffer_spikes[node_id]` is consumed verbatim — never re-encoded.
- A missing non-raw input slice in cycle-accurate mode while *other* slices ARE cached is a hard error (`ValueError`): silently uniform-encoding a missing slice hides upstream emission bugs.
- The "raw input only" stage uses uniform encoding of the gathered rate.
- In legacy rate mode, missing slices fall back to `rates_to_spike_train` with the configured `spike_mode`.

## Dependencies

- **Internal**: `mimarsinan.mapping.ir` (`ComputeOp`, `IRSource`), `mimarsinan.mapping.hybrid_hardcore_mapping` (`HybridStage`, `HybridHardCoreMapping`), `mimarsinan.models.activations` (`LIFActivation`), `mimarsinan.chip_simulation.spike_modes` (low-level encoders).
- **External**: `torch`, `spikingjelly.activation_based` (IFNode, surrogates), `numpy` (deployment-side numpy wrappers — added in Phase C).

## Dependents

- `mimarsinan.models.hybrid_core_flow.SpikingHybridCoreFlow` — torch path.
- `mimarsinan.chip_simulation.simulation_runner` — numpy path (Nevresim, Phase C).
- `mimarsinan.chip_simulation.sanafe.runner` and `mimarsinan.chip_simulation.lava_loihi_runner` — numpy path (Phase C).
- `mimarsinan.tuning.tuners.lif_adaptation_tuner` — indirectly (via the hybrid flow used in `LifChipAlignedFinetuneTuner`, Phase D.2).
