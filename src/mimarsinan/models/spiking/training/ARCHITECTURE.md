# models/spiking/training/ -- Spike-train KD training forwards

Differentiable forwards that fine-tune a model **through** its deployed spiking
dynamics (not a pointwise analytical surrogate), so KD recovery optimizes the
behaviour the chip actually runs. Analogous to LIF's `run_cycle_accurate`, but
for the cascaded `ttfs_cycle_based` single-spike cascade.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ttfs_segment_forward.py` | `TTFSSegmentForward`, `classify_spike_producers`, `partition_spike_segments`, `partition_perceptron_segments` | Segment-aware TTFS spike forward over a `ModelRepresentation` exec graph. Walks the graph; **value-producing** nodes (raw input, unbounded raw-Linear/Conv `ComputeOp`s) run host-side, **spike-producing** regions (perceptrons + transparent routing downstream of them) run a `T`-cycle single-spike sim. Each segment: entry perceptron (`is_encoding_layer`) encodes value→TTFS spike (`TTFSActivation(encoding=True)`); interior perceptrons ramp from arriving single spikes; **latency-gated** per `depth` (perceptron-hops from entry = local `ChipLatency`), 1-cycle delay per core hop; each region node consumed by a value node is decoded `count_of_latched_spikes / T * activation_scale` over its own window `[depth, depth+T)`. Segments run sequentially (TTFS single-spike timing is causal). Differentiable end-to-end. |

## Design notes

- **Why pre-mapping:** segment entries come from `perceptron.is_encoding_layer`
  (set by `torch_mapping.encoding_layers.mark_encoding_layers` at conversion), so
  the driver runs on the unmapped trainable model — the fine-tuning step keeps its
  pipeline slot. Mirrors the deployed `SpikingHybridCoreFlow` but without a mapping.
- **Encoding cut:** an `is_encoding_layer` perceptron consumes a *decoded value*, so
  the partition never unions it with its source — its upstream spike region decodes,
  it re-encodes. Linear compute ops (mean/transpose/add) commute with the decode, so
  running them on spikes then decoding equals decoding then running them.
- **Latency gating:** each core integrates only inside `[depth, depth+T)` (no
  premature bias-only firing), with a 1-cycle output delay per perceptron hop.
  `depth` is the per-segment perceptron-hop count (local-latency approximation of
  `ChipLatency`). Exact per-core latency / per-source windows / multi-core tiling are
  mapping-dependent and not reproduced pre-mapping (residual train↔deploy gap).

## Dependencies

- **Internal**: `models.nn.activations.ttfs_spiking` (`TTFSActivation`),
  `mapping.mappers.structural` (`InputMapper`), `mapping.mappers.compute_op_mapper`
  (`ComputeOpMapper`), `torch_mapping.encoding_layers` (`_wraps_unbounded_raw_linear_or_conv`).
- **External**: `torch`.

## Dependents

- `tuning.tuners.ttfs_cycle_adaptation_tuner.TTFSCycleAdaptationTuner` installs
  `TTFSSegmentForward` as `model.forward` during TTFS-cycle fine-tuning.

## Exported API (`__init__.py`)

`TTFSSegmentForward`, `partition_perceptron_segments`.
