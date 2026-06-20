# models/spiking/training/ -- Spike-train KD training forwards

Differentiable forwards that fine-tune a model **through** its deployed spiking
dynamics (not a pointwise analytical surrogate), so KD recovery optimizes the
behaviour the chip actually runs. Analogous to LIF's `run_cycle_accurate`, but
for the cascaded `ttfs_cycle_based` single-spike cascade.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ttfs_segment_forward.py` | `TTFSSegmentForward`, `classify_spike_producers`, `partition_spike_segments`, `partition_perceptron_segments` | **Thin wrapper** over the unified `spiking.segment_forward.SegmentForwardDriver` with `TtfsSegmentPolicy` (the walk, classification, and partition live there; the partition symbols are re-exports). Each segment: entry perceptron (`is_encoding_layer`) encodes value→TTFS spike (`TTFSActivation(encoding=True)`); interior perceptrons ramp from arriving single spikes; **latency-gated** per `depth` (perceptron-hops from entry = local `ChipLatency`), 1-cycle delay per core hop; each region node consumed by a value node is decoded `count_of_latched_spikes / T * activation_scale` over its own window `[depth, depth+T)`. Segments run sequentially (TTFS single-spike timing is causal). Differentiable end-to-end. The `boundary_surrogate_temp` ctor kwarg (default `None`) is forwarded to `TtfsSegmentPolicy` — when set, the offload-boundary re-encode gets a straight-through backward so the genuine gradient trains every segment (forward byte-identical). |
| `blended_genuine_forward.py` | `BlendedGenuineForward` | Picklable `model.forward` override computing `(1 - rate) * teacher(x) + rate * genuine(x)`. `genuine` is a lazily built `TTFSSegmentForward` over the live model (`LazyExecutorForward` pattern — dropped on pickle); `teacher` is a frozen snapshot (`tuning.teacher.freeze_module`). The settable `.rate` scalar is read live each call so an axis can ramp it; `rate=0` reads the teacher exactly, `rate=1` the freshly built genuine cascade exactly, with gradients flowing into the model (not the teacher). The pure-cascade branch is exposed as the public `genuine_logits(x)` so the KD loss can add a genuine-CE term without reaching into a private member. The teacher→genuine OUTPUT blend that walks the cascade smoothly from ~ANN to the deployed single-spike dynamics. A `boundary_surrogate_temp` kwarg (default `None`) is forwarded to the genuine `TTFSSegmentForward` for the offload-boundary STE. |

## Design notes

- **Why pre-mapping:** segment entries come from `perceptron.is_encoding_layer`
  (set by `torch_mapping.encoding_layers.mark_encoding_layers` at conversion), so
  the driver runs on the unmapped trainable model — the fine-tuning step keeps its
  pipeline slot. Mirrors the deployed `SpikingHybridCoreFlow` but without a mapping.
- **Encoding cut:** an `is_encoding_layer` perceptron consumes a *decoded value*, so
  the partition never unions it with its source — its upstream spike region decodes,
  it re-encodes.
- **Every ComputeOp is a value boundary:** host ComputeOps run once on decoded
  values (never per-cycle on spikes), matching HCM where every host op executes
  between neural stages. This holds for non-linear ops (LayerNorm) too.
- **Latency gating:** each core integrates only inside `[depth, depth+T)` (no
  premature bias-only firing), with a 1-cycle output delay per perceptron hop.
  `depth` is the per-segment perceptron-hop count (local-latency approximation of
  `ChipLatency`). Exact per-core latency / per-source windows / multi-core tiling are
  mapping-dependent and not reproduced pre-mapping (residual train↔deploy gap).

## Dependencies

- **Internal**: `spiking.segment_forward` (`SegmentForwardDriver`,
  `TtfsSegmentPolicy`, partition re-exports).
- **External**: `torch` (transitively).

## Dependents

- `tuning.tuners.ttfs_cycle_adaptation_tuner.TTFSCycleAdaptationTuner` installs
  `TTFSSegmentForward` as `model.forward` during TTFS-cycle fine-tuning.

## Exported API (`__init__.py`)

`TTFSSegmentForward`, `partition_perceptron_segments`.
