# Latency engines — when to use which

Mimarsinan has **two** latency systems. They must not be merged: each runs at a different pipeline phase on a different graph representation.

## Summary

| Engine | Module | Input | Output | Used by |
|--------|--------|-------|--------|---------|
| `IRLatency` | `ir_latency.py` | `IRGraph` / `NeuralCore` | Per-core **tier** tag (`NeuralCore.latency`) — topological depth in the neural subgraph | SCM (`IRMapping`), wizard layout preview |
| `ChipLatency` | `chip_latency.py` | `HardCoreMapping` | Per-core **cycle index** for simulation scheduling | HCM, nevresim, Lava, SANA-FE |

## `IRLatency` (topology tier)

- **Question answered:** “How deep is this neural core in the IR dependency DAG?”
- **Algorithm:** Memoized walk from inputs; latency ≈ max(upstream neural latency) + 1 across neural hops.
- **Complexity:** O(nodes); no per-axon walk, no `is_off_` handling.
- **When to call:** Before hard-core packing, when assigning layout / latency-group metadata on `NeuralCore`.
- **Do not use for:** Cycle-accurate spike timing, buffer alignment, or nevresim `ChipModel` generation.

## `ChipLatency` (cycle schedule)

- **Question answered:** “On which simulation cycle does each hard core’s active window start, given axon delays and pruning artifacts?”
- **Phases:**
  1. Backward walk from `output_sources` → per-neuron delays; skip axons with `is_off_`.
  2. `_enforce_core_latency_invariant` — consumer latency ≥ max(source core latency) + 1.
  3. `_align_shiftable_cores` — shift cores whose only live inputs are off / always-on / segment-input so firing aligns with deepest consumer (fixes HCM↔nevresim↔SANA-FE drift after pruning).
- **When to call:** After softcores are packed into `HardCoreMapping`, before HCM / C++ codegen / parity simulators.
- **Do not use for:** Wizard “latency group” previews on unpacked IR, or SCM metric assignment on `NeuralCore` tags alone.

## Decision tree

```
Adding scheduling or layout feature?
├─ Operates on IRGraph / NeuralCore before hard-core pack?
│  └─ Touch IRLatency only (or layout mapper latency tags derived from it).
└─ Operates on HardCoreMapping / simulation / codegen?
   └─ Touch ChipLatency only (after pack, before sim).
```

## Shared helpers (optional)

If both engines need the same upstream-neural ID walk, add a tiny helper in `ir_segmentation.py` (e.g. `iter_upstream_neural_ids`). **Do not** expose a single `calculate_latency()` API across IR and chip domains.

## Pitfalls (ChipLatency)

- After IR pruning, a core may have only bias / always-on inputs → can fire at cycle 0 while consumers read at `L`. `_align_shiftable_cores` fixes this class of parity bugs.
- Incorrect chip latencies wedge NF→SCM accuracy on deep LIF graphs; use strict templates under `templates/mnist_*_strict_*` for regression.
