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

## The window rule of a per-cycle cascade (U-TGAUGE5, `cycle_windows.py`)

- **Where it holds:** every plan whose mode policy answers `runs_per_cycle_cascade` (the LIF family and the timing
  family's cascaded schedule; the hybrid flow dispatches on the same property). Each core steps from its `ChipLatency`
  latency `L` and fills its axons at cycle `L + k` from each producer core's buffer of the previous cycle: at a latency
  gap `g` that is the producer's relative cycle `k + g - 1`, so only `g = 1` reads the producer's own cycle `k`.
- **One edge set:** an edge is an in-segment read that the one liveness predicate keeps (U-LIVE, below), the set the
  relay pass, `ChipLatency` and the assertion read. `silence_dead_segment_reads` also turns every dead in-segment read
  off before the relays (Soft Core Mapping before quantization; `depth_balancing.balance_cycle_windows` for a program
  built outside it).
- **What the dead reads leave unread (U-TGAUGE6, U-LIVE):** the unit is the neuron. A neuron no node and no output
  reads is marked compacted (`pruned_col_mask`, the IR's own column compaction), so the declared and the unbounded
  geometry hold the same neurons, and the same pass removes every core it leaves with no reader and no output source
  (every neuron unread), with that core's own reads, to a fixed point, by one call of the IR's whole-node deletion
  (`IRGraph.remove_nodes`, which IR pruning removes its dead cores with); a psum or coalescing group goes whole or
  stays, host operations stay. Such a core carries nothing, and under per-hop re-timing it would be a level stage
  without outputs. Soft Core Mapping's deployment-record fragment states `dead_reads_off` and `unread_cores_removed`
  only where a read was turned off, so a program without one keeps its record's bytes.
- **The relays:** the recipe of every per-cycle cascade arms `lif_depth_balancing_relays` (`ConversionPolicy.derive`,
  from the predicate); a declared `false` switches them off.
- **The assertion:** `ir_window_gaps` at the end of Soft Core Mapping's derivation (the executed units: whole segments,
  or one level each under per-hop re-timing), and `core_window_gaps` once per segment-cache fill in the software
  per-cycle executor (`_run_neural_segment_rate`, after the count path). Either refuses by name
  (`UnbalancedFanInError`, a `ToolchainRefusal`) with one of three causes: declared off, not armed, or the relays left
  the gap. The chip backends compute `ChipLatency` themselves and read programs the mapping has asserted.

## One liveness predicate for every scheduling rule (U-LIVE, `upstream.py`)

- **The rule:** an axon is live for scheduling exactly when it is not off, it reads neither the scheduled unit's input
  (the segment input, or a stage input no earlier pass of the fused segment produces) nor the always-on line, and its
  weight row holds a nonzero weight in a column the deployed core keeps (an IR node's compaction masks apply; a bound
  operand bank's placeholder, NaN until its payload is written, counts as nonzero). Per neuron, its entry is nonzero.
  A wired axon whose row is all zero, from any source (Weight Quantization rounding a weight to code zero, an exact
  zero of the trained module, a pruned column), counts as a source for no rule.
- **Where it is read:** `upstream.live_axons` through two adapters, `core_reads` (a hard core) and `ir_live_reads` (an
  IR node): `ChipLatency` (its sweep, shift alignment, consumer map and per-neuron delays), `IRLatency` (the IR's tiers
  and the record's `ir_max_latency`), the relay planner, the window rule (the dead-read pass, `ir_window_gaps`,
  `core_window_gaps`, `carried_window_gaps`), the pass projection (fused and own latencies, pins, delivery delays, the
  stage-input timing rule), the static check's `identity_timing` and the program check's core facts. A test over src's
  syntax trees holds the list (`tests/unit/mapping/test_live_sites.py`): a function in the schedule's closure that
  reads an axon's marks or a core's weights by itself is red until reviewed.

