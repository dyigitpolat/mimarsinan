# Time-Domain Weight-Reuse Scheduling — Design + Round-1 Prototype

## The insight

A deployment **schedule** over IR cores currently models **every** pass as a
*reprogram*: load X params onto Y cores. But for a layer where the **same** weights
apply to many inputs — a conv kernel sweeping a huge spatial map — the kernel banks
can be loaded **once** across cores (fixed mapping, maximum parallelism) and the
spatial positions **time-multiplexed** through the resident banks, with **no
reprogramming between the reused passes**. The cost of a reuse pass is activation
**data movement** at a sync point, not a parameter reload.

So a schedule = **M weight-reuse phases** (cheap) + **N reprogram phases**
(expensive, X params across Y cores), with `N` = #weight-distinct layers and
`M` = `total_passes − N`.

This is **not** on-chip weight-sharing hardware (impossible) — the weights are
physically resident across cores, just **not re-loaded** between reused passes.

## The boundary is already in the IR (and was being discarded)

`conv2d_mapper._map_to_ir` registers **one** `WeightBank` per conv
(`max_neurons=None` ⇒ `group_sizes=[out_channels]` ⇒ one `bank_id`) then calls
`add_shared_neural_core(weight_bank_id=bank_id)` **per spatial position**. So every
position-softcore of a conv shares **one** bank. The grouping is recoverable in
O(cores):

```python
{core.weight_bank_id for core in graph.get_neural_cores()}
```

A `NeuralCore` with its **own** `core_matrix` (an FC core, no shared bank) cannot be
time-multiplexed, so it is always its own reprogram.

## Why the cost model is the round-1 keystone (GAP-R)

`cost_extraction.py` had **zero** reprogram / data-movement term: the energy proxy is
`Σ_d neurons_d · S_d` (soma-only), latency is `Σ timesteps`, cores is per-segment
count. So **today a reuse phase and a reprogram phase cost identically (both 0)**. No
build-side optimization can be *valued* until the cost model tells them apart — which
is why the cost model must land **first**, before any replica-fill build.

New term:

```
mj_reuse = mj_per_reprogram · params_reloaded
         + mj_per_sync      · activation_bytes_moved
```

with chip coefficients `mj_per_reprogram` / `mj_per_sync` **defaulting to 0.0** ⇒ the
whole term is `0.0` ⇒ **byte-identical**.

`params_reloaded` = Σ over the N reprogram passes of the resident weight count (each
distinct bank / owned core counted **once**, not once per reused position).
`activation_bytes_moved` = Σ over the `total_passes − 1` sync barriers of the gathered
slice size (the existing `_build_consumer_counts` / `SegmentIOSlice.size` walk; wired
in R2/R3).

## Round-1 prototype (this branch) — SHIPPED, default-off byte-identical

Three localized, additive, tests-first pieces. They change **no** sim / packing /
build behavior and **no** deployed number; they make the reuse savings **visible** and
Pareto-rankable.

1. **Phase classification keystone** — `mapping/weight_reuse.py` (new):
   - `classify_segment_phases(cores, weight_banks) -> SegmentReusePhases` groups a
     segment's `NeuralCore`s by `weight_bank_id` into `reprogram_passes` (N) +
     `reuse_passes` (M) + `params_reloaded`.
   - `weight_reuse_plan_from_graph(graph) -> WeightReusePlan` aggregates over a graph
     (runs on the **post-pruning** IR for free — pruning shrinks the same per-core
     counts that feed the grouping, so the pruning intersection composes with no extra
     machinery).
   - `format_weight_reuse_summary(plan)` is the SCM-gate one-liner.
   - `SegmentReusePhases` / `WeightReusePlan` carry derived `total_passes`,
     `sync_barrier_count` (= `total_passes − 1`), `reuse_fraction`.

2. **Cost term** — `chip_simulation/cost_extraction.py`:
   - `CostRecord` gains `reprogram_passes` / `reuse_passes` / `params_reloaded` /
     `activation_bytes_moved` (all default 0) + the derived `total_passes` /
     `sync_barrier_count` / `reuse_fraction` + `reuse_mj(mj_per_reprogram=…,
     mj_per_sync=…)` (the `weight_reuse_mj` term, 0.0 at default coefficients).
   - `COST_RECORD_FORMAT_VERSION` 2 → 3 (`from_dict` rejects unknown fields by design,
     so the format bump is mandatory and guarded).

3. **Reserved capability bit** — `mapping/platform/mapping_structure.py`:
   - `ChipCapabilities.allow_weight_reuse` (default False), read by
     `from_platform_constraints`, exposed as a `MappingStrategy` property,
     **NOT** in `permission_kwargs` — the exact `allow_per_layer_s` reserved-gate
     pattern. The SCM-gate weight-reuse print is gated behind it (default-off ⇒ no
     print ⇒ byte-identical output).

### Quantified VGG16@224 collapse

13 conv + 3 FC ⇒ **N = 16** weight-distinct reprogram phases. Over the full
feature-map positions the schedule is `16 reprogram + 137 775 reuse phases` (≈100%
reused; the design's headline `N≈16 / M≈142` is the same split after positions are
batched into core-budget passes). Reprogram **reload events** drop from one-per-pass
to `N = 16` — ≥10× fewer (≈8600× at full position granularity); the widest segment
(the design's `features_6`) collapses `50 176` reprograms → `1 reprogram + 50 175
reuse`.

## Deferred (honest multi-round sequencing — gated on round-1)

- **R2 — replica-fill physical build + cross-pass sim residency** (parity-critical):
  `_build_scheduled` builds **one** resident-bank pool and streams positions instead
  of a fresh pool per pass; sim keeps banks resident across a reuse phase
  (`_evict_segment_cache` / stage-IO LRU). Turns visible savings into REAL on-hardware
  reuse. Also wires `activation_bytes_moved` from the real `SegmentIOSlice` walk.
- **R3 — VGG16@224 reuse-mode probe** against a real `_build_scheduled` + cost
  extraction, confirming `N≈16 / M≈142` end-to-end.

Round-1 makes the savings **measurable**; R2/R3 make them **real**.

## Base-state note (honest)

The task brief referenced a stale git snapshot (`HEAD=92734b9`, local main +49) on
which an E4 scheduling-aware capacity estimate (`CapacityEstimate` /
`estimate.py::_scheduled_estimate`, `phase_count`) was assumed shipped. The actual
worktree base is `bcacfeb` (= `origin/main`), on which that machinery does **not**
exist. Round-1 was therefore landed on the seams that **do** exist on this base — the
IR `weight_bank_id` grouping, `CostRecord`, and the `ChipCapabilities` reserved-bit
pattern — which is exactly the keystone slice (the cost model + classifier) the design
designates as round-1. Wiring the classifier into a `CapacityEstimate._scheduled_estimate`
`reuse_phase_count` / `reprogram_phase_count` is deferred to whichever round lands that
estimate (the classifier is the reusable primitive it would call).
