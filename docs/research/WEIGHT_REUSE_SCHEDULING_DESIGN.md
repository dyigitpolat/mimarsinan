# Time-Domain WEIGHT-REUSE Scheduling — design

**Question.** The Scheduled-mapping path (`allow_scheduling`) models EVERY deployment
phase as a **reprogramming** pass: a fresh physical core pool is built per sub-segment
(`_make_available_hardware_cores` per pass), i.e. "load X params onto Y cores" every
phase. For VGG16@224 that is ≈158 phases (capacity estimate) / 209 (per-segment
fresh-pool) / 239 (real `split_softcores_by_capacity` sub-segments), all charged as if
each were a full weight reload — except that **no cost term charges them at all** (GAP-R).

But a 224²-spatial conv is **one weight kernel applied to ~50K input positions**. The
positions do NOT each need their own weight reload: the kernel is **physically resident**
(one registered `weight_bank_id`); the only thing that changes pass-to-pass is **which
input patch / output position streams through the fixed mapping**. That is the
user's insight — a **time-domain WEIGHT-REUSE phase**:

> Load the kernel ONCE across all cores (fixed mapping, max parallelism), then
> **time-multiplex the input/output data** through them, gathering outputs at sync
> points, with **NO reprogramming between the reused passes**. Cost = activation **data
> movement** at the sync points, NOT parameter reload.

So a schedule decomposes into **M weight-reuse phases (cheap) + N reprogramming phases
(expensive: X params onto Y cores)**. This is **NOT** on-chip weight-sharing hardware
(impossible — the chip cannot share one crossbar across positions): the weights are
**physically resident across many cores, just not re-loaded between the reused passes**.

This doc decides: (1) the WEIGHT-REUSE scheduling **mode**; (2) the **cost model** that
splits a schedule into M reuse + N reprogram phases and slots into `cost_extraction` +
the scheduling-aware capacity estimate; (3) the **fixed-mapping-max-parallelism +
time-mux-output** mechanics; (4) the **pruning intersection**; (5) the highest-leverage
tractable **prototype target** this round. Every claim is read off the shipped code seams
cited inline. **No production code was modified to write this doc.**

**Git state (verified).** `HEAD = 92734b9`; local `main` is 49 commits ahead of
`origin/main` (`bcacfeb`), and `origin/main` is an ancestor of `HEAD`
(`git merge-base --is-ancestor origin/main HEAD` → true). Local main is the freshest;
**no fast-forward needed**. The E4 round-1 scheduling-aware capacity work
(`E4_PLACEMENT_SCALING_DESIGN.md` §3 "proposed extension") is **already SHIPPED** on this
HEAD: `estimate_cores_needed` takes `allow_scheduling`, `CapacityEstimate` carries
`scheduled` / `peak_phase_cores` / `phase_count`, and the SCM gate prints "N reprogram
phases" (`soft_core_mapping_step.py:264-270`).

---

## 0. The load-bearing fact: the reuse relationship is ALREADY in the IR, and DISCARDED

The "same weights, many inputs" relationship is **already recorded losslessly** — the
codebase just never uses it as a reuse lever:

- `conv2d_mapper.py::_map_to_ir` (216-279): with `max_neurons=None` (the conv mapper's
  default, from `conv_mixin.py::_convert_conv2d`), `group_sizes = [out_channels]` ⇒
  **ONE** `register_weight_bank` call ⇒ **one `bank_id` per conv layer**. Then
  `for pos in range(h_out*w_out)` calls `add_shared_neural_core(weight_bank_id=bank_id,
  ...)` **once per spatial position** (266-273). EVERY position-softcore of a conv layer
  points at the **SAME** `weight_bank_id`.
- The IR `WeightBank` (`mapping/ir/types.py:14`) is a shared object;
  `NeuralCore.weight_bank_id` (`types.py:91`) is the back-reference, with
  `has_shared_weights` / bank resolution already on the class. So **"these N cores reuse
  one resident weight set" is a fact `{core.weight_bank_id for core in segment.nodes}`
  exposes in O(cores), with zero new bookkeeping**.

**Why it is currently a REPROGRAM, not a REUSE.** Every downstream consumer collapses the
shared-bank fact into independent reprogramming work:

| consumer | seam | what it does today |
|---|---|---|
| capacity estimate | `estimate.py::_scheduled_estimate` (259) | `phase_count = Σ ceil(segment_bound/budget)` — every chunk of positions = a full reprogram pass |
| scheduled splitter | `schedule_split.py::split_softcores_by_capacity` (38) | `_halve_until_packs` recursively splits the conv's positions across many **fresh-pool** passes; never asks "do these share a bank?" |
| physical build | `hybrid_build_scheduled.py::_build_scheduled` (80) | `_make_available_hardware_cores(cores_config)` = brand-new pool **per sub-segment** = "reprogram every pass" |
| sim (in-stage) | `stage_io.py::_get_segment_tensors` (162) | uploads each bank ONCE per stage (`_ensure_bank_tensor`) and slices per position — **in-stage reuse already exists** — but the `id(stage)`-keyed LRU (`flow.py:135 _evict_segment_cache`) EVICTS between scheduled passes, so cross-pass reuse is thrown away |
| cost model | `cost_extraction.py` (GAP-R) | NO reprogram OR data-movement term: energy ∝ `Σ_d neurons_d·S_d` (soma), latency = `Σ_d timesteps_executed_d`; a 158-phase schedule is costed as if reprogramming were free |

**The pivotal honesty:** because `cost_extraction` has **no** reprogram/data-movement term
today, there is **literally no cost difference between a reuse phase and a reprogram
phase** to exploit yet. The cost model is the FIRST thing that must exist before any
build-side savings are real — which sets the prototype target (§5).

---

## 1. The WEIGHT-REUSE scheduling MODE

### 1.1 Definition

A **weight-reuse phase** is a maximal run of consecutive scheduled passes over **one
NeuralSegment that resolves to a SINGLE `weight_bank_id`** (or, more generally, a set of
banks small enough to be co-resident — see §1.4). Across that run:

- **the kernel banks are loaded ONCE** (fixed mapping, max parallelism — fill all cores
  with this layer's weights, §3);
- **input patches / output positions are TIME-MULTIPLEXED** through the fixed mapping;
- **outputs are gathered at the segment-exit sync point** (the existing `state_buffer`
  read/write between adjacent `HybridStage`s — `flow.py::forward`);
- **NO reprogramming** happens between the reused passes — the only inter-pass cost is the
  **activation data movement** at each sync point.

A **reprogramming phase** is a phase that **swaps the resident banks**: the first pass of a
new segment / a new `weight_bank_id` set. Its cost is the full "move X params onto Y
cores" reload.

A schedule is then **M weight-reuse phases + N reprogramming phases**, where
**N = number of distinct resident-bank-set boundaries** (≈ number of weight-distinct
layers) and **M = total passes − N** (the position-streaming passes that reuse a resident
bank).

### 1.2 The reuse-vs-reprogram boundary IS the existing same-bank grouping

The classification is NOT new machinery — it is the grouping the splitter already computes:

- `split_softcores_by_capacity` groups softcores by `latency_tag` (one conv = one latency
  group, all same DAG depth) and, when `allow_coalescing`, by coalescing-bundle
  (`schedule_budget._coalescing_bundles`). Within one latency group **all softcores share
  one `weight_bank_id`** for a conv. So:
  - **passes that split ONE latency group** (a conv's positions, via `_halve_until_packs`)
    ⇒ **REUSE passes** (same resident bank, just more positions);
  - **a pass that crosses into a new latency group / segment** ⇒ a **REPROGRAM pass** (new
    bank set).

The boundary is therefore **"does this pass's resident `weight_bank_id` set equal the
previous pass's?"** — and that set is `{sc.weight_bank_id}` over the pass's softcores,
already on every spec.

### 1.3 RESERVED capability gate (default-off, byte-identical)

Add the mode behind a **reserved `ChipCapabilities` bit**, following the EXACT
`allow_per_layer_s` pattern (`mapping_structure.py:88-144`):

```
allow_weight_reuse: bool = False     # a.k.a. allow_fixed_mapping
```

- Declared in `ChipCapabilities`, read by `from_platform_constraints` (one
  `bool(constraints.get("allow_weight_reuse", False))` line), exposed as a
  `MappingStrategy.allow_weight_reuse` property, **NOT** spread into
  `permission_kwargs()` (it is a phase-classification/cost capability, not a layout/verify
  input — same treatment `allow_per_layer_s` gets).
- **Default False ⇒ every phase classified as reprogram ⇒ byte-identical** to today on
  every axis (the cost term and the phase split both degenerate to the current model).
- The SAME 4 consumers that branch on `allow_scheduling` learn to branch on
  `allow_weight_reuse` (capacity estimate, splitter, sim cross-pass cache, cost model).
  Round-1 touches only the **cost model + capacity estimate classification** (§5); the
  build/sim cross-pass residency (`_evict_segment_cache`) is round-2+.

### 1.4 Co-residency: when can M passes truly reuse?

A reuse phase requires the kernel banks to **stay physically resident across all its
sub-passes**. Two regimes:

1. **Fixed-mapping fits the whole layer at peak** (one resident tiling, positions streamed
   through it): the layer's **atomic unit** (`frags·groups` of one position-softcore) is
   tiny (≤18 for VGG's widest conv), and a **single tiling of the kernel** fits a small
   slice of the chip; the chip is **filled with replicas of that tiling** (§3) and many
   positions run in parallel per pass. The banks never change ⇒ all passes reuse.
2. **Even one position-tiling + replicas exceed the chip** (pathological — a single conv
   whose one coalesced bundle > whole chip): genuinely infeasible, already the
   atomic-unit hard gate in `_scheduled_estimate` (`max_atomic ≤ cores_available`). Not a
   reuse case.

For VGG16@224 every conv is regime 1 (atomic unit ≤18 ≪ 2048-core budget), so **every
conv layer is one reuse phase**, and the only reprogram phases are the **layer
boundaries**.

---

## 2. The COST MODEL (the GAP-R term, reuse-aware)

### 2.1 Two cost classes

| phase class | count | cost | unit |
|---|---|---|---|
| **reprogram** | N (≈ #weight-distinct layers) | `mj_per_reprogram(X·Y)` per phase: load X params onto Y cores | param bytes / cores written |
| **reuse** | M (= total_passes − N) | `mj_per_sync(bytes)` per sync barrier: activation data movement only | activation bytes at the sync point |

Total deployment cost gains an additive term on top of the existing soma forward-pass
energy:

```
mj_total ≈ base_soma_term                          # existing: Σ_d neurons_d·S_d (unchanged)
         + N · mj_per_reprogram(params_reloaded)   # NEW: weight reloads (N reprogram phases)
         + (M + N − 1) · mj_per_sync(act_bytes)     # NEW: activation movement at every sync barrier
```

`(M + N − 1)` = total sync barriers = `total_passes − 1` (every phase boundary, reuse or
reprogram, hands activations through the `state_buffer`; the count is exactly
`compute_schedule_sync_count` = Σ max(n_passes−1,0) per segment plus inter-segment
handoffs). The split into the reprogram subset (N) vs reuse subset (M) is what makes the
schedule cheap: **reuse passes pay only `mj_per_sync`, not `mj_per_reprogram`**.

`mj_per_reprogram` and `mj_per_sync` are **declared chip coefficients** (like the SANA-FE
energy coefficients already consumed), defaulting to **0.0** ⇒ the whole new term is 0 ⇒
byte-identical. They become non-zero only on a chip that declares a reprogramming /
DMA energy, behind `allow_weight_reuse`.

### 2.2 The quantities are all already enumerable

- **`reprogram_passes` (N), `reuse_passes` (M):** from the phase classification (§1.2) on
  the capacity estimate's per-segment pass split — `N = Σ_segment (1 if segment has a
  single resident bank-set else its bank-boundary count)`, `M = phase_count − N`.
- **`params_reloaded`:** Σ over reprogram passes of (X params × Y cores written) =
  the resident bank's `weights.size` × the tiling's hard-core count, read off the
  `WeightBank` the segment's `weight_bank_id` resolves to.
- **`activation_bytes_moved`:** Σ over sync barriers of the `SegmentIOSlice.size` sum on
  each `HybridStage.input_map` / `output_map` (`hybrid_types.py:23-43`).
  `flow.py::_build_consumer_counts` (94) **already walks exactly these slices** to
  refcount the `state_buffer` — the data-movement term reuses that existing walk
  (`Σ s.size for s in stage.input_map`), so it is a pure read, no new traversal.

### 2.3 Slots in `cost_extraction` + the capacity estimate

- **`CostRecord`** (`cost_extraction.py:136`) gains four additive fields (all defaulting
  so old records still parse — but **bump `COST_RECORD_FORMAT_VERSION` 2→3**, since the
  reader rejects unknown fields by design at `from_dict`:204-210):
  `reprogram_passes: int = 0`, `reuse_passes: int = 0`, `params_reloaded: int = 0`,
  `activation_bytes_moved: int = 0`. Optionally a derived `reprogram_mj` /`sync_mj` for
  the Pareto scatter (the existing `cost_tuple()` / `_dominates` axes can stay; a later
  round adds the new mj as a 5th axis once coefficients are real).
- **`extract_cost_record`** reads the phase split from the capacity estimate (passed
  alongside the SANA-FE snapshot) and the activation bytes from the hybrid mapping's
  stage I/O maps. Pure read of numbers already on disk / in the mapping.
- **`CapacityEstimate`** (`estimate.py:72`) gains `reuse_phase_count: int` and
  `reprogram_phase_count: int` (with `reprogram_phase_count + reuse_phase_count =
  phase_count`), computed in `_scheduled_estimate` from the per-segment same-bank
  grouping. Defaults preserve the current single-`phase_count` reporting. The SCM gate
  print (`soft_core_mapping_step.py:265`) extends to "N reprogram + M reuse phases".

This is precisely the GAP-R term the E4 design (§2 GAP-R, §5 round-3) reserved
("`reprogram_passes`, `mj_per_reprogram`") — now refined with the **reuse split** the
user's insight adds.

### 2.4 Expected reduction for VGG16@224 — quantified

VGG16@224 has **13 conv layers** (each one weight bank) + 3 FC layers. Under the current
all-reprogram model the ~158-phase schedule is **158 reprogram passes**. Under the
weight-reuse mode:

- Each conv layer's positions are split across many passes **that all reuse one resident
  bank** ⇒ those are **REUSE passes**.
- Only the **first pass of each weight-distinct layer** is a **REPROGRAM pass**.

So **N ≈ 16 reprogram phases** (13 conv + 3 FC) and **M ≈ 158 − 16 = 142 reuse phases**.
The single most expensive segment, `features_6` (50,176 position-softcores ⇒ ~74 phases on
2048 cores), collapses from **74 reprogram passes to 1 reprogram + 73 reuse passes**.

**The reprogram cost drops by ≈10× (158 → ~16 reload events)**; the residual cost is the
142 reuse-pass sync-barrier activation movements (cheap, bounded by feature-map byte size,
not param byte size). This is the entire point of the user's insight: **most of the 158
phases become cheap reuse phases.**

(Honest caveat: "≈16" assumes each conv's one resident tiling fits and is replicated
across the chip so positions stream without a bank swap. If a chip is so small that even
one position-tiling+replicas cannot hold the layer's whole working set, a layer may need
intra-layer bank-set rotation — but for VGG conv banks the kernel is ≤4608×256 params,
trivially resident, so the 16-reprogram figure holds on a 2048-core chip.)

---

## 3. FIXED-MAPPING-MAX-PARALLELISM + TIME-MUX-OUTPUT mechanics

For a layer that produces a huge number of position-softcores from ONE kernel:

1. **Fill all cores with the kernel (max parallelism).** The kernel's one tiling
   (`frags·groups` hard cores, ≤18 for VGG) is **replicated** across the chip:
   `floor(budget / atomic_unit)` replicas run in parallel. On a 2048-core chip with a
   3-core `features_6` tiling, that is **682 positions per pass** (matching
   E4 §2's 682 softcores/phase). The mapping is **fixed** — all replicas hold the SAME
   resident bank.
2. **Time-multiplex the input positions through the fixed mapping.** Pass *p* feeds
   patches for positions `[p·682, (p+1)·682)` into the resident replicas; outputs for those
   positions are produced in parallel. Cores within one latency group have **no inter-core
   deps** (`schedule_split.py` docstring), so any position partition is functionally
   equivalent — exactly the property `_halve_until_packs` already relies on.
3. **Gather outputs at the segment-exit sync point.** Each pass writes its 682 positions'
   outputs into the `state_buffer` via the `HybridStage.output_map`
   (`SegmentIOSlice`s). The downstream consumer reads the assembled full feature map at the
   barrier (`flow.py::forward` sequential stage walk). The output is **time-multiplexed
   into the buffer**, gathered at the boundary — no reprogramming between passes.

The mechanics already exist for the **build/sim** side (the splitter produces the passes,
the flow gathers at sync points). The **only** delta the mode adds is: (a) **keep the
resident banks loaded across the passes** instead of evicting (`_evict_segment_cache` must
become reuse-aware — round-2), and (b) **classify+cost** those passes as reuse not
reprogram (round-1, §5). The "fill all cores with replicas of one resident tiling" is a
build-side optimization that the current `_build_scheduled` does NOT yet do (it builds a
fresh pool per sub-segment); under the mode it builds **one replica-filled pool and
streams positions** — that is the round-2/3 physical-build change, gated and tested
separately.

---

## 4. The PRUNING INTERSECTION

Pruning (`mapping/pruning/`) shrinks per-core axon/neuron counts, which feeds **exactly the
inputs the reuse-vs-reprogram math reads**:

- Pruning changes `core.get_input_count()` / `get_output_count()`, which feed
  `_segment_lower_bound`'s `frags = coalescing_fragment_count(in_count, max_axons)` and
  `groups = ceil(out/max_neurons)` (`estimate.py:141-143`). So pruning directly changes:
  - **(a) cores-needed per segment** (smaller `frags·groups` ⇒ smaller `segment_bound`);
  - **(b) `phase_count = Σ ceil(segment_bound/budget)`** ⇒ **fewer passes**;
  - **(c) the atomic-unit / fixed-mapping tiling size** ⇒ **more replicas fit ⇒ more
    positions per pass ⇒ even fewer passes**;
  - **(d) the per-phase data-movement** (smaller pruned feature maps ⇒ fewer activation
    bytes at each sync point).
- **The shared design surface:** the reuse cost term reads **post-pruning IR counts** for
  free — `estimate_cores_needed` already runs on the pruned `ir_graph` at the SCM gate
  (the gate runs after pruning in the pipeline). So **pruning fewer/smaller cores ⇒ fewer
  AND cheaper reprogram passes AND smaller per-phase data movement** — the exact
  intersection the user names. No code currently couples pruning to a reprogram cost
  because **no reprogram cost exists yet**; once the §2 term lands, pruning's benefit
  shows up automatically in `reprogram_passes`, `params_reloaded`, and
  `activation_bytes_moved` because all three are derived from the (already pruned) IR
  counts and bank sizes.

The clean abstraction is therefore: **the cost term is a pure function of the
post-pruning IR + the resident-bank grouping + the chip coefficients** — pruning and
weight-reuse compose by both feeding that one function, with no special-case coupling.

---

## 5. Highest-leverage tractable PROTOTYPE TARGET (this round)

**Target: the COST-MODEL extension distinguishing reuse-vs-reprogram phases (§2),
default-off byte-identical.** Concretely, round-1 ships:

1. **Capacity-estimate phase classification** (`estimate.py::_scheduled_estimate`): add
   `reprogram_phase_count` (N) + `reuse_phase_count` (M) to `CapacityEstimate`, computed
   from the per-segment **same-`weight_bank_id` grouping** (the boundary in §1.2).
   Default/non-scheduled ⇒ unchanged single `phase_count`.
2. **The GAP-R cost term** (`cost_extraction.py`): add `reprogram_passes`, `reuse_passes`,
   `params_reloaded`, `activation_bytes_moved` to `CostRecord` (bump
   `COST_RECORD_FORMAT_VERSION` 2→3), and the `mj` term
   `base + N·mj_per_reprogram + (M+N−1)·mj_per_sync` with chip coefficients defaulting to
   0.0. `extract_cost_record` reads the phase split from the estimate and the activation
   bytes from the hybrid mapping's `input_map`/`output_map` (reusing the
   `_build_consumer_counts` walk).
3. **The reserved capability bit** `allow_weight_reuse` on `ChipCapabilities`
   (`from_platform_constraints` + `MappingStrategy` property), default False, NOT in
   `permission_kwargs` — the exact `allow_per_layer_s` reserved-gate pattern.

**Rationale:**

- It is the **one change that makes a reuse phase cheaper than a reprogram phase** — today
  they cost identically (both 0, GAP-R), so **no build-side optimization can be valued
  until the cost model can tell them apart**. The cost model is the keystone the user's
  whole insight rests on; everything else (residency, replica-fill build) is only worth
  building once the cost says it pays.
- It is **small, localized, additive, and byte-identical default-off**: it adds reporting
  fields + a 0-coefficient cost term + a reserved bit, modifies **no** sim behavior, no
  packing, no build. It reuses the **already-shipped** scheduling-aware estimate
  (`phase_count`, `peak_phase_cores`) and the existing `_build_consumer_counts` slice
  walk.
- It is **the diagnostic every downstream consumer reads**: the Pareto scatter (R3), the
  coverage ledger, and the eventual replica-fill build all need the (N reprogram, M reuse,
  params_reloaded, act_bytes) numbers this produces. It also makes the **pruning
  intersection (§4) measurable for free** — pruning's effect on N/M/bytes appears the
  moment the term exists.

**Why not the alternatives this round (honest multi-round feasibility):**

- **The replica-fill physical build** (`_build_scheduled` builds one resident-bank pool +
  streams positions instead of a fresh pool per pass) — this is the change that makes the
  reuse REAL on hardware, but it is **gated on the cost model**: without a cost that
  rewards reuse, there is no metric to validate the build against, and it is a deeper,
  parity-critical change to the packer + sim residency. **Round 2.**
- **Cross-pass weight residency in the sim** (`_evict_segment_cache` keeps banks resident
  across a reuse phase's sub-passes — `flow.py:135`, `stage_io.py` LRU keyed on
  `id(stage)`) — required for the sim to actually MODEL reuse (today it re-uploads), and a
  prerequisite for measuring the data-movement term against real runs. **Round 2**, paired
  with the build.
- **A VGG16@224 reuse-mode probe** confirming N≈16 / M≈142 against a real
  `_build_scheduled` + cost extraction — the empirical confirmation of §2.4. **Round 3**,
  gated on rounds 1-2.

**Multi-round honesty.** Round-1 is a **reporting + cost + reserved-gate** change: it makes
the reuse savings **visible and Pareto-rankable** but does **not** yet change a single
deployed number (the chip still reprograms every pass — the cost term just stops pretending
those passes are free and stops mis-charging reuse passes as reloads once coefficients are
declared). Rounds 2-3 turn the visible savings into **real** residency on the build/sim
side. This sequencing is deliberate: ship the **measurement** first (cheap, byte-identical,
unblocks the Pareto), then the **mechanism** (deep, parity-critical), then the **probe**.

---

## Key file index (all read-grounded; no production code modified to write this doc)

| concern | file:seam |
|---|---|
| reuse relationship recorded (1 bank / conv, N positions share it) | `src/mimarsinan/mapping/mappers/conv2d_mapper.py::_map_to_ir` (221-275: 1 `register_weight_bank`, per-pos `add_shared_neural_core(weight_bank_id=bank_id)`) |
| shared-bank back-reference | `src/mimarsinan/mapping/ir/types.py:91` (`NeuralCore.weight_bank_id`), `:14` (`WeightBank`) |
| GAP-R: cost model has NO reprogram/data-movement term | `src/mimarsinan/chip_simulation/cost_extraction.py` (`CostRecord`, `extract_cost_record`; energy ∝ Σ neurons·S only) |
| cost-record format version (bump 2→3) | `cost_extraction.py:59` (`COST_RECORD_FORMAT_VERSION`); unknown-field reject `from_dict` (204-210) |
| scheduling-aware capacity estimate (phase classification slot) | `src/mimarsinan/mapping/verification/capacity/estimate.py::_scheduled_estimate` (259-299); `CapacityEstimate` (72-97) |
| same-bank / coalescing-bundle grouping (reuse boundary) | `src/mimarsinan/mapping/support/schedule/schedule_budget.py::_coalescing_bundles` (51); `schedule_split.py::split_softcores_by_capacity` (latency-group + bundle grouping) |
| fresh-pool-per-pass = "reprogram every pass" build | `src/mimarsinan/mapping/packing/hybrid_build_scheduled.py::_build_scheduled` (80, `_make_available_hardware_cores`) |
| sync barrier I/O slices (data-movement quantity) | `src/mimarsinan/mapping/packing/hybrid_types.py::SegmentIOSlice` / `HybridStage.input_map` (23-43) |
| existing slice walk to reuse for act-bytes | `src/mimarsinan/models/spiking/hybrid/flow.py::_build_consumer_counts` (94-125) |
| in-stage bank reuse (exists) + cross-pass eviction (to fix in R2) | `src/mimarsinan/mapping/packing/stage_io.py::_get_segment_tensors` (162, `_ensure_bank_tensor`); `flow.py::_evict_segment_cache` (135) |
| sync-barrier count | `src/mimarsinan/mapping/verification/layout_verification_scheduling.py::compute_schedule_sync_count` |
| reserved-gate pattern to copy (`allow_weight_reuse`) | `src/mimarsinan/mapping/platform/mapping_structure.py::ChipCapabilities` (88-144, `allow_per_layer_s`); `MappingStrategy` property (200-211) |
| SCM-gate phase reporting (extend to N+M) | `src/mimarsinan/pipelining/pipeline_steps/mapping/soft_core_mapping_step.py::_run_capacity_gate` (245-278) |
| pruning intersection inputs | `estimate.py::_segment_lower_bound` (`frags`/`groups` from `get_input_count`/`get_output_count`, 141-143) |
| suggester pass-count heuristic (NOT a deployment cost) | `src/mimarsinan/mapping/verification/suggester/hw_config_suggester_scheduled.py` (`cost = core_area·passes^latency_weight`) |
| ground truth | `docs/research/E4_PLACEMENT_SCALING_DESIGN.md` (§2 GAP-R, §3, §5), `docs/research/PROGRAM_PLAN_v2.md` (E4 round 1) |
