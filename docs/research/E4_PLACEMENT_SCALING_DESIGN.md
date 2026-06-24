# E4 — Scheduled-Path Placement Scaling (no weight sharing)

**Question (corrected, per `PROGRAM_PLAN_v2.md` §"E4 direction CORRECTED").** Weight
sharing is **NOT** possible on the chip. Each conv output position × channel needs its
own physical weight placement (softcore), so a 224²-spatial conv genuinely produces
hundreds of thousands of softcores — that is **intrinsic, not a mapper bug**. The lever
for "cores too few / too small" is the existing **Scheduled mapping path**
(`allow_scheduling` / `MappingStrategy`): the chip is **re-programmed across phases /
sync barriers**, reusing physical cores across phases. So the capacity question is
reframed:

> Not "do 413K cores fit at once" (they don't) — but **under scheduling, does the PEAK
> phase fit the physical chip, and at what PHASE-COUNT + reprogramming cost?**

This doc decides: (1) irreducible-vs-overproduced; (2) the scheduled-path model +
VGG16@224 feasibility numbers; (3) a scheduling-aware capacity model; (4) the GAP-1
reassembly fix; (5) the highest-leverage tractable prototype for this round. **No
production code was modified to write this doc** — every number is read off the shipped
code seams cited inline.

---

## 1. Is the 413–465K count IRREDUCIBLE or OVER-PRODUCED?

**Verdict: ~413–465K is NOT the irreducible no-weight-sharing count. The TRUE
irreducible spatial-unroll floor for VGG16@224 conv layers is ~137,800 softcores. The
~3× overshoot is **100% the coalescing-fragment-as-separate-core accounting** against a
too-narrow 256-axon budget — NOT spatial unroll, NOT neuron-splitting, NOT duplicate
cores.**

### Where the number comes from (the conv → softcore lowering)

- `torch_mapping/converter_handlers/conv_mixin.py::_convert_conv2d` builds a
  `Conv2DPerceptronMapper` with **no** `max_neurons` (defaults `None`).
- `mapping/mappers/conv2d_mapper.py::_map_to_ir` (lines 216–279): with
  `max_neurons=None`, `group_sizes = [out_channels]` (ONE group), and the loop
  `for pos in range(h_out * w_out)` calls `add_shared_neural_core` **exactly once per
  spatial output position** — emitting **one IR softcore per position**, with
  `patch_fanin = in_channels·k·k` axons and `out_channels` neurons. This is the
  irreducible spatial unroll and it **is correct** given no weight sharing
  (`add_shared_neural_core` references a single registered `weight_bank_id` — the bank is
  shared; the *placement* is not, which is exactly the no-weight-sharing reality).
- The ~413–465K explosion is produced **LATER**, in the capacity estimate
  (`mapping/verification/capacity/estimate.py::_segment_lower_bound`): for each IR core it
  computes `frags = coalescing_fragment_count(in_count, max_axons) = ceil(fanin/256)` and
  `groups = ceil(out_channels/max_neurons)`, and the per-core lower bound includes
  `frags·groups`. The whole-net summed bound is dominated by `Σ positions · frags`.

### The over-production is the fan-in/256 fragment multiplier (`frags`), nothing else

| layer fan-in | `frags = ceil(fanin/256)` | role |
|---|---|---|
| 27 (conv0) | 1 | floor |
| 576 (conv with 64ch 3×3) | 3 | ×3 |
| 1152 | 5 | ×5 |
| 2304 | 9 | ×9 |
| 4608 (conv8–13) | **18** | ×18 |

- **Irreducible** (1 softcore / output position, `max_neurons=256`): **137,788 cores**.
  `groups = 1` for **all 13** conv layers (every `out_channels ≤ 256 ≤ max_neurons`), so
  neuron-splitting contributes **ZERO** overhead at `max_neurons=256`.
- **Capacity-estimate count** (`Σ positions · frags · groups`, `max_axons=256`):
  ~459–465K (symmetric reconstruction 464,912; the E3 probe measured 416,560 using its
  slightly-smaller asymmetric W=111 pooling geometry → fewer late-layer positions).
  Either way **~3.0× the irreducible floor**.
- The first overflowing segment `features_6` = **50,176 softcores of (576 axons, 64
  neurons)**. That 50,176 is **pure spatial unroll → irreducible & correct**. Its blow-up
  to 112,896 cores in the estimate is `50,176 · ceil(576/256)=3` — the `frags` multiplier.

### Is `frags` truly irreducible? — Partially, and it is double-counted as the overflow driver

`coalescing_fragment_count` splits a wide fan-in (> `max_axons`) across multiple physical
crossbars whose partial sums are membrane-transferred (`allow_coalescing`). On a 256-axon
chip that IS a real per-position cost. BUT:

1. It is a function of the **tiny 256-axon budget**, not of VGG — widen the crossbar and
   it collapses. The E3 CIFAR probe used **2048-axon cores** giving only **940 hard
   cores** total (`frags=1` everywhere); on a 4608+-axon crossbar VGG16@224 returns to the
   ~138K spatial floor.
2. Coalescing **FUSES** the N fragments back into ONE logical wide core
   (`mapping_structure.py` docstring: "one full-width core that the packer fuses from N
   hard cores"). Whether those N fragments are "N cores" or "1 wide core" is a **chip-grid
   choice**, not an intrinsic model property.

So the 3× is an artifact of **(256-axon budget) × (fragments-as-cores accounting)**, NOT
model-intrinsic. **No redundant/duplicate cores exist beyond this**: the lowering is
vectorized (single shared weight bank), `_chunk_sizes` only chunks output channels (never
triggers at 256), and there is no suboptimal-tiling waste.

### Neuron-split (`groups`) is NOT the problem at 256

`groups > 1` only appears if `max_neurons < out_channels`. At `max_neurons=256` every VGG
conv has `groups=1`. At `max_neurons=64` (the probe's 64-neuron `features_6` softcores)
`groups` inflates the irreducible floor to ~211,680 and the estimate to ~1.06M — but
`features_6`'s first conv has `oc=64` so `groups=1` there too, confirming the 50,176
first-segment figure is pure (correct) spatial unroll.

**Bottom line:** irreducible spatial-unroll floor = **~137,800 softcores** (intrinsic,
correct). The ~413–465K is **~3.0× over** the floor and is **entirely** the
coalescing-fragment-as-separate-core accounting against a 256-axon budget — reducible by
widening `max_axons` (`frags→1`) or by counting a coalesced group as one logical core,
**NOT** by any change to the spatial unroll.

---

## 2. The SCHEDULED-PATH model — how time-multiplexing deploys a > chip-size model

### How it works end-to-end (functionally complete in the tree today)

1. **Dispatch.** `mapping/packing/hybrid_build_pool.py::build_hybrid_hard_core_mapping`
   reads a resolved `MappingStrategy`; when `strategy.allow_scheduling` it calls
   `_build_scheduled` (`hybrid_build_scheduled.py:80`) instead of `_build_single_pool`.
   `allow_scheduling` is a `ChipCapability` resolved from `platform_constraints`
   (`mapping/platform/mapping_structure.py`), default **False** (`config_schema/defaults.py`).

2. **Reprogramming model = FRESH CORE POOL PER PASS.** `_build_scheduled` partitions the IR
   (`partition_ir_graph`: neural segments split by host `ComputeOp` barriers) and for each
   neural segment calls `_flush_scheduled_subsegments` → `_split_segment_by_capacity` →
   `split_softcores_by_capacity`. Each resulting sub-segment is flushed by
   `_flush_scheduled_segment` (`hybrid_segment.py`), which calls
   `_make_available_hardware_cores(cores_config)` to build a **brand-new hardware pool for
   that pass**. That fresh pool per sub-segment **is** the chip-reprogramming model: the
   same physical cores are reused pass-to-pass; only the **peak pass** must fit.

3. **Splitting ONE giant conv across phases.**
   `mapping/support/schedule/schedule_split.py::split_softcores_by_capacity` groups
   softcores by `latency_tag`, greedily accumulates a latency group into a running pass,
   calls `pack_layout` after each addition, and closes the pass at the first overflow.
   CRUCIAL: a 224²-conv is ONE latency group (all same DAG depth) of ~50K softcores that
   cannot pack at once — so its **within-group binary-halving fallback**
   (`_halve_until_packs`) recursively halves the group until each piece packs, splitting a
   single conv layer across many phases. Cores in one latency group have **no inter-core
   deps**, so any partition is functionally equivalent.

4. **Sync barrier + simulator.** Adjacent `HybridStage(kind="neural")` entries (each its own
   `HardCoreMapping`, stamped `schedule_segment_index` / `schedule_pass_index`) hand off
   through the `state_buffer` at segment-level rates — **no synthetic ComputeOp inserted**.
   `models/spiking/hybrid/flow.py::SpikingHybridCoreFlow.forward` iterates stages
   sequentially through the buffer. N passes ⇒ N−1 sync barriers
   (`layout_verification_scheduling.py::compute_schedule_sync_count`). Scheduled stages
   carry through HCM / nevresim / SANA-FE / Lava records.

5. **Verifier already reframes budget as PEAK not SUM.**
   `verifier/mapping_verifier_hw.py::verify_hardware_config(..., allow_scheduling=True)`:
   when single-pass packing fails it partitions per segment via
   `estimate_passes_for_layout_validated`, reports `total_passes`,
   `max_cores_per_pass = effective_core_budget` (the 0.8× heterogeneous discount),
   `schedule_sync_count`, and sets `feasible=True`. Its `est_min` hint is
   `max(per_sc_costs)` under scheduling vs `sum(per_sc_costs)` without (lines 221–224). So
   the **verifier/wizard path is already scheduling-aware**.

### VGG16@224 feasibility under scheduling — the numbers

On a realistic **256×256×2048** chip (2048 cores, 256 axons, coalescing on):

- Each (576-axon, 64-neuron) softcore needs `ceil(576/256)=3` hard cores ⇒ **682
  softcores/phase** fit 2048 cores (2046 used).
- `features_6` alone (50,176 softcores): `ceil(50,176/682) = 74 phases`, peak-phase = 2046
  cores — **FITS**.
- Whole VGG16@224: phase-count ≈ `ceil(315,816 / 2048) = 155 phases`, peak-phase = 2048
  cores — **FITS**. (315,816 = the E3-headline whole-net static SUM bound, coalescing on,
  256 axons; worst single segment `features_6` axon-bound = `50,176·576/256 = 112,896`.)

**Empirical confirmation the splitter time-multiplexes one latency group** (probe, from the
synthesis): 5000 softcores of (576-axon, 64-neuron) on a 256×256×64 chip →
`split_softcores_by_capacity` returned **239 sub-segments**, peak **21 softcores/phase**
(~63 coalesced cores, fits 64); `verify_hardware_config(allow_scheduling=True)` →
`feasible=True, total_passes=239, max_cores_per_pass=64`. **Mechanism proven.**

### Is 224²-conv deployable at all? — HONEST verdict

**YES on a realistic chip, time-multiplexed — but at a phase-count that is large and whose
reprogramming cost is currently UNMODELED.** On 256×256×2048, VGG16@224 maps in ~155
phases with the peak phase fitting; on the 1000-core/256-axon "realistic ImageNet budget"
the E3 probe used, it is ~155–416 phases depending on coalescing width. The peak-phase
constraint is satisfiable; the open honesty is **the phase-count × per-phase
reprogramming-latency/energy product**, which no cost term charges (see GAP-R below). So
the correct framing for the coverage ledger is **"feasible-via-scheduling, N≈155 phases,
reprogramming cost UNMODELED"** — not "infeasible," and not "free."

### Two gaps in the scheduled path

- **The real bug (a static SUM gate that never branches on scheduling) — see §3.** The
  build/verify machinery above CAN time-multiplex VGG16@224, but
  `estimate_cores_needed` SUMS per-segment bounds against a single summed budget and raises
  `CapacityExceededError` **before** the scheduled builder ever runs. With the capacity
  gate on, `allow_scheduling=True` does **not** map VGG16@224 today.

- **GAP-R: reprogramming cost is ABSENT.** There is NO reprogramming
  latency/energy term anywhere (`grep -i reprogram|reconfig|reload` over the cost path
  finds nothing). `chip_simulation/cost_extraction.py` models `energy ∝ Σ_d neurons_d·S_d`
  (soma-dominated) and `latency_steps = Σ_d timesteps_executed_d` with **zero** inter-pass
  reprogramming penalty; `cores` is per-segment count, not peak-phase. The only place
  pass-count enters a cost is the config SUGGESTER
  (`suggester/hw_config_suggester_scheduled.py`: `cost = core_area · passes^latency_weight`)
  and the verifier's `schedule_sync_count` — **neither charges weight-reload time/energy**.
  A 155-phase VGG16@224 schedule is modeled as if reprogramming were free.

---

## 3. A SCHEDULING-AWARE capacity model (extend the merged E4r1 estimate)

**The fix is small, localized, and unblocks an already-tested scheduled build/verify
path.** Today `estimate_cores_needed(ir_graph, platform_constraints)` takes **no**
`allow_scheduling` parameter and computes `cumulative += bound` (SUM) against one summed
budget, raising when the SUM overflows. This is **non-scheduling-aware** and is the static
gate that rejects VGG16@224 before the scheduled builder runs. It is invoked at TWO
blocking call-sites:

- (a) `pipeline_steps/mapping/soft_core_mapping_step.py::_run_capacity_gate`
  (`capacity_gate` default True).
- (b) the GPU scheduler `capacity_precheck` (per `tests/unit/gpu/test_scheduler_capacity_gate.py`
  — rejects `torch_vgg16` with `cores_needed=315816 > 1000`, names
  `overflowing_segment=neural_segment_until:features_6`, no GPU claimed).

### Proposed extension (design surface)

Add `allow_scheduling: bool = False` to `estimate_cores_needed`, and have feasibility +
reporting branch on it. The peak-phase is bounded **below** by the largest single segment's
lower bound and bounded **above** by per-segment ceilings; the sound, cheap model is:

- **Without scheduling (unchanged):** `feasible = (Σ_segment bound ≤ budget)`;
  `cores_needed = Σ bound`. Byte-identical to today.
- **With scheduling:** `peak_phase = max_segment bound` (each segment reprograms a fresh
  pool; a single segment that cannot be split below `budget` is the true peak — but since
  one latency group splits freely via `_halve_until_packs`, the achievable peak is
  `min(max_segment_bound, budget)` whenever the segment's atomic unit — one coalescing
  bundle — fits). `feasible_via_scheduling = (max_atomic_unit_cost ≤ budget)`;
  `phase_count = Σ_segment ceil(segment_bound / budget)`;
  `reprogramming_passes = phase_count` (the cost the new term in GAP-R consumes).

Extend `CapacityEstimate` with `scheduling_aware: bool`, `peak_phase_cores: int`,
`phase_count: int`, and report VGG16@224 as **"feasible-via-scheduling, ~155 phases, peak
2048 cores"** instead of "infeasible, needs 315,816." Pass `allow_scheduling` through both
call-sites (resolved from the same `MappingStrategy` ChipCapability the builder uses), so
the enqueue pre-check and the SCM gate stop rejecting schedulable nets.

**The atomic-unit feasibility check is the only hard gate under scheduling:** a single
coalescing bundle (or a single non-splittable softcore's `frags·groups`) must fit the
budget. For VGG16@224 the largest atomic unit is `frags=18` (4608-fanin conv) ⇒ 18 cores ≪
1000-budget, so **every VGG conv is schedulable**; only a model whose single coalesced
softcore exceeds the whole chip is truly infeasible.

This reuses the existing `_segment_lower_bound` (no new packing logic), keeps the SOUND
lower-bound property, and reframes the verdict — the scheduled build/verify path it unblocks
(`_build_scheduled`, `verify_hardware_config(allow_scheduling=True)`) **already exists and
is tested**.

---

## 4. The GAP-1 reassembly fix

**Symptom (E3, CIFAR-VGG):** per-neuron LIF `k==k` reassembly mis-attributes **1276/65536**
(~2%) neurons of conv-perceptron 1, **while the per-neuron total is identical (654==654) and
`out_max_abs==0.0`**. Spikes are conserved and the decode is value-exact; only *which
physical hard-core neuron maps back to which logical neuron* is scrambled.

**Root seam:** `tests/integration/_split_reassembly.py::hcm_per_perceptron_counts` →
`_reassemble`. It inverts the packing by: filtering `coalescing_role ∈ {None, master,
accum}` and `psum_role ∈ {None, accum}` (lines 49–51), sorting neuron-split fragments by
`neuron_range_in_original[0]` (line 57), and concatenating per IR core in IR-id order. When
**many fused + split fragments of one wide IR core interleave** at VGG scale, this
orig-offset sort / role filter no longer produces the correct 1:1 neuron map. The
bookkeeping is sound at small fan-out (the harness PASSES at `wide_dim=64`) but does not
survive deep interleaving.

**Fix direction (test-first, no simulator-dynamics change — this is pure bookkeeping over
records the sim already emits):**

1. **Carry an explicit per-fragment identity tuple** `(ir_core_id, neuron_range_in_original)`
   on each placement record and reassemble by **grouping on `ir_core_id` then ordering by
   `neuron_range_in_original[0]` WITHIN that group** — instead of a single global orig-offset
   sort that conflates fragments across interleaved IR cores. The packing already stamps
   `neuron_range_in_original`; the fix is to make the reassembler key on `(ir_core_id, range)`
   jointly, not on `range` alone after a coalescing/psum-role filter that can admit
   same-`range` fragments from different IR cores.
2. **Replace the role-filter heuristic with the authoritative master/accum chain.** The psum
   partial-sum chain has a designated master fragment per logical neuron; reassemble the
   value at the master (the `coalescing.py` fusion already defines "one full-width core the
   packer fuses from N hard cores"), rather than admitting `{None, master, accum}` and hoping
   the sort disambiguates.
3. **Add a scale regression test** at VGG-CIFAR fan-out (wide conv, fuse + split) asserting
   per-neuron `k==k` (not just per-neuron totals). This is the attributability keystone for
   LIF and currently has **no fallback** instrument — so the regression test is the
   load-bearing artifact.

**Scope:** GAP-1 blocks per-neuron *attributability* (fault localization, neuron-level
energy/provenance) on coalesced+split conv nets; it does **NOT** block the value-domain
deployment claim (`out_max_abs==0.0` holds). It is a bookkeeping bug in the test-side
reassembler, not in the deployed sim.

---

## 5. Highest-leverage tractable PROTOTYPE TARGET (this round)

**Target: the scheduling-aware capacity diagnostic (§3).** Rationale:

- It is the **one change that flips VGG16@224 from "infeasible" to "feasible-via-scheduling,
  N phases"** — directly serving the corrected E4 mandate (the Scheduled path is the lever,
  not weight sharing). It un-blocks the enqueue pre-check and SCM gate that currently reject
  every ImageNet-spatial conv net before the (already-built, already-tested) scheduled
  builder runs.
- It is **small and localized** (add `allow_scheduling` to `estimate_cores_needed` + branch
  feasibility on PEAK vs SUM + thread the flag through the two call-sites), reuses the
  existing `_segment_lower_bound` (no new packing), and preserves the SOUND lower-bound
  property — byte-identical when `allow_scheduling=False`.
- It is the **diagnostic that everything else reads**: the coverage ledger (E1), the GPU
  scheduler precheck, and the cost/reprogramming model (GAP-R) all need the PEAK-phase +
  phase-count numbers this produces.

**Why not the alternatives this round:**

- **GAP-1 bit-exact fix** — high value but narrower: it restores per-neuron *attributability*,
  not deployability; the value-domain claim is unaffected. Sequence it **second** (it is a
  contained test-side bookkeeping fix with a clear regression test).
- **A Scheduled-path mapping probe on VGG16@224** — valuable validation, but it is **gated on
  §3**: the static SUM gate rejects the net before `_build_scheduled` runs, so the probe needs
  the scheduling-aware estimate (or `capacity_gate=False`) to even start. Run it **third**, as
  the empirical confirmation that §3's `phase_count≈155, peak≤2048` prediction matches a real
  `verify_hardware_config(allow_scheduling=True)` + `_build_scheduled` pass.

**Recommended round-1 deliverable:** the scheduling-aware `estimate_cores_needed`
(tests-first: SUM-vs-PEAK feasibility, `phase_count`, atomic-unit hard-gate, VGG16@224 reads
"feasible-via-scheduling ~155 phases"), then (round 2) the GAP-1 reassembly fix + regression
test, then (round 3) the VGG16@224 scheduled-build probe + the GAP-R reprogramming cost term
on `cost_extraction` (`reprogram_passes`, `mj_per_reprogram`).

---

## Key file index (all read-grounded; no production code modified to write this doc)

| concern | file:seam |
|---|---|
| conv → 1 softcore/position lowering | `src/mimarsinan/mapping/mappers/conv2d_mapper.py::_map_to_ir` (216–279) |
| conv mapper construction (no max_neurons) | `src/mimarsinan/torch_mapping/converter_handlers/conv_mixin.py::_convert_conv2d` |
| `frags` over-production source | `src/mimarsinan/mapping/verification/capacity/estimate.py::_segment_lower_bound` |
| coalescing fragment count | `src/mimarsinan/mapping/platform/coalescing.py::coalescing_fragment_count` |
| coalescing fuses N→1 logical core | `src/mimarsinan/mapping/platform/mapping_structure.py` |
| static SUM gate (the real bug) | `estimate.py::estimate_cores_needed` (sums; no `allow_scheduling`) |
| SCM-step capacity gate call-site | `src/mimarsinan/pipelining/pipeline_steps/mapping/soft_core_mapping_step.py::_run_capacity_gate` (245) |
| GPU scheduler precheck call-site | `tests/unit/gpu/test_scheduler_capacity_gate.py::capacity_precheck` |
| scheduled build (fresh pool/pass) | `src/mimarsinan/mapping/packing/hybrid_build_scheduled.py::_build_scheduled` |
| one-latency-group splitter | `src/mimarsinan/mapping/support/schedule/schedule_split.py::split_softcores_by_capacity` (+`_halve_until_packs`) |
| scheduling-aware verifier (PEAK) | `src/mimarsinan/mapping/verification/verifier/mapping_verifier_hw.py::verify_hardware_config` (allow_scheduling) |
| sync-barrier count | `src/mimarsinan/mapping/verification/layout_verification_scheduling.py::compute_schedule_sync_count` |
| scheduled sim forward | `src/mimarsinan/models/spiking/hybrid/flow.py::SpikingHybridCoreFlow.forward` |
| cost model (NO reprogram term) | `src/mimarsinan/chip_simulation/cost_extraction.py` (GAP-R) |
| suggester pass-count cost | `src/mimarsinan/mapping/verification/suggester/hw_config_suggester_scheduled.py` |
| GAP-1 reassembly seam | `tests/integration/_split_reassembly.py::hcm_per_perceptron_counts` → `_reassemble` |
| E3 measured numbers | `docs/research/E3_SCALE_PROBE.md` |
