# D3 — End-to-end Scheduled-Build probe (weight-reuse phases REALIZED, bit-exact)

**Question.** Does the Scheduled path actually *realize* weight-reuse phases — a
real schedule of reprogram + reuse passes that BUILDS and runs bit-exact — or does
only the *capacity ESTIMATE* know about scheduling while the build/sim are untested
at the overflow regime?

**Status: CONFIRMED-BY-MECHANISM.** On a small but GENUINE conv graph that really
overflows a tight core budget, (a) the estimate reports a multi-phase scheduled
verdict, (b) the Scheduled BUILDER realizes exactly that phase structure AND the
weight-reuse classifier decomposes the same IR into N reprogram + M reuse passes,
and (c) the scheduled-built deployment is **bit-exact** (float64 `atol=0`) to both
the non-scheduled single-pool reference and the torch neuromorphic forward.

Test: `tests/integration/test_scheduled_build_probe.py` (5 tests, all PASS).

---

## 1. The small genuine case

`deep_cnn` `depth=4, width=6, base_activation=ReLU`, input `(1, 8, 8)`, `T=4`, LIF
spiking mode. Hard core `256×256` (wide enough that no single fan-in/width is the
constraint), budget `count = 6` (deliberately tight).

Why this case is genuine, not contrived:

- It maps **fully on-chip** (plain Conv-BN-ReLU stack; `deep_cnn` is the project's
  designated valid trainable-deep vehicle — see `VALIDITY_AUDIT.md`).
- Its two same-shape conv stages **share weight banks**, so the weight-reuse
  classifier sees *real* reprogram-vs-reuse passes (not an all-owned FC graph where
  reuse would be 0).
- It **genuinely overflows** the 6-core budget as a single pool, so scheduling
  actually triggers (not a model that fits anyway).

Measured IR (36 neural cores, 3 weight banks):

| bank | #position cores | role |
|---|---|---|
| 0 | 16 | conv stage (shared across 16 spatial positions) |
| 1 | 16 | conv stage (shared across 16 spatial positions) |
| 2 | 4  | conv stage (shared across 4 positions) |

0 owned (non-shared) cores. Two neural segments (a host op — pooling/flatten —
splits the graph): `features_10` (32 cores) and `head_pool` (4 cores).

---

## 2. (a) ESTIMATE — scheduling triggers on a genuine overflow

`estimate_cores_needed(ir, …, allow_scheduling=True)`:

| field | value |
|---|---|
| `scheduled` | `True` |
| `feasible` | `True` |
| `cores_needed` (single-pool SUM) | **13** |
| `cores_available` (budget) | 6 |
| `peak_phase_cores` | **6** (≤ budget ✓) |
| `phase_count` | **3** (> 1 ✓) |
| per-segment bounds | `features_10: 11`, `head_pool: 2` |
| `ceil(bound/budget)` per segment | `features_10: 2`, `head_pool: 1` → Σ = **3** |

Same IR/budget **without** scheduling: `feasible=False`, `cores_needed=13` (the SUM
overflows the 6-core pool). So the scheduled path is the *only* way this maps —
scheduling is genuinely needed, not cosmetic.

---

## 3. (b) BUILD vs ESTIMATE structure — the realized schedule matches

The Scheduled BUILDER (`build_hybrid_hard_core_mapping`, `allow_scheduling=True`)
realizes **exactly 3 neural stages**, matching the estimate's `phase_count`:

| realized neural stage | hard cores | ≤ budget |
|---|---|---|
| `features_10_cap0` | 6 | ✓ |
| `features_10_cap1` | 6 | ✓ |
| `head_pool`        | 2 | ✓ |

- Stage count `3` == estimate `phase_count` `3` == Σ `ceil(segment_bound/budget)`.
- Each stage's hard-core occupancy ≤ 6; `max == 6 ==` estimate `peak_phase_cores`.
- The non-scheduled single-pool build (generous budget) emits **2** neural stages
  (one per segment); the scheduled build emits **3** — the extra one is the
  capacity-split of the over-budget `features_10` segment into two reprogramming
  passes. (Bit-exactness, §4, holds despite the extra stage.)

**Weight-reuse phase decomposition** (`weight_reuse_plan_from_graph`, a pure read of
the SAME IR):

```
3 reprogram + 33 reuse phases (36 total, 91.7% reused; 4584 params reloaded)
```

| quantity | value | derivation |
|---|---|---|
| `reprogram_passes` (N) | **3** | = #distinct shared banks (3) + #owned cores (0) |
| `reuse_passes` (M)     | **33** | = positions reusing a resident bank = 36 − 3 |
| `total_passes`         | **36** | == #IR neural cores ✓ |
| `reuse_fraction`       | 0.917 | most passes reuse the resident conv kernel |
| `sync_barrier_count`   | 35 | = total_passes − 1 (the design's M + N − 1) |

Note the **two distinct decompositions** of the same schedule, both asserted by the
probe and both matching the real graph:

- **Capacity phases** (`estimate.phase_count = 3`): time-multiplex of softcores
  across reprogramming passes so the PEAK fits the budget — `Σ ceil(seg_bound/budget)`.
- **Weight-reuse phases** (`plan`: 3 reprogram + 33 reuse): which passes reload a
  distinct bank vs reuse a resident one — the conv-position residency the cost model
  charges.

They are not the same number and are not meant to be; the probe locks the structure
of each (capacity stages == phase_count, each ≤ budget; reuse total == #cores,
reprogram == #distinct banks) so a regression in *either* the estimate, the builder,
or the classifier trips a loud, diagnosable failure.

---

## 4. (c) BIT-EXACT — scheduled build is value-identical where both fit

`T=4`, 6 random samples, LIF, float64:

| comparison | max\|Δ\| |
|---|---|
| scheduled build vs non-scheduled single-pool reference | **0.0** |
| scheduled build vs torch neuromorphic forward (NF) | **0.0** |
| non-scheduled reference vs torch NF (control) | **0.0** |

Scheduling time-multiplexes the SAME softcores onto a fresh per-phase core pool; it
does not perturb a single deployed value. This reuses the existing torch↔sim
fidelity primitives (`chip_aligned_segment_forward` for the NF; `SpikingHybridCoreFlow`
for the deployed sim) — the same lock style as `tests/integration/test_torch_sim_fidelity.py`.

---

## 5. The roadmap's "VGG@224 → 16 reprogram + 142 reuse" claim

**Verdict: CONFIRMED-BY-MECHANISM, NOT by a heavy VGG@224 run.**

What this probe *does* establish on a real graph:

- The weight-reuse classifier correctly collapses shared-bank conv positions to
  `1 reprogram + (positions − 1) reuse` per bank — exactly the per-conv collapse the
  VGG number rests on (here: bank 0 → 1 reprogram + 15 reuse, etc.).
- `reprogram_passes` == #distinct weight banks (here 3), the same rule that yields
  "13 conv + 3 FC = 16" for VGG16@224.
- The Scheduled BUILDER realizes the estimated phase structure and the result is
  bit-exact — so the schedule is not just an estimate, it builds and runs.

What this probe **does NOT** do (stated honestly):

- It does **not** build VGG16@224 (a heavy ~40s mapping path; the existing
  `test_vgg16_imagenet_is_feasible_via_scheduling_on_realistic_chip` already covers
  the VGG *estimate* and is marked `slow`). The "16 reprogram + 142 reuse" figure
  for VGG@224 is an **analytical per-conv-block projection**
  (`docs/research/GAPR_P2_DEFENSIBLE_COST_MODEL.md`, `WEIGHT_REUSE_SCHEDULING_DESIGN.md`),
  not a number produced by this probe.
- **Integrity note on the "142":** the design docs carry two different VGG reuse
  counts depending on the counting unit — `~142` when reuse is aggregated *per conv
  block* (M = total_passes − 16 over a per-block position model) and `137,775` when
  counted *per spatial position* (`WEIGHT_REUSE_SCHEDULING_DESIGN.md`). Both come
  from the SAME classifier rule (`reuse = total − distinct_banks`); they differ only
  in how `total_passes` is enumerated. The mechanism is what this probe confirms; the
  specific VGG count remains the cost model's stated projection, with its own
  uncertainty band, and would need the heavy VGG@224 mapping run to materialize
  end-to-end.

So: the *mechanism* (classifier + scheduled builder, bit-exact on a real
weight-bank-sharing conv graph) is **confirmed**; the *specific VGG@224 number* is a
defensible projection that still needs a heavy run to realize, not a result of this
probe.

---

## 6. Reproduce

```
PYTHONPATH=<repo>/src:<repo>/../spikingjelly <env>/bin/python -m pytest \
  tests/integration/test_scheduled_build_probe.py -q
```

5 tests: (a) estimate triggers scheduling; (b) build realizes the estimate's phase
structure + reuse-plan structure matches the graph; (c) bit-exact vs reference and
torch + the scheduled path emits strictly more stages than single-pool.
