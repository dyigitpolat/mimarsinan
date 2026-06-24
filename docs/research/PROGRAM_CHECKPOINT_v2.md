# Mimarsinan Program Checkpoint v2

**Genuine single-spike TTFS ANN→SNN deployment fidelity, and its scale-up frontier.**

This is the v2 program checkpoint. It supersedes the science narrative of
`PROGRAM_CHECKPOINT.md` (whose findings still stand) by folding in the
gate-v2 tiered-validity re-classification (`VALIDITY_AUDIT.md`), the live
measured coverage fraction (`scripts/campaign/coverage_report.py` over the
165-row `runs/campaign/ledger.jsonl`), and the E1–E8 engineering program
that carried the work from "can it map" to "what does it cost to map at
ImageNet scale." Every number below is from the science-valid ledger or a
named measured probe. Nothing here is invented or extrapolated.

---

## 1. Vision and where we stand

**The thesis.** A neuromorphic deployment is *transferable tuning on the
mappable surface*: an ANN is trained, then folded into a genuine spiking
chip mapping, and the only fidelity question that matters is how much
accuracy the *fold* costs — because the chip lowering itself is bit-exact.
We have hardened that last claim into a measured invariant: across every
valid vehicle, NF↔SCM per-neuron agreement is **0.0000%** mismatch and
torch↔sim parity is **1.0**. All loss lives in the trainable ANN→single-spike
TTFS conversion. This lets us attribute every accuracy point to the
conversion, not to a leaky simulator.

**Where the science landed.** Synchronized TTFS is a **lossless deep
default** — it holds the ANN ceiling at every measured depth and dataset on
a valid vehicle (worst case ≤3.06pp). Cascaded single-spike TTFS carries a
real **firing-gain death-cascade** governed by a *dual-axis law* (depth ×
task-hardness multiply). No firing-gain rescue knob works on a valid
convnet today, so synchronized is the unconditional recommendation. This is
a complete, honest, from-scratch diagnostic tier.

**Where validity landed.** Validity is now *tiered*, not binary
(gate-v2): a 20% on-chip floor and a 50% majority, checked on **both**
params and MACs. This re-instates `deep_mlp` d6/d8 as `VALID_FLAGGED`
evidence (its flag is a fixable placement gap) and promotes ViT-B to a
`VALID_FLAGGED` headline vehicle (its flag is a genuine research gap:
on-chip attention/LayerNorm). Genericity is reported as a **measured
fraction** with a named untested frontier, not a claim.

**Where the frontier is.** The named, measured frontier is ImageNet-scale
convolution. Weight *sharing* on the chip is impossible (each conv output
position × channel needs its own physical softcore), but weight *reuse* in
the time domain (load once, stream data; schedule = reuse-phases +
reprogram-phases) makes VGG16@224 **feasible-via-scheduling**. The frontier
moved from "can't map" to "maps at N phases — now cost the N." That cost
term is the next dollar of engineering.

---

## 2. Validated findings (the science)

All numbers below are from the science-valid ledger. Chip lowering is
bit-exact everywhere (NF↔SCM 0.0000%, torch↔sim 1.0), so every gap is a
conversion gap.

### 2.1 Synchronized TTFS is a lossless deep default (positive headline)

Synchronized `ttfs_cycle_based` (sequential latency groups, full S-step
window, grid-quantized input, NF==kernel per-neuron bit-exact) holds the ANN
ceiling at every measured depth/dataset on a valid vehicle. It is the
unconditional deep-model default.

| Vehicle / dataset | Result | Verdict |
|---|---|---|
| deep_cnn MNIST, every depth d4–d12 | ≤0.18pp; **d12 sync 0.9917 vs ANN 0.9887 = +0.30pp** (sd 0.07pp) | lossless |
| deep_cnn FMNIST | d4 1.86 → d6 2.78 → d8 2.98 → d10 3.06pp | MET throughout |
| deep_cnn KMNIST | d4 1.99 → d6 0.80 → d8 0.55 → d10 0.40pp | MET |
| lenet5 MNIST (full-test SCM) | sync 0.9891 vs ANN 0.9913 = **+0.22pp** | lossless |

Every sync cell on a valid vehicle is **≤3.06pp** of its ANN.

### 2.2 Cascaded death-cascade is a dual-axis law (depth × dataset)

Cascaded `ttfs_cycle` (genuine single-spike, latency-gated, fire-once,
ramp-reconstructed) carries a real firing-gain deficit. Synchronized is the
confound-free reference, so **casc→sync gap = pure firing-gain**.

**Depth axis** (deep_cnn MNIST, w16, S=4, clean rc=0) — a *sharp threshold*,
not a smooth widening:

| Depth | Cascaded | casc→sync | Note |
|---|---|---|---|
| d4 | 0.9883 | −0.15pp | tied |
| d5 | 0.9917 | +0.07pp | tied |
| d6 | 0.9383 | **+5.2pp** | SHARP ONSET |
| d8 | 0.9517 | +4.16pp | |
| d10 | 0.9517 | +4.00pp | bounded plateau |
| d12 | n=1 | — | inconclusive |

Verdict: **CONFIRMED-WITH-CONFOUND** — lossless ≤d5, a sustained ~4–5pp
plateau ≥d6. "Monotonically widening with depth" is **REFUTED on MNIST**
(the gap *shrinks* 5.2→4.16→4.00 — that's cascaded seed variance, not
growth). The old rc=1-confounded read inflated d10 to 13.86pp; the clean
rc=0 read is ~4pp.

**Dataset axis at fixed depth** — the MNIST no-collapse corner does **not**
generalize:

| Depth | MNIST | FMNIST | KMNIST |
|---|---|---|---|
| d4 casc→sync | −0.15 | +3.90 | +6.19 |
| d5 casc→sync | +0.07 | +6.03 | +4.62 |

**Depth × dataset compound** (deep_cnn FMNIST, monotone-widening, all clean
rc=0):

| Depth | casc→sync | casc→ANN |
|---|---|---|
| d4 | +3.90 | 5.76 |
| d6 | +6.11 | 9.11 |
| d8 | +11.34 | 14.28 |
| d10 | **+17.91** | **20.97** |

KMNIST widens d4 +6.19 → d8 +7.19 → d10 +15.98pp.

**Worst case = deep × hard.** deep_cnn d10 FMNIST casc 0.7250 vs ANN 0.9347 =
**20.97pp**; d10 KMNIST casc 0.8025 vs ANN 0.9663 = **16.38pp**. Best case =
shallow × easy (d4 MNIST −0.15pp). **Depth and task-hardness multiply.**

**Architecture-dependent onset** (CNN delays but does not abolish):

- `lenet5` cascaded (n=1000) deployed→ANN: MNIST 0.39pp < KMNIST 3.06pp <
  FMNIST 7.86pp (real residual on a valid CNN); SVHN ≈19pp but non-finalized.
- `deep_cnn` onset at **d6** — conv-shared/pooled structure delays it vs
  `deep_mlp`'s d4 onset; the deficit tracks the greedy single-spike
  partial-sum chain length.
- Reference (now `VALID_FLAGGED`) `deep_mlp`: earliest/worst onset — d4
  +4.32pp, d8 +9.27 MNIST / +15.71 FMNIST.

### 2.3 Rescue is negative on the valid convnet

The first valid deficit cell (deep_cnn d6 MNIST onset, on-chip 99.41%, 3
seeds, S=4) was gridded over the two rescue knobs:

- **`conversion_policy`** (controller revive→refine routing): **does not
  rescue.** cpFalse 0.9500 vs cpTrue 0.8983 → −5.17pp mean / −1.50pp median;
  cpTrue s2=0.77 is a genuine rc=0 *finalized* collapse (a bad basin, not a
  crash).
- **`ttfs_theta_cotrain`** (per-channel θ gain-trim): **broken** — all 6
  cotTrue runs are rc=1 on Conv2D (`RuntimeError` forward fail at
  `Conv2DPerceptronMapper` `features_3`, shape mismatch 28 vs 16). The 0.99+
  `target_metric` floats are **stale pre-deployment ANN-stage artifacts**,
  not deployed accuracies.

**Conclusion:** no working firing-gain rescue knob on the valid convnet; the
~5pp d6 plateau has no fix today; synchronized stays the unconditional
default (sync ceiling at d6 = 0.9904 full-10k vs ANN 0.992).

The famous **+7pp controller "rescue"** (0.8754→0.9452) was **only** on the
INVALID `deep_mlp` d8, and ablation isolated it as the **controller driver**
(the gradual 8-rung ramp + adaptive post-finalize recovery; finalize-cliff
~0.45→~0.94) — **not** keystone/policy/escalation (cpFalse 0.9396 ≈ cpTrue
0.9417 with the fast ladder off). On valid `lenet5`/FMNIST, rescue is only
partial (+1.17pp, then a hard ~5.6–5.9pp AC2 floor). `conversion_policy` is
deficit-proportional, moves no valid cell to AC2-MET, and the
ESCALATE-vs-MATCH branch is never separable in a real run.

### 2.4 The five-mode landscape

`mlp_mixer_core` / MNIST / S=4 / n=3, 15/15 rc=0; deployed 3-seed mean vs ANN:

| Mode | Deployed | Gap | Note |
|---|---|---|---|
| ttfs (analytical) | 0.9807 | +0.15pp | lossless |
| ttfs_quantized | 0.9773 | +0.55pp | ~lossless, fastest (~500s) |
| ttfs_cycle SYNCHRONIZED | 0.9607 | +2.19pp | tightest genuine spiking (sd 0.25pp) |
| lif | 0.9600 | +2.29pp | budget statement (LIF reaches ≥ANN at stabilize 400→6000, not this config) |
| ttfs_cycle CASCADED | 0.9523 | **+3.11pp** | LOSSY OUTLIER, last by every measure (sd ±0.91pp) |

Cascaded is **−0.84pp below its sync sibling** (confound-free — the only knob
is `ttfs_cycle_schedule`; per-seed sync−casc all positive: +1.62/+0.18/+0.72),
and −2.5..−2.9pp vs analytical. Its highest variance is the cold-cascade
death-cascade fragility — a shallow corroboration of the §2.2 depth law.

### 2.5 Tiered validity (gate-v2) — the re-classification

**Hard constraint.** A valid deployment runs ≥50% of total params on chip.
The gate is `onchip_majority.py:assert_onchip_majority_or_raise`, wired into
`SoftCoreMappingStep` after IR pruning, default-on, 50%-majority floor
(merged 35312a0). On-chip = total_params − unique host-side ComputeOp params
(offloaded encoder Linear/Conv, classifier readout, attention; deduped by
module identity).

**Gate-v2 (tiered).** Replaces the binary gate with three tiers on a 20%
floor / 50% majority, checked on **both** params and MACs:

| Tier | Condition | Behavior |
|---|---|---|
| INVALID | below 20% on either metric | reject at enqueue |
| VALID_FLAGGED | 20% ≤ min < 50% | deploys; counts as transferable-tuning evidence; records a research-gap (placement vs unsupported_op) |
| VALID | ≥50% both | majority |

Re-classification of `deep_mlp` w64:

| Depth | On-chip | Tier | Rows |
|---|---|---|---|
| d4 (& parse-unknown) | ~19.5% | INVALID (below 20% floor) | 20 |
| d6 / d8 | ~28.7% / ~36.1% | VALID_FLAGGED_placement (RE-INSTATED) | 30 |
| d12/d16/d24/d32 | ≥~45% | VALID_FLAGGED_placement (also ANN-training-floor confound) | 20 |

So `deep_mlp` d6/d8 death-cascade is **VALID-FLAGGED evidence again** (the
chip does 29–36% of the work, ≥floor; its flag = the host encoder Linear is
*offloadable* → ~99% VALID via `encoding_layer_placement=offload`). d4 onset
stays INVALID. ViT-B (0.33/0.33) = VALID_FLAGGED with a **research-gap** flag
(on-chip attention/LN, a future frontier) — distinct from `deep_mlp`'s
*placement* flag.

Per-family on-chip fractions (valid vehicles): `lenet5` 99.1%, `deep_cnn`
98.5–99.5%, `mlp_mixer_core` 90.1%, `deep_mlp` w64 19.7% (d4) / 36.4% (d8).

### 2.6 Measured coverage (genericity as a fraction)

Live `coverage_report.py` over `runs/campaign/ledger.jsonl` (165 rows):

- **97 covered cells** (science-valid ledger rows), by tier: **VALID=43,
  VALID_FLAGGED=38, INVALID=16**.
- **deep_cnn breadth claim** (`ttfs_cycle_based × {MNIST,FMNIST,KMNIST,SVHN} ×
  {cascaded,synchronized}`): **6/8 covered → coverage fraction 0.750**, with the
  named untested frontier = **deep_cnn × SVHN × {cascaded,synchronized}** (the
  hardest dataset, not yet finalized). This is the honest "genericity as a
  measured fraction" headline — a number *and* a named gap, not an assertion.
  deep_cnn is ≈38% of covered cells.
- **RESEARCH-GAP frontier: NONE** — no unsupported-op flags in the current
  ledger.
- **PLACEMENT-FIXABLE frontier: 1 op** — `encoding_layer(placement)`:
  offloadable encoders that un-flag the cell (not a research gap).
- Axis classification: **INTERACTING** = firing, sync, quantization, S,
  depth; **ORTHOGONAL** = encoding_placement (collapsed → `subsume`),
  pruning, backend, mapping_strategy, vehicle, dataset, regime.

---

## 3. Engineering landed (tests-first, merged to main)

Each item: what + why, one line, all merged.

- **E2 — Static validity pre-check** (merge b70f458 / commit 6be1897). Lifts
  on-chip-fraction to a pure resolver computed at enqueue + pipeline-assembly,
  **before** any train/fuse/place. Kills ~45% of retired compute (72/158
  ledger rows previously burned a full pipeline before rejection). In-IR
  `assert_onchip_majority_or_raise` kept as defense-in-depth. Includes the E7
  transformer fork.

- **gate-v2 — Tiered validity** (merge a02d469 / commit 65f6b3a + doc
  4045938, reclass 72a6766). Three tiers on params AND MACs (floor 20% /
  majority 50%); VALID_FLAGGED deploys + records a research-gap classified as
  `placement` (supported Linear/Conv host-placed under subsume, fixable via
  offload) vs `unsupported_op` (attention/LayerNorm, no on-chip SNN mapping
  yet). Consequence: ViT-B (0.33/0.33) becomes a VALID_FLAGGED headline
  vehicle, not retired; `deep_mlp` d6/d8 re-instated.

- **E1 — Hypervolume coverage ledger** (merges 9dfa724, b7757a5 / commits
  405f773, 481bff8). Extends the certification cell from (firing × sync ×
  backend) to the full config tuple; classifies axes orthogonal vs
  interacting; emits `coverage_report` = genericity as a measured fraction
  with a named untested frontier. Adds depth as a coverage coordinate + a
  claimed-default SSOT (real-ledger coverage). Reports {VALID, VALID_FLAGGED,
  INVALID, untested} per cell + the research-gap frontier.

- **Residual Tier 0 — host-side merge** (merge 5477a47 / commit 206dc2a,
  design 1cc8f17). `y = x + F(x)` deploys **bit-exact with ZERO production
  code change** as a multi-input host-side ComputeOp
  (`ComputeAdapter(operator.add)`) — the same path `SkipPerceptronMixer`
  uses. Latency alignment is by topological order + state_buffer refcounting
  (the buffer *is* the delay line, indexed by node_id not cycle) — no delay
  buffer, no shiftable-core change. Per-branch scale correctness via
  `ScaleNormalizingWrapper`. NF↔SCM structurally exact (shared host
  arithmetic, float64). Shipped as the `MinimalResidualBlock` fixture + parity
  tests. **Unblocks WS2** (residual/norm backbone for Wave-B).
  Dimension-changing (projection) residual still needs a projection
  PerceptronMapper on the skip branch first.

- **E4 round 1 — capacity diagnostic** (merge e0252a6 / commit 90dd016).
  Static `estimate_cores_needed(ir, platform)` at SCM time → a clean capacity
  verdict (needed vs available + which segment overflows) replacing the HCM
  "no more hard cores" mid-pack crash. Wired into the SCM-step gate
  (`_run_capacity_gate`, default True) + the GPU scheduler enqueue precheck so
  infeasible-at-scale configs never claim a GPU.

- **E4 — scheduling-aware capacity** (merge 92734b9 / commits 6c8d99a,
  e10e604). Extends `estimate_cores_needed` with `allow_scheduling`:
  feasibility branches on PEAK-phase (max segment bound) vs SUM.
  `CapacityEstimate` gains `scheduling_aware`/`peak_phase_cores`/`phase_count`.
  VGG16@224 flips infeasible → "feasible-via-scheduling, ~155–209 phases,
  peak fits 2048." The atomic-unit hard-gate = a single coalescing bundle
  must fit budget (VGG largest = frags=18 ≪ 1000, so every conv is
  schedulable). Byte-identical when `allow_scheduling=False`. Reuses
  `_segment_lower_bound` (no new packing).

- **E6 — cascaded-rescue quarantine + kill-gate** (PROGRAM_PLAN_v2 Phase 3;
  campaign-policy E8). Cascaded-rescue is **quarantined behind the Pareto
  gate**: do not fix `theta_cotrain` on Conv2D or run more rescue grids until
  the Pareto allocator shows cascaded has a cost/energy advantage worth
  rescuing on a VALID vehicle. The campaign loop now optimizes
  information-value (cheapest unanswered cell) with kill-gate propagation (a
  settled/failed cheap rung cancels dependent expensive rungs). Scheduler
  daemon hardened: 714226d skips malformed backlog batches instead of
  crashing.

---

## 4. The scale frontier (E3 → E4)

Two regimes, two truths — all measured probes, no training, no production
code modified.

### 4.1 E3 scale probe: bit-exactness does not survive at conv-headline scale

- **VGG16 @ 3×32×32 (CIFAR):** SCM completes, HCM placement completes (940
  cores, 10.4s), **VALUE-DOMAIN bit-exact** (`out_max_abs == 0.0`, rung-3
  lock holds), on-chip 0.9956.
- **VGG16 @ 3×224×224 (ImageNet conv headline):** SCM completes **only** with
  coalescing on (87,644 cores); HCM placement **FAILS** "No more hard cores
  available" — the first early conv segment (`features_6`) **alone** = 50,176
  softcores and exhausts the 1000-core budget; total est ~416,560 cores vs
  1000 = ~416× overflow; on-chip fraction 0.0.

**Verdict:** the bit-exact attributability keystone does **not** survive at
conv-headline scale; E4 placement work is REQUIRED before any GPU-weeks.
Genericity holds on the *value domain* of the mappable surface; the
*instruments* (per-neuron attributability, the placer) do not scale.

### 4.2 E3 → E4 reframe: weight sharing is impossible; weight reuse is time-domain

The spatial-multiplexing hypothesis is dead. **Weight sharing on the chip is
not possible** — each conv output position × channel needs its own physical
softcore; the 224² spatial unroll is **intrinsic**, not a mapper bug. The
lever for "cores too few/small" is the existing **scheduled** mapping path
(`allow_scheduling`): the chip is re-programmed across phases / sync
barriers, reusing physical cores over phases. The simulator must **never**
assume weight sharing (it may later be an explicit opt-in HW capability,
never assumable).

### 4.3 E4: the scheduled path makes ImageNet conv feasible-via-scheduling

- Irreducible spatial-unroll floor for VGG16@224 = **~137,788 softcores**
  (1/output position, no weight sharing).
- The 416–465K count is ~3.0× over the floor = entirely the
  coalescing-fragment-as-separate-core accounting (`frags = ceil(fanin/256)`)
  against a too-narrow 256-axon budget — **reducible** by widening
  `max_axons` (frags→1; e.g. 2048-axon CIFAR cores → 940 total) or counting a
  coalesced group as one logical core; **not model-intrinsic**.
- Neuron-split contributes **zero** overhead at `max_neurons=256` (all 13
  convs out_ch ≤256).
- On a 256×256×2048 chip, `allow_scheduling` flips infeasible → feasible:
  peak phase fits (2048), every conv's atomic coalescing unit (≤18) ≪ budget,
  VGG16@224 needs **~155 phases** (per-segment fresh-pool model; doc headline
  ~209 phases cross-checked against real `split_softcores_by_capacity` = 239
  sub-segments, lower bound 235 ≤ 239, sound).
- Mechanism empirically proven (5000 softcores → 239 sub-segments, peak
  21/phase, verify feasible).

**So "413K is extreme even scheduled" = COSTLY, not infeasible.** The
frontier moved from "can't map" to "maps at N phases — now cost the N."

### 4.4 E7 (transformer validity fork, measured byte-exact)

ViT-B FAILS the ≥50% gate at ~0.33 on **both** metrics (param 0.3309, MAC
0.3311) — only the 12 MLP-block first Linears map on-chip. The MAC metric does
**not** rescue ViT (attention is param- AND MAC-heavy host-side), so the
"redefine the gate as MAC/energy" escape is **foreclosed by data**. VGG16
passes comfortably (0.9997 both). Recommendation (now superseded by gate-v2
tiered validity): ViT becomes a VALID_FLAGGED headline vehicle; the
conv-backbone (ResNet-50/VGG/ConvNeXt) is the headline first; on-chip
attention/LN is a separate future contribution.

---

## 5. Proposed improvements (first-class items)

### (a) Weight-reuse scheduling — the key cost vector (in flight)

Load weights once + time-multiplex the data, with **no reprogramming**;
schedule = **M cheap reuse-phases + N reprogram-phases**
(fixed-mapping-max-parallelism + time-mux-output). This **intersects
pruning** — pruning shrinks cores → fewer phases → cheaper schedule.
**Note: on-chip weight *sharing* is impossible; weight *reuse* is
time-domain (load once, stream data).** The current scheduled model is
fresh-core-pool-per-pass (pure reprogram-per-phase); the new mode adds reuse
phases. It is being re-done on current main, **extending** the merged
`estimate_cores_needed` (the round-1 prototype was authored on a stale base
missing merged E4r1 and was refuted by a package collision).

### (b) GAP-R — reprogramming latency/energy cost term

There is currently **no** reprogramming latency/energy term anywhere
(`grep reprogram|reconfig|reload` over the cost path finds nothing).
`chip_simulation/cost_extraction.py` models energy ~ Σ_d neurons_d·S_d
(soma-dominated) and `latency_steps` = Σ_d timesteps_d with **zero**
inter-pass reprogramming penalty; `cores` is a per-segment count, not
peak-phase. Only the config *suggester*
(`hw_config_suggester_scheduled.py`: cost = core_area·passes^latency_weight)
and the verifier's `schedule_sync_count` touch pass-count, and **neither
charges weight-reload time/energy**. A ~155–209-phase VGG16@224 schedule is
modeled as if reprogramming were free. The phase-count × per-phase
reprogramming-latency/energy product is the open honesty gap; it must enter
the Pareto. Next: a `reprogram_passes` / `mj_per_reprogram` term on
`cost_extraction`.

### (c) Residual Tier 1 — on-chip merge (LIF window alignment)

Branch `residual-tier1-onchip` (commit bed8b36, doc §7 d739e01; ISOLATED /
NOT MERGED). A mapper-graph rewrite
(`mapping/support/residual_merge.py::lower_residual_adds_to_onchip_merge`)
replaces the param-free host add with a frozen identity-concat `[I|I]`
signed-IF merge Perceptron (no bias, `requires_grad=False`) fed by
`_ResidualConcatMapper`, reusing `PerceptronMapper→map_fc→add_neural_core`
(so neuron_split / axon_fuse / coalescing work for free). Config-gated
`onchip_residual_merge` (default OFF → byte-identical host add). **Proven:**
param-free, on-chip fraction ON≥OFF, merge stays in ONE neural segment,
default-OFF byte-identical, 906+ tests unbroken. **Open (round 3):** full
NF==HCM bit-exactness NOT reached (`max|Δ|=0.125`; per-neuron parity 0.25,
honestly xfailed). The in-segment merge diamond has two sources at **different
cascade depths** (stem d1, F d2); `LifSegmentPolicy.run_segment`'s per-cycle
cascade does not reproduce the HCM per-source latency-WINDOW integration at
the merge boundary (stale/zero source buffer past `src_lat+T`) → a sparse
~1/T residual. A naive depth-aware train-shift over-corrected (off-by-1 →
off-by-4) and was reverted. Round 3 = focused depth-aware per-source window
alignment in `run_segment` matching the HCM merge-window boundary, then wire
into the prod conversion flow behind the flag.

### (d) GAP-1 — per-neuron attribution reassembly at scale

Seam `tests/integration/_split_reassembly.py::hcm_per_perceptron_counts →
_reassemble`. Per-neuron LIF k==k reassembly mis-attributes ~2%
(1276/65536) of conv-perceptron-1 neurons at VGG-CIFAR scale under
coalescing + neuron-splitting, **while** the per-neuron total is identical
(654==654) and `out_max_abs==0.0`. Spikes are conserved + the decode is
value-exact, but *which* physical hard-core neuron maps to *which* logical
neuron is scrambled. Scale-emergent bookkeeping gap (PASSES at wide_dim=64).
For LIF this is **THE** per-neuron lock — no second instrument behind it. It
does **not** block the value-domain deployment claim. Fix direction
(test-first, pure bookkeeping, no sim-dynamics change): key reassembly on
`(ir_core_id, neuron_range_in_original)` **jointly** instead of a global
orig-offset sort + role filter that admits same-range fragments from
different IR cores; use the authoritative master/accum psum chain; add a
VGG-CIFAR-fan-out scale regression test asserting k==k.

### (e) E5 Pareto allocator decides cascaded-rescue

The Pareto allocator (energy/latency, with GAP-R's reprogramming term
included) is the gate that decides whether cascaded-rescue is ever worth
fixing. The principled firing-gain rescue lever (per-channel θ co-train) is
**untestable on the valid convnet until the Conv2D `theta_cotrain` forward
bug is fixed** (shape 28 vs 16 at `features_3`). That fix is cheap and is the
only thing that makes the rescue lever measurable — but it stays quarantined
(E6) behind the Pareto verdict.

---

## 6. Further work toward the goals (ordered roadmap)

The ordered sequence, with what runs in parallel and isolated:

1. **Finish the E4 scheduling/cost vectors** — (a) weight-reuse scheduling +
   (b) GAP-R reprogram cost term. These complete the honest cost model:
   `estimate_cores_needed` already gives peak + phase_count; GAP-R charges the
   phase count. *(In flight; extends merged E4r1.)*
2. **A real VGG@224 scheduled-build probe** — confirm peak / phase_count
   against `verify_hardware_config(allow_scheduling=True)` + `_build_scheduled`
   (the path already exists and is tested).
3. **E5 Pareto allocator** — energy/latency Pareto with the reprogramming term
   included; this is the gate that settles cascaded-rescue (E6 quarantine
   releases here).
4. **Residual Tier 1 on-chip merge** — round-3 LIF window alignment, then wire
   behind the flag. *(Isolated branch; parallel-safe — does not touch the cost
   path.)*
5. **GAP-1 per-neuron reassembly fix** — restore the LIF per-neuron lock at
   scale. *(Pure bookkeeping; parallel-safe — does not touch sim dynamics.)*
6. **Wave-B rigor** (cheap, parallel): published-baseline head-to-heads
   (RMP/QCFS/percentile-norm), a residual/transformer backbone for
   deep-trainable points, CIs/ablations bundle. Residual Tier 0 already
   unblocked the residual backbone.
7. **Wave-C pretrained near-SOTA + ImageNet** (GPU-weeks, gated on Wave-B):
   ResNet-50/ViT-B, CIFAR→ImageNet — **now reachable** via the Scheduled path
   + weight-reuse at a *costed* phase budget.

**Parallelism / isolation.** Items 4 (residual Tier 1, isolated branch) and 5
(GAP-1 reassembly, pure bookkeeping) run in parallel with the
E4-cost/E5-Pareto critical path (items 1–3), because neither touches the cost
path or sim dynamics. Wave-B (item 6) is cheap and parallel; Wave-C (item 7)
is gated on Wave-B.

---

## 7. Honest scope — demonstrated vs flagged

**Demonstrated and measured:**

- Chip lowering is bit-exact on every valid vehicle (NF↔SCM 0.0000%,
  torch↔sim 1.0) — all loss is in the conversion.
- Synchronized TTFS is a lossless deep default (≤3.06pp worst valid cell;
  deep_cnn MNIST ≤0.18pp).
- The cascaded death-cascade is a real dual-axis law (depth × task-hardness
  multiply; worst valid d10 FMNIST casc→ANN 20.97pp), with sync as the
  confound-free reference.
- No firing-gain rescue knob works on a valid convnet today; the +7pp
  controller rescue was an INVALID-vehicle artifact and was ablation-isolated
  to the driver.
- Validity is tiered and measured (gate-v2; 97 covered cells: VALID 43,
  VALID_FLAGGED 38, INVALID 16; deep_cnn breadth coverage 0.75, SVHN untested).
- ImageNet conv is feasible-via-scheduling (~138K irreducible softcores;
  ~155–209 phases; peak fits 2048).

**Flagged frontier (named, not closed):**

- **Publication frontier is unbuilt** (the venue bar is NOT met): no
  published-baseline head-to-heads; no residual/transformer backbone for
  deep-trainable points (plain Linear+ReLU/CNN never trains past d≈8 →
  ANN-training-floor confound); no CIs/ablations bundle; no energy/latency
  Pareto allocator; no pretrained near-SOTA bridge.
- **Research-gap frontier in the ledger is currently NONE** (no
  unsupported-op rows), but ViT-B carries a research-gap flag (on-chip
  attention/LN unbuilt) in the validity audit.
- **The cost model is incomplete** until GAP-R charges reprogramming — a
  ~155–209-phase schedule is currently modeled as if reprogramming were free.
- **Two attributability instruments do not yet survive at scale**: GAP-1
  (per-neuron LIF reassembly, ~2% mis-attribution under
  coalescing+neuron-split at VGG-CIFAR; value domain still exact) and
  residual Tier 1 (max|Δ|=0.125 at the merge boundary).

**Confounds carried (named):**

1. Cascaded deep_cnn uses an **n=200 subsample** (~1.5–3.5pp/seed binomial
   noise) while sync reports full 10k — read casc→sync **gaps**, not third
   decimals.
2. Deep rc=1 HCM packing crashes were superseded by bigcores
   `cores.count=480` rc=0 re-runs.
3. Thin seed counts at the deepest cells (n=1/n=2; some rc=−9 OOM/timeout).
4. **Resolution HARDENS the law** — n=1000 reads 8.51/11.14pp at d8/d10,
   *larger* than n=200 (not a grid artifact; a genuine mid-pipeline SCM
   collapse 0.9939→0.1873→0.7375).
5. **No at-chance confound on any VALID cell** (all ANN ≫ chance) — but
   INVALID deep_mlp d≥12 ARE training-floor confounds.

**Net.** The from-scratch diagnostic tier is **done and honest**. The science
is settled within its measured scope; the engineering has moved the scale
frontier from "can't map" to "maps at a costed phase budget." The next dollar
is the E4 cost vectors (weight-reuse + GAP-R) → E5 Pareto → Wave-B rigor →
Wave-C pretrained near-SOTA + ImageNet.
