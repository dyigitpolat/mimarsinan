# Program Plan v2 — Post-Review Adjustment (Genericity at the Hypervolume)

*Supersedes the Wave-B/C-first ordering in `PROGRAM_CHECKPOINT.md §7–8`. Adopts the
`review/campaign-checkpoint` technical guidance (E1–E8). The science in the checkpoint
stands; this changes **what we spend the next GPU-week and engineering-hour on**.*

## The review in one line

The rigor is established in the **easy corner** but the claims must land at the
**headline regime** (ResNet-50 / ViT-B / ImageNet). The gap is not a missing spiking
primitive — it is three structural things, all fixable with the abstractions already in
the tree:

1. **Validity is post-hoc retirement, not a static pre-check** → ~45% retired compute
   (72/158 ledger rows burned a full pipeline before rejection).
2. **The hypervolume is unmodeled; coverage is per-run, not per-cell** → genericity is
   *asserted*, not *measured*.
3. **The validity gate (≥50% on-chip *params*) collides with "host the hard ops" on the
   headline models** → a param-majority gate may **retire ViT-B on its own rule** (attention
   is param-light but MAC-heavy), and the metric (params vs MACs vs energy) is a load-bearing
   choice to make *now*.
   Plus: the load-bearing **instruments** (bit-exact mapping, synchronized-as-control, the
   gate) are validated only in the easy corner; the **placement engine** is a single greedy
   pass already failing (rc=1) at d10 small-CNN; and **cascaded-rescue is being invested in
   before proving cascaded is worth rescuing** (the Pareto gate, not the `theta_cotrain` fix,
   should decide that).

## Decisions adopted

- **Validity becomes a cheap, upstream, static pre-check** (E2). Lift the on-chip fraction
  to a pure resolver computed at *enqueue* and at *pipeline assembly*, before any
  train/fuse/place. The in-IR `assert_onchip_majority_or_raise` stays as defense-in-depth.
- **The hypervolume becomes explicit and measured** (E1). Extend the certification cell from
  `(firing × sync × backend)` to the full config tuple; classify axes orthogonal vs
  interacting (justified by cheap screening — `encoding_placement` offload≡subsume is already
  one orthogonal result); emit a `coverage_report` = the genericity claim as a measured
  fraction with a named untested frontier.
- **The campaign loop optimizes information value, not throughput** (E8). Validity pre-check
  at enqueue · information-value priority (cheapest unanswered cell) · kill-gate propagation
  (a settled/failed cheap rung cancels its dependent expensive rungs) · a coverage dashboard
  the director consults. "Never idle" → "never idle on invalid or low-information work."
- **Instruments validated at scale *cheaply, before* the GPU-weeks campaign** (E3, E4).
  A mapping-scalability probe (ResNet-50/ViT-B-shaped IR straight through SCM→HCM — asserts
  *completes* + *bit-exact*, no training) and a synchronized-at-scale probe (deep residual on
  CIFAR). The **placement engine** gets diagnosable failure (capacity report at SCM time),
  capacity-aware placement (bin-packing/look-ahead vs greedy), and profiled scaling.
- **The transformer validity-metric fork is decided up front** (E7). Statically compute
  ViT-B / VGG16 on-chip *param* fraction; if < 50%, choose deliberately between (a) on-chip
  attention/LN, (b) **redefine the gate as on-chip MAC/energy majority** (more likely to pass
  *and* arguably the more honest "the chip does the work", dovetails with the energy proxy),
  or (c) bound the headline scope. **The gate metric is the user's call** — we supply the
  data + recommendation, and build the resolver metric-pluggable (params | macs | energy).
- **Cascaded-rescue is quarantined behind the Pareto gate** (E5, E6). Do *not* fix
  `theta_cotrain` on Conv2D or run more rescue grids until the Pareto allocator shows
  cascaded has a cost/energy advantage worth rescuing on a *valid* vehicle. Build the Pareto
  allocator on `cost_extraction` (`energy_proxy_neuron_steps`, `max_ft_pass_wall_s`) and
  extend `ConversionPolicy.propose_recipe` to pick `(mode, schedule, S)` from a budget — this
  is simultaneously the "automatic genericity" evidence and the resolution of the cascaded
  question. If synchronized dominates on the Pareto, the death-cascade law becomes a
  publishable *characterization* result, not open engineering debt.

## Reordered sequencing

| Phase | Items | Why | Cost |
|---|---|---|---|
| **1 — now, unblock** | **E2** static validity pre-check · **E1** hypervolume + coverage ledger · **E8** coverage-driven campaign policy | recover ~45% wasted compute; make genericity measurable | cheap (code) |
| **2 — before GPU-weeks** | **E3** mapping-scalability + synchronized-at-scale probes · **E7** transformer validity-metric decision | de-risk the frontier for ~free; prevent the two failure modes that waste the expensive campaign | cheap |
| **3 — the decision lever** | **E5** Pareto allocator + `propose_recipe` · **E6** quarantine cascaded-rescue | settle whether cascaded-rescue (§6d) is worth doing *at all* | medium |
| **4 — scaling fix, as E3 demands** | **E4** placement engine: diagnosable + capacity-aware + profiled | the component most likely to block Wave-C | medium |
| **5 — Wave-B rigor** | baselines (RMP/QCFS/percentile-norm) · residual/norm backbone · CIs/ablations | *only on covered, valid, instrument-validated cells* | medium |
| **6 — Wave-C headline** | pretrained near-SOTA bridge · CIFAR→ImageNet · dual-regime certification · full matrix | entered with instruments proven, hypervolume mapped, validity decided | GPU-weeks |

## Starting now

**E2 (static validity resolver) + E7 (headline-model fractions)** — the keystone that
unblocks the rest: a pure `estimate_onchip_fraction(model, encoding_placement,
mapping_strategy, metric)` that reproduces the measured fractions (deep_mlp 19.7/36.4%,
deep_cnn 98.5%, mlp_mixer 90.1%, lenet5 99.1%) *without* training/placement, wired into the
campaign enqueue as a validity pre-check, and used to answer E7 (does ViT-B pass the param
gate?). Tests-first, worktree-isolated, patch for review + safe-merge.

## E7 result (measured, byte-exact vs the authoritative gate)

**ViT-B FAILS the >=50% on-chip gate at ~0.33 on BOTH metrics** (param frac **0.3309**,
MAC frac **0.3311**) — only the 12 MLP-block first Linears map on-chip; the second MLP
Linear, all 12 attention blocks, 25 LayerNorms, patch-embed conv, and head are host. The
**MAC metric does NOT rescue ViT** (attention is param- AND MAC-heavy host-side), so review
option (b) [redefine the gate as MAC/energy] is **foreclosed by the data**. VGG16 passes
comfortably (**0.9997** both metrics). The fork narrows to **(a)** build on-chip attention/LN
(expensive, a genuine contribution) or **(c)** headline on conv backbones (ResNet-50 / VGG /
ConvNeXt — they pass cleanly). Recommendation: **conv-backbone headline first; transformers
become a separate on-chip-attention contribution if the conv headline lands.** The static
gate earned its keep — it foreclosed a GPU-weeks dead-end for free, before any training.

## Vision clarified — tiered validity + the research-gap frontier (supersedes the E7 "conv-only" recommendation)

**Core claim:** the ANN→SNN **tuning/conversion mechanisms are transferable & generic across
models** (architectures × datasets × regimes), achieving **lossless + fast** deployment **on
the operations the chip can currently map** — and this holds *even though the deployment
toolset is incomplete*. The un-mappable operations are an explicit, flagged **research
frontier**, not a scope limit.

**Validity gate v2 (tiered, both params AND MACs):**
- **floor = 20%** on-chip (both metrics) AND the flow maps correctly for the supported ops.
  Below 20% on either metric → **INVALID** (host does ~everything; reject at enqueue).
- **20% ≤ min(param,mac) < 50% → VALID_FLAGGED** — deploys + counts as transferable-tuning
  evidence, and records a **research-gap**: the host-side ops keeping it below majority,
  classified as *placement* (a supported Linear/Conv host-placed under subsume → fixable via
  `offload`) vs *unsupported_op* (attention / LayerNorm / … → no on-chip SNN mapping yet, the
  future-research target).
- **≥50% (both) → VALID** majority, no flag.

**Consequences:** ViT-B (0.33/0.33) becomes a **VALID_FLAGGED headline vehicle** (research gap
= on-chip attention/LN), not a retired model. deep_mlp's flag is *placement* (host encoder is a
supported Linear → `offload` → ~0.99 VALID), so it is un-retired via offload, not flagged.
deep_cnn / lenet5 / mlp_mixer / VGG16 are VALID majority. **E3 toolset shortfalls** (placement
engine at ResNet-50/ViT-B scale) are recorded as research gaps; genericity of the *mechanisms*
must still hold on the mappable surface. The coverage ledger (E1) reports
{VALID, VALID_FLAGGED, INVALID, untested} per cell + the named research-gap frontier.

## E3 verdict (measured) → E4 is now a CONFIRMED Wave-C prerequisite

E3 mapping-scalability probe (VGG16 @ 32² and 224², ViT-B; no training):
- **CIFAR-scale conv maps + completes + value-bit-exact** (940 cores, ~10s placement).
- **ImageNet-scale conv does NOT map on the current toolset**: default chip → `WideFanInUnsupportedError`
  (FC fan-in 25088 > 256 axons → coalescing required); with coalescing → placement **`No more hard
  cores available`** (one early conv segment = 50,176 softcores; **~416,560 cores needed vs 1000 budget,
  416× overflow**); plus a super-linear placement/sim-time wall. Doc: `docs/research/E3_SCALE_PROBE.md`.

**Flagged research gaps** (per the vision — toolset shortfalls, not blockers): **GAP-1** per-neuron
*attribution* reassembly scrambles ~2% at VGG-32 scale (value-exact + spike-conserved; a bookkeeping
bug in `hcm_per_perceptron_counts` under fuse+split); **GAP-2** placement budget exhaustion at
ImageNet scale (the E4 capacity problem); **GAP-3** wide-flattened-conv-FC needs coalescing (a
mapping prerequisite).

**Reorder:** E4 moves up from "as E3 demands" to a **confirmed Wave-C prerequisite**. E4 round 1 (cheap,
launching now) = the **capacity diagnostic**: a static `estimate_cores_needed(ir, platform)` computed at
SCM time → a clean *capacity verdict* (needed vs available + which segment overflows) replacing the HCM
"no more cores" crash, wired into the enqueue pre-check so infeasible-at-scale configs never claim a GPU.
E4 round 2 = capacity-aware placement (bin-packing/look-ahead vs greedy) + the GAP-1 reassembly scale-fix.
Genericity of the *mechanisms* holds (conversion + value-domain bit-exact at CIFAR); the *toolset* gap to
ImageNet scale is now named + measured.

## E4 direction CORRECTED (user insight): no weight sharing — the Scheduled path is the lever

**Weight sharing is NOT possible on the chip** (the spatial-multiplexing hypothesis is dead). Each conv
output position × channel needs its own physical weight placement (softcore); the spatial unroll of a
224²-spatial conv is therefore *intrinsic*, not a mapper bug. The existing mechanism for "cores too small
/ too few" is the **Scheduled mapping path** (`allow_scheduling`): it distributes core placement **across
time** — the chip is **re-programmed across phases / sync barriers**, reusing physical cores over phases
instead of instantiating all at once. **The simulator must NOT assume weight sharing** (it may later be an
*explicit, opt-in* hardware capability, never assumable for all deployments).

So the E4 capacity question is reframed: not "do 413K cores fit at once" (they don't), but **under
scheduling, does the PEAK phase fit the physical chip, and at what PHASE-COUNT + reprogramming
latency/energy cost?** And **413K softcores is still extreme even time-multiplexed** (~413 phases on a
1000-core chip), so the open investigation is: is 413K the irreducible no-weight-sharing spatial-unroll
count, or is there mapping over-production / a better tiling — and what realistic chip size + phase budget
makes a 224²-conv deployable at all? The E4 capacity diagnostic (round 1, merged) must become
**scheduling-aware** (feasible-via-time-multiplexing if peak-phase fits; report phase-count + cost), and
the scaling program must center on the **Scheduled path**, not weight sharing.
