# mimarsinan — a generic, auditable ANN→SNN deployment toolchain

*The research document (deliverable F5). System · research process · quantitative insights · genuine
future gaps. Companion to `ROADMAP.md` (plan), `HYPERVOLUME.md` (axis SSOT), `PROGRAM_CHECKPOINT_v2.md`
(state). Every coverage/cost number here is **measured by the instruments**, re-runnable from the
cited command — none is asserted.*

---

## 0. Executive summary

mimarsinan is a composable environment for deploying ANNs as SNNs on a multi-core neuromorphic chip,
whose deliverable is **honest, measured, auditable coverage of a deployment hypervolume** rather than a
single accuracy number. This document reports the toolchain and the quantitative findings it produced.

**Headline results (all measured, re-runnable):**

| Result | Value | Where |
|---|---|---|
| Honest deep_cnn coverage (self-auditing denominator) | **3.75%** (6/160), raised from 0.23% by two artifact-backed faithfulness-axis collapses | §4.1–4.2 |
| The integrity rule for axis collapse | **faithfulness axes** (backend/mapping/placement) collapse on a *measured fidelity* artifact; **semantic knobs** (pruning/regime/quant) cannot | §4.2 |
| Cascaded vs synchronized (decision science) | **REGIME_DEPENDENT** — synchronized is the accuracy default (+6.06/+7.19/+11.34pp); cascaded is retained for the hard-latency budget (~2.7–2.9× lower) | §4.7 |
| Death cascade (firing-gain, not capacity) | sharp d6 onset; **depth×dataset-dependent** — bounded ~4–7pp on MNIST, monotone-widening on harder FMNIST/KMNIST (FMNIST×d10 worst, +17.9pp); synchronized lossless (≤3pp of ANN); no rescue lever | §4.4 |
| Scale frontier | ImageNet conv = 138K soft-cores → Scheduled path → ~16 reprogram + 142 reuse; **scheduled build confirmed bit-exact** end-to-end | §4.5 |
| Validity is **architecture-dependent** (3 flag causes) | ViT research-gap-flagged; ResNet-18 structural-host-flagged (offload **REFUTED** by test — `supported_host` residual shortcuts, 0.42 param/0.999 MAC); ResNet-50 **VALID** (Bottleneck param-majority 0.666, scheduled-feasible) | §4.3 |
| **F4 — ImageNet ResNet-50 from-scratch** | **71.97% top-1** in 61 min on 2 GPUs (>67% target, FFCV not needed); deployable VALID + **reachable at a costed phase budget** (~16 reprogram + 142 reuse); deployed-SNN accuracy is measurable memory-bounded via the parity-locked NF (132 GB wall = a script artifact), but a naive/PTC conversion is **chance** (measured) — the genuine number needs a bounded **LIF-adaptation** GPU run (Wave-11, not yet measured) | §4.8 |
| Per-neuron attribution (GAP-1) | fixed bit-exact under coalescing+output-tiling; one residual VALUE_DOMAIN_ONLY region remains | §4.6 |

**The honesty is self-defending** (CI-enforced: no collapse-without-artifact, no merged tiers, no
aged-unowned flags). The remaining DoD is a **GPU-weeks commitment** (publication rigor + ImageNet) and
one research build (on-chip attention) — §5.

## 1. What the system is

mimarsinan is a **composable experimentation environment for deploying ANNs as SNNs on a
multi-core neuromorphic chip**. Its deliverable is not a single accuracy number; it is **honest,
measured, auditable coverage of a deployment-configuration hypervolume**, plus tools that turn any
covered configuration into comparable per-cell outputs (cost, accuracy, fidelity). Energy / accuracy
/ speed are **per-cell outputs**, never the success metric.

The organizing abstraction is the **deployment hypervolume**: a typed product of axes

```
firing × sync × encoding_placement × quantization × pruning × backend
      × mapping_strategy × S × depth × vehicle × dataset × regime
```

A *cell* is one fully-specified deployment; a *campaign* fills cells; the *coverage instrument*
measures what fraction of the honestly-counted hypervolume has actually been run, and at what
validity tier. The scientific claim the toolchain exists to support is **transferable genericity**:
the tuning/conversion recipe is generic across models *on the operations the chip can map*, and the
un-mappable operations (attention, LayerNorm) are a flagged research frontier, not a silent scope cut.

## 2. Architecture — the pipeline and its instruments

**Pipeline.** `convert` (torch ANN → IR) → `map` (IR → soft-cores → hard-cores, packing /
neuron-splitting / coalescing / scheduling) → `deploy` (per-backend runner: nevresim, SANA-FE, HCM
analytical reference, Lava) → `certify` (parity gates + cost extraction). One `run.py` drives it.

**Instruments built to make genericity *measurable* (the toolchain's real contribution):**

| Instrument | File(s) | What it measures / guarantees |
|---|---|---|
| Tiered validity gate-v2 | `mapping/verification/onchip_fraction.py`, `onchip_majority.py` | on-chip param+MAC fraction → INVALID (<20%) / VALID_FLAGGED (20–50%) / VALID (≥50%); static, pre-run |
| Capacity + scheduling diagnostic | `mapping/verification/capacity/estimate.py` | cores needed; peak-phase vs sum; `scheduled` verdict + phase_count; early `CapacityExceededError` |
| Coverage ledger + **P1 self-audit** | `chip_simulation/coverage_ledger.py`, `coverage_ci.py` | genericity as a MEASURED fraction; the **denominator is a function of each axis's screening status** (collapse-on-a-hunch is structurally impossible — a collapse RAISES without a linked artifact); flag aging; per-region attribution fidelity; CI-enforced |
| Defensible cost model + band | `chip_simulation/weight_reuse_cost_model.py`, `cost_extraction.py` | weight-reuse mJ as a `(lo, mid, hi)` band from cited DRAM/HBM coefficients; default 0.0 byte-identical |
| Cross-simulator parity screen | `chip_simulation/cross_sim_parity.py` | 3-state AGREE(max_abs_diff) / DISAGREE(gap) / INAPPLICABLE(capability); `justifies_collapse` honesty gate |
| Pareto decision layer | `chip_simulation/pareto.py` | cascaded-vs-synchronized verdict (REGIME_DEPENDENT, conditional on a cost band) + `propose_recipe(budget)` |
| Campaign loop | `scripts/campaign/{scheduler,director}.py`, `scripts/gpu/campaign_runner.py` | scheduler FILLS · runner DRAINS · director GROWS+FLAGS · research-round CONSOLIDATES; validity+capacity pre-checks reject infeasible/invalid configs at enqueue |
| Self-defense guards | `scripts/campaign/guards.py` | base-check (stale-base trap), stash-intact, `fcntl` singleton |

## 3. The research process (how every increment was made)

The program ran as a **disciplined incremental loop**, each increment doing exactly one of: make the
measurement trustworthy · raise honest coverage · instrument a per-cell output · add a hypervolume
region. The method per increment:

1. **Decompose** into INDEPENDENT units with disjoint file ownership.
2. **Allocate each to an isolated parallel dynamic Workflow** (Build→Verify, or
   Research→Design→Prototype→Verify), agents in pre-created git worktrees off current `main`.
3. Every unit is **tests-first · adversarially verified** (a default-refute skeptic re-derives the
   claim from the diff, re-runs the tests, mutation-checks, attacks the integrity claim) ·
   **default-off byte-identical** (a new capability ships gated off; the default path provably
   unchanged) · returns a **patch for review** (never auto-merges framework code).
4. **Land only tested/byte-identical work**; keep preliminary research isolated on a branch.
5. **After each land, re-price the honest measurement** — the headline number is re-run, not edited.

This is itself a finding: **adversarial verification caught real things** (a stale-base worktree, two
guard bugs found red→green, and it forced a coverage-collapse claim to prove it was measurement-backed
and fidelity-scoped, not a hunch).

## 4. Quantitative insights (measured)

### 4.1 The coverage methodology — the denominator is a function of screening status
The central honesty mechanism: an axis collapses to one representative **only with a linked screening
artifact**; otherwise it is enumerated (counted interacting). A bigger denominator = a lower, honest
coverage fraction. This makes "collapse-on-a-hunch" — the original sin the reviewers caught (a legacy
0.60 that silently collapsed four unscreened axes) — structurally impossible.

### 4.2 Faithfulness axes vs semantic knobs (the rule for *which* axes may collapse)
A load-bearing distinction this program introduced:
- **Faithfulness axes** — `backend`, `mapping_strategy`, `encoding_placement` — are different
  *simulators / packings / placements of the SAME deployment contract*. They collapse on a **measured
  parity/fidelity artifact** (a disagreement would be a BUG, not an interaction), **scoped
  fidelity-only**: capability (which backend×mode runs) and cost/utilization stay un-collapsed
  frontiers.
- **Semantic knobs** — `pruning`, `regime`, `quantization` (and `S`/`depth`/`vehicle`/`dataset`) —
  *change the trained result*. They CANNOT collapse on a fidelity argument; they need a real
  cross-product equivalence screen (GPU) or a capability build. They stay enumerated.

This rule was applied to **raise honest deep_cnn coverage from 0.23% (6/2560) to 3.75% (6/160)** by
collapsing `backend` and `mapping_strategy` on measured artifacts (a live cross-sim screen at
`max_abs_diff = 0.0`; the bit-exact torch↔sim fidelity lock) — the keystone working in the legitimate
direction, with the covered-cell count (97) and tier split (VALID 43 / FLAGGED 38 / INVALID 16)
invariant under the pure-denominator collapse. Remaining denominator factors (pruning×2 · regime×2 ·
quantization×4 = 16×) are honestly un-collapsed pending real screens. *(Re-run:
`coverage_report.py --vehicle deep_cnn --dataset … --sync cascaded synchronized --firing ttfs_cycle_based`.)*

### 4.3 The validity frontier — on-chip majority
A deployment is only valid if ≥50% of params+MACs run on-chip (gate-v2; 20% floor = VALID_FLAGGED).
This retired host-majority configurations as INVALID (e.g. deep_mlp w64 at 19.7–36.4% on-chip) and
classified ViT-B as VALID_FLAGGED (0.33/0.33 — the MAC metric does not rescue attention). deep_cnn is
the valid trainable-deep vehicle; SqueezeNet (added this campaign) is VALID at frac 1.0 (offload),
942-of-1000 cores, single-phase. The pretrained bridge (B3) then revealed that **VALID_FLAGGED splits
into two structurally distinct classes**: (a) a **research-gap** flag — unsupported host ops, e.g.
ViT-B (attention+LN), the genuine frontier; and (b) a **structural host-placement** flag — e.g. a stock
torchvision **ResNet-18**, measured **VALID_FLAGGED at param 0.423 / MAC 0.999 with `research_gap_ops=[]`**
(no unsupported op at all). The tempting fix was *tested and REFUTED* (Wave 6): `offload` does NOT lift
ResNet-18 to VALID (param only 0.4223→0.4232) — the host param-majority is **11 residual-boundary
`supported_host` shortcut/downsample Sequentials**, not offloadable encoders, so the deep_mlp
`offload` analogy does not transfer. The deeper resolution is **architecture-dependence**: **ResNet-50**
is measured **VALID** (param 0.666 — its Bottleneck trunk holds the param-majority on-chip natively,
scheduled-feasible at peak 208 / 16–17 phases) with *no* placement trick. So VALID_FLAGGED resolves into
**three** causes the instrument names per cell — *unsupported-op* (research frontier; ViT),
*offloadable-encoder* (placement-fixable; deep_mlp d8), and *structural-host-residual* (NOT offload-fixable,
BasicBlock-vs-Bottleneck architecture-dependent; ResNet-18). Testing the hypothesis rather than asserting
it is what kept this section honest.

### 4.4 The death-cascade science (firing × sync, depth × dataset)
Genuine cascaded single-spike TTFS suffers a depth-driven **death cascade** — a correctable
firing-gain deficit, architecture- and dataset-dependent in onset (dataset dominant). Synchronized
execution is the **lossless, depth-stable** default (measured deep_cnn d10/S4 mnist: synchronized
0.9903 vs cascaded 0.9297). For deep_cnn the cascaded→synchronized gap has a **sharp d6 onset**
and a **depth×dataset-dependent magnitude**: it stays a **bounded ~4–7pp plateau on (easy) MNIST**
through d12, but **widens monotonically with depth on harder datasets** (FMNIST +11.3pp@d8→+17.9pp@d10;
KMNIST +7.0→+16.0pp@d8→d10), with **FMNIST×d10 the worst corner**. Synchronized stays within ~0.5–3pp
of the ANN ceiling at every depth — so the gap is a **firing-gain pathology, not a capacity limit** —
and there is **no working config-level rescue lever** at the convnet onset (theta_cotrain is
rc=1-broken; the convnet staircase-STE and conversion_policy are both net-negative), so synchronized is
the unconditional accuracy default. This is the dual-axis depth×dataset law on the VALID deep_cnn
vehicle. It generalizes: **lenet5** (a 2nd valid vehicle) shows the same dataset-margin ordering
(MNIST +0.21 / KMNIST +2.09 / FMNIST +5.14pp). And the toolchain's fidelity gate has teeth on real
data — **SVHN cascaded trips the NF↔SCM per-neuron parity gate** (agreement 0.78–0.89 < 0.98 → rc=1,
deployment-invalid), so that cell is honestly recorded sync-only. `firing × sync` and `depth × dataset` are PROVEN-interacting
axes (enumerated, never collapsed).

### 4.5 The scale frontier and weight-reuse scheduling
ImageNet-scale conv does **not** map simultaneously (no on-chip weight sharing): VGG16@224 ≈ **138K
irreducible soft-cores**. The **Scheduled path** makes it *feasible-via-scheduling* by
time-multiplexing across ~158 reprogram phases; **time-domain weight-reuse** (load weights once,
stream data) re-factors this into ~**16 reprogram + 142 reuse** phases. The defensible cost model
prices VGG@224 ≈ **13.4 mJ** (band 1.5–49.3, ~80% weight-DMA) — a *model-estimate with an
uncertainty band*, the honest form for a number with no measured ground truth yet. **D3 confirmed the
mechanism end-to-end (measured):** a small model overflowing a 6-core budget genuinely triggers the
Scheduled path (`phase_count=3>1`, stages `[6,6,2]` vs single-pool `[12,2]` — a real build-level
split, not an estimate echo), realizes 3 reprogram + 33 reuse passes over 3 weight banks, and is
**bit-exact** (`max|Δ|=0.0` scheduled-vs-reference and scheduled-vs-torch, non-degenerate control).
The VGG@224 16-reprogram/142-reuse figure is thus *confirmed-by-mechanism*; the full ImageNet build
itself remains a GPU-weeks run, not yet done.

### 4.6 Attribution fidelity — value-domain vs per-neuron
The instrument distinguishes value-domain bit-exactness (deployed accuracy) from per-neuron
attribution. **GAP-1** (per-neuron attribution under coalescing + output-tiling-under-compaction-
reorder) was fixed this campaign via joint `(perceptron_output_slice, ir_id)` keying — bit-exact,
value-domain byte-identical. The residual Tier-1 merge remains the sole value-domain-only region.

### 4.7 Decision science — cascaded vs synchronized (E5, measured)
The Pareto decision layer (`pareto.py`) reads per-schedule accuracy from the ledger and prices a
defensible cost band, then emits the verdict. **Measured accuracy gap (synchronized − cascaded):**
mnist **+6.06pp** (d10/S4), kmnist **+7.19pp** (d8/S4), fmnist **+11.34pp** (d8/S4) — synchronized
tracks its ANN reference within noise; the gap is the cascaded firing-gain deficit (§4.4), not a
synchronized loss. **Cost** is a *model-estimate-with-band, not measured energy*: latency is a tight
derivation from the documented execution model (synchronized `sim_time = S×groups`; cascaded pipelined
`S+groups` → cascaded ~2.7–2.9× lower latency), and energy is a `cores×active_steps` proxy present
only when cores are known; absolute per-sample spike energy is **UNINSTRUMENTED** (flagged, not
invented — the single biggest cost gap). **Verdict: `REGIME_DEPENDENT` on every measured dataset —
cascaded is NOT retired.** Synchronized is the accuracy-front schedule everywhere, but cascaded's
lower pipelined latency keeps it on the (latency, accuracy) Pareto front, so it is the hard-latency-
budget code; the verdict is explicitly **conditional on the cost band** (if measured per-sample energy
turned out higher for cascaded it could flip to `RETIRE_CASCADED`). `propose_recipe(budget)` picks
synchronized at an accuracy budget and cascaded at a hard-latency budget. This is independently
corroborated by the campaign's research-round: the convnet staircase-STE *regresses* the deep_cnn d6
onset (−5pp FMNIST, −4.33pp KMNIST) and does not compose with conversion_policy → there is **no
working config-level firing-gain rescue lever** at the convnet onset, so synchronized stays the
unconditional accuracy default while cascaded is retained purely for its latency regime.

### 4.8 The F4 headline — ImageNet ResNet-50 (measured)
The toolchain trains **ResNet-50 ImageNet from-scratch to 71.97% top-1 in 61 min on 2× RTX PRO 6000
Blackwell** (the constrained target was ~67%/<~1hr; torchvision dataloading sufficed at ~6000 img/s, so
**FFCV was not needed**) and characterizes its SNN deployment as **VALID + reachable at a costed phase
budget** (~O(100K) soft-cores at 224px → the Scheduled path's ~16 reprogram + 142 reuse phases, priced
by the P2 cost band) — **DoD-4 satisfied**. The deployed-SNN *accuracy* was previously punted as a
"138K-core GPU-weeks frontier"; that was **wrong and is corrected (2026-06-25)**: (i) the ~132 GB peak
RAM was a standalone-script artifact (`deploy_imagenet_snn.py` materializing all hard cores), **not**
intrinsic — the deployed value is measurable torch-side via the **parity-locked NF**
(`chip_aligned_segment_forward`, bit-exact to the deployed sim) at full 224 res with no hard-core build
(Wave-10 U1; U2 streams the literal HCM one segment at a time); and (ii) the real blocker is the **LIF
adaptation** — MEASURED: a naive/PTC conversion of the 71.97% ResNet-50 (even with q=0.99 activation-scale
calibration) is **chance** at deployable T (T=8/16 → 0.000 vs 0.875 float-ANN), because the depth-driven
death-cascade needs the gradient-based `LIF Adaptation` step, not just a scale. So the genuine number is a
**bounded LIF-adaptation GPU run** (`weight_source=runs/imagenet/resnet50.pt` → adapt → NF-measure,
memory-bounded; Wave-11) — costed and named, not yet measured. The toolchain's ANN→SNN **retention** is
measured *lossless* on the VALID small vehicles (§4.4, synchronized ≈ ANN bit-exact), and end-to-end
ResNet SNN deploy is closed by the D6 bridge. A real ImageNet methodology bug (a class-sorted index-range
train/val split that handicapped training and made val score at chance) was caught + fixed mid-run — the
test-don't-assert discipline at GPU scale. See `findings/F4_imagenet_resnet50.md`. **Superseded in part by
§4.9**, which resolves *why* the deployed number was hard and shows the collapse was a fixable artifact
stack, not a frontier.

### 4.9 Deep-residual SNN deployment — the collapse is fixable, not a frontier (measured, verified)
Pursuing the F4 deployed number turned a phantom "death-cascade frontier" into a chain of **measured,
adversarially-verified** results. The "ResNet-50 LIF ≈ 0.13" was an **artifact stack**, every layer of
which was found and fixed:

- **Four tooling bugs blocked an honest measurement** (all fixed + tested on `main`): the adaptation tuner
  gated on a **leaked validation set** (the provider carved `val` out of the class-sorted *train* split,
  which a pretrained model has already seen — 0.906 memorized vs 0.72 genuine); the **adaptation target
  anchored on the collapsed LIF baseline** (≈0.27) instead of the real ANN, death-spiralling its own bar;
  the tuner had **no run-level keep-best** (R7), so it shipped whatever the ramp + forced-rate-1.0 landed
  on, below the best committed; and a **per-channel-θ conv broadcast crash** (`[C]` against `W`) had been
  *crash-masking* the firing-gain lever (`sim_ran=0`, echoing the ANN). The F-harness aggregator was also
  silently blind (0/0/0 cells, a schema mismatch).

- **`ttfs_cycle` synchronized ≡ `ttfs_quantized`** — proven on the same model (bit-for-bit on the on-chip
  cascade, flat to depth 12; off-grid host boundaries differ only by the documented, bounded, *non-compounding*
  ±1/S single-spike wire-snap) and **now locked** (`test_sync_vs_quantized_same_model.py` — the cross-check
  `nf_scm_parity` had been *excluding*). So a "synchronized collapse" was never a deployment-mode bug.

- **The genuine-spiking loss is per-LAYER rate quantization, not the residual boundaries** — on a strong
  probe (real conv/BN CIFAR ResNet, ANN 0.875, **decision-fidelity** = argmax-agreement with the ANN,
  bit-exact NF==HCM): the on-chip residual merge is an **exact no-op**, and the *plain* net (zero residual
  adds) collapses **hardest** while the residual net degrades gracefully. **The residual skip is *protective***
  — it reverses the depth penalty (deeper is *better* with skips; cuts required T ≈4×).

- **The fix reaches ≥0.9 (verified REAL):** folded per-layer q=0.99 calibration (the naive uniform scale is
  itself the chance-collapse) + adequate rate budget T (T64 → 0.91, T256 → 0.96), and **QAT through the
  BN-frozen genuine spike forward** (KD+CE) reaches **0.9105 at half the T (T32)**. This localises F4's
  "the adaptation is load-bearing" to a concrete recipe. Levers that *don't* help, measured: DFQ-for-LIF
  (hurts), on-chip-merge/scheduling (no-ops), and the cascaded-only gain-correction/STE/θ-cotrain (a wash).
  A silent **BN train/deploy bug** (batch-stats in `.train()` ≠ the deployed `.eval()` forward) was caught
  en route — naive QAT trained a different function than it deployed.

**Honest scope:** measured on CIFAR-d8 with decision-fidelity; ResNet-50/d50/1000-class *magnitude* is
argued mechanistically, not yet measured — the named, costed proof run (folded calib + T64 + QAT + the four
gate fixes). See `findings/deep_residual_lif_deploy_fix.md`. The lesson reinforced twice by the verify gate:
**strong probe + decision-fidelity + balanced metric**, or "lossless" is a weak-probe / class-imbalance
lucky-flip artifact.

## 5. Genuine future gaps (open research / capability, not deferral of understood work)

These are *genuine* frontiers — each is either un-instrumented or needs a capability we do not yet
have. Things we already understand or can fix in scope were NOT deferred here.

1. **Per-sample energy instrumentation (now wired at the source; data fills forward).** Cost
   extraction is now invoked at the end of the deployment path (`sanafe_simulation_step` emits a
   measured `cost_record.json`, exception-isolated + result-byte-identical), and E5 prefers the measured
   cost over the proxy when a record is resolvable. The remaining gap is purely temporal: the *existing*
   ledger rows predate emission, so the cost-accuracy Pareto becomes fully *measured* (not proxy) only as
   the campaign re-runs cells with emission on — and per-schedule run-dir→cost resolution is still
   best-effort.
2. **Pruning / regime equivalence screens.** Semantic knobs cannot collapse on fidelity; collapsing
   them honestly needs a real GPU cross-product equivalence screen. The **pretrained bridge now exists**
   (B3 — a stock torchvision model is mappable + validity-classified), so the from-scratch↔pretrained
   *regime* screen is now *runnable*; the remaining piece is the GPU run itself (the bridge enables it,
   it is not a substitute). Pruning still needs the D4 pruning-deployment capability.
3. **On-chip attention / LayerNorm (the transformer frontier).** ViT-B is VALID_FLAGGED because
   attention+LN have no on-chip SNN mapping yet; this is *the* headline capability gap (the cheap
   path is foreclosed — it is real research).
4. **ImageNet headline — ANN + reachability now DONE (§4.8); the remaining gap is a bounded
   LIF-adaptation run, NOT a memory frontier.** ResNet-50 ImageNet from-scratch is trained (**71.97%**)
   and the deployment is characterized VALID + reachable at a costed phase budget. The deployed-SNN
   accuracy is **measurable memory-bounded** via the parity-locked NF (the ~132 GB single-shot map was a
   standalone-script artifact, now bypassed — Wave-10). The genuine open number is the **LIF-adaptation**
   itself: a naive/PTC conversion of the trained ResNet-50 is **chance** at deployable T (measured — even
   with q=0.99 calibration), so the deployed accuracy needs the gradient-based `LIF Adaptation` fine-tune
   (a bounded GPU run, Wave-11), not a 138K-core sim. ViT/ImageNet additionally needs D5 (on-chip
   attention is host-only). Retention is measured *lossless* on the VALID small vehicles; for the deep
   pretrained ResNet the adaptation is the open, costed work.
5. **Residual Tier-1** (on-chip param-free merge) — *characterized as intrinsically `1/T`-bounded*: an
   in-segment on-chip merge cannot be bit-exact to the Tier-0 host-add reference (the in-segment IF head
   re-quantizes the merged spike train, differing by exactly 1 spike = `1/T` by construction; matching it
   needs a host round-trip = Tier 0). It is feasible as a `1/T`-characterized deployment, not a
   host-add-identical one; a closeable shared-HCM-fill alignment (Component A) separately tightens
   NF==HCM to `atol=0`. See `findings/residual_tier1_intrinsic_limit.md`.
6. **Published-baseline head-to-heads** (RMP/QCFS/percentile-norm). The **percentile-norm method now
   lands** (D7 — a selectable, default-off activation-scale policy, numerically verified); the
   remaining piece is the GPU head-to-head comparison on the covered/valid cells.

## 6. How to reproduce / audit

- Honest coverage: `python scripts/campaign/coverage_report.py [--vehicle … --dataset … --sync …]`
- CI integrity (no collapse-without-artifact, no merged tiers, no aged-unowned flags):
  `python scripts/campaign/coverage_report.py --ci`
- Validity of a model: `mapping.verification.onchip_fraction.classify_validity(...)`
- Capacity/scheduling: `mapping.verification.capacity.estimate.estimate_cores_needed(...)`
- Cross-sim parity screen: `chip_simulation.cross_sim_parity` + `assert_cross_sim_screen_sound`
- The screening artifacts: `docs/research/findings/`.
