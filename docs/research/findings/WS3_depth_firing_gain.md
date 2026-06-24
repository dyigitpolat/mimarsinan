> ⚠️ **VALIDITY:** deep_mlp results below are an **INVALID host-majority config** (<50% params on-chip; see [VALIDITY_AUDIT.md](VALIDITY_AUDIT.md)). The phenomena may be real but are NOT valid on-chip deployments. Valid trainable-deep vehicle = **deep_cnn**.

# WS3 — Depth × Firing-Gain (cascaded vs synchronized)

**Question.** As `deep_mlp` gets deeper, does the cascaded (single-spike TTFS,
ramp-reconstructed) deployment collapse while the synchronized schedule holds
near the float ANN ceiling? If cascaded collapses with depth, is it a
*correctable per-channel firing-gain* deficit (so the θ-cotrain gate-fix is worth
building on `deep_mlp`), or does synchronized simply own deep deployment and the
cascaded mode should be retired for deep models?

**Status: COMPLETE — 56/56 jobs done (round-1 + round-2).** Both rounds have
produced finalized, deployed results. The headline survives but is **bounded**:
the depth × firing-gain law is established only over the **trainable** depth band
(MNIST/FMNIST, d ≤ 8, width 64), and every attempt to probe deeper — including
the round-2 *wider* d16/d24 @ width 128 rescue — fell into a **training-floor
confound** (the plain `Linear+ReLU` ANN itself never trained past d ≈ 8, so
deployed accuracy carries no firing-gain information at those depths). The deep
firing-gain test is therefore **gated on WS2** (residual/norm backbone).

---

## 1. Matrix (all 56 jobs done)

`deep_mlp`, `spiking_mode=ttfs_cycle_based`, schedule knob =
`ttfs_cycle_schedule ∈ {cascaded, synchronized}`, plain Linear+ReLU stack
(`layer_wise_lr_decay=1`, no residual, no norm). Deployed = soft-core spiking-sim
metric; ANN ref = final Pretraining "Test accuracy"
(`steps.json` → `Pretraining.target_metric`, verified == final `Test accuracy`
metric for all 56 runs).

- **Round-1 (DONE):** MNIST × d {4, 8, 16, 32} × {casc, sync} × seed {0,1,2}
  (w64) + FMNIST × d {4, 8, 16, 32} × {casc, sync} × seed 0 (w64).
- **Round-2 (DONE):** MNIST densify `d6, d12` @ w64 + deep-and-wide rescue
  `d16, d24` @ **w128**, × {casc, sync} × seed {0,1,2} = 24 jobs.

---

## 2. The full table — deployed vs ANN reference, by trainability

3-seed mean ± population sd where n=3 (MNIST); n=1 for FMNIST (seed 0 only).
**ANN ref** is the float source-network test accuracy *before any TTFS fold*. A
cell is **TRAINABLE** only if the ANN ≫ chance (MNIST chance = 0.1135 const-pred
floor; FMNIST = 0.10). gap = ANN − deployed.

### TRAINABLE rungs (ANN ≫ chance) — the only firing-gain evidence

| dataset | d | w | sched | deployed (mean±sd) | ANN ref | gap (pp) |
|:--------|--:|---:|:-----|:-------------------|:--------|---------:|
| mnist  | 4 | 64 | cascaded     | **0.9267 ± 3.25pp** | 0.9817 | 5.50 |
| mnist  | 4 | 64 | synchronized | **0.9699 ± 0.12pp** | 0.9806 | 1.07 |
| mnist  | 6 | 64 | cascaded     | **0.9400 ± 1.32pp** | 0.9795 | 3.95 |
| mnist  | 6 | 64 | synchronized | **0.9640 ± 0.17pp** | 0.9792 | 1.52 |
| mnist  | 8 | 64 | cascaded     | **0.8717 ± 1.53pp** | 0.9794 | 10.78 |
| mnist  | 8 | 64 | synchronized | **0.9644 ± 0.24pp** | 0.9792 | 1.48 |
| fmnist | 4 | 64 | cascaded     | 0.8450 | 0.8878 | 4.28 |
| fmnist | 4 | 64 | synchronized | 0.8705 | 0.8885 | 1.80 |
| fmnist | 8 | 64 | cascaded     | 0.7250 | 0.8846 | 15.96 |
| fmnist | 8 | 64 | synchronized | 0.8574 | 0.8835 | 2.61 |

95% Student-t CIs (n=3, t=4.303) on the MNIST cascaded rungs: d4 ±8.08pp,
d6 ±3.29pp, d8 ±3.79pp; synchronized: d4 ±0.29pp, d6 ±0.42pp, d8 ±0.59pp.

### CONFOUNDED rungs (ANN ≈ chance) — training-floor, NOT firing-gain

| dataset | d | w | sched | deployed | ANN ref | note |
|:--------|--:|---:|:-----|:---------|:--------|:-----|
| mnist  | 12 | 64  | cascaded / sync     | 0.100 / 0.1135 | **0.1135** | ANN never trained |
| mnist  | 16 | 64  | cascaded / sync     | 0.100 / 0.1135 | **0.1135** | ANN never trained |
| mnist  | 32 | 64  | cascaded / sync     | 0.100 / 0.1135 | **0.1135** | ANN never trained |
| mnist  | 16 | **128** | cascaded / sync | 0.100 / 0.1135 | **0.1135** | **wider DIDN'T rescue** |
| mnist  | 24 | **128** | cascaded / sync | 0.100 / 0.1135 | **0.1135** | **wider DIDN'T rescue** |
| fmnist | 16 | 64  | cascaded / sync     | 0.090 / 0.100  | **0.100**  | ANN never trained |
| fmnist | 32 | 64  | cascaded / sync     | 0.100 / 0.100  | **0.100**  | ANN never trained |

At every confounded rung the ANN's **training** accuracy also peaks at chance
(MNIST trainAcc max 0.1128, val peak 0.156; FMNIST trainAcc max 0.10) — the
source network never learned. Deployed ≈ chance for *both* modes there because
the ANN is at chance, **not** because of anything in the TTFS fold. These are
excluded from every cascaded-vs-sync claim.

---

## 3. The law — and its honest, bounded scope

**On the trainable ladder (MNIST d4/d6/d8 @ w64; corroborated by FMNIST d4/d8):**

1. **Synchronized holds near the ANN ceiling, depth-stable and tight.** MNIST
   sync deploys 0.9699 → 0.9640 → 0.9644 across d4→d6→d8 — essentially flat,
   within ~1.5pp of the float ANN (~0.979) at every depth, seed sd ≤ 0.24pp.
   FMNIST sync stays within ~2.6pp of its ANN at d8. **Verdict
   `synchronized_holds_near_ann` confirmed on all 5 trainable cells.**

2. **Cascaded is worse, depth-degrading by the d4→d8 endpoints, and noisy.**
   MNIST cascaded 0.9267 (d4) → 0.8717 (d8): the ANN-gap grows 5.5pp → 10.8pp and
   the cascaded→sync gap grows 4.3pp → 9.3pp d4→d8. FMNIST cascaded falls harder,
   0.845 → 0.725 (gap to ANN 4.3pp → 16.0pp d4→d8). Cascaded seed sd reaches
   3.25pp (vs sync's 0.24pp) — a high-variance, fragile code consistent with the
   prior "death-cascade / greedy partial-sum firing-gain deficit" framing.
   **Verdict `cascaded_firing_gain_degraded` on all 5 trainable cells.**

**HONESTY — the densified midpoint refutes a *clean monotonic* law.** The round-2
d6 rung does **not** sit on a smooth monotone line:

- MNIST cascaded **mean is non-monotonic**: d4 0.9267 → d6 **0.9400** → d8 0.8717.
  d6 is *higher* than d4 (within overlapping seed CIs), then d8 drops sharply.
- The cascaded→sync **gap is non-monotonic**: +4.32pp (d4) → +2.40pp (d6) →
  +9.27pp (d8). It **narrows at d6 before blowing out at d8** — it does *not*
  widen smoothly.

So the defensible statement is **bounded**: by the *endpoints* d4→d8 the cascaded
mode is clearly worse and the gap clearly larger, and the d8 collapse (and the
even sharper FMNIST d8 collapse) is real and reproduced across 3 seeds and 2
datasets. But "cascaded falls **monotonically** with depth and the gap **widens
monotonically**" — the literal round-1 ledger phrasing — is **not** supported once
d6 is inserted. The law is: *cascaded degrades with depth and is strictly
dominated by synchronized on every trainable rung*, with a sharp, seed-stable
collapse appearing by d8. Number of clean trainable depths: **3** (MNIST d4/d6/d8),
**+2** corroborating FMNIST points (d4/d8). Maximum trainable depth reached: **8**.

**THE SCOPE LIMIT (the headline confound).** The plain `Linear+ReLU` `deep_mlp`
is **ANN-untrained at depth ≥ 12**. Round-2's explicit deep-rescue bet — widen to
**w128** so the stack trains deeper — **failed**: d16w128 and d24w128 ANN are
*still* pinned at chance (0.1135, trainAcc 0.1128). Width did not break the
training floor. Therefore:

> **The depth × firing-gain law is established only for d ≤ 8 (trainable band).
> No conclusion — neither a deeper collapse law nor a gate-fix verdict — can be
> drawn for d ≥ 12 from this campaign, because no trainable network exists there.
> The deep firing-gain test is GATED on WS2 (a residual/norm backbone that lets
> the ANN train past d ≈ 8).**

This is exactly the third branch of the round-1 pre-registered decision rule
(d16/d24 @ w128 also hit chance → bound the law to d ≤ 8 until a backbone is
added). It is a *real, bounded result*, not a failed one: we now know (a) the
firing-gain law holds and is strong over d4–d8, and (b) the plain-MLP backbone —
not the firing-gain mechanism — is what caps the depth at which the question can
even be asked.

---

## 4. Recommendation

### 4.1 Is synchronized the deep-model default? — **Yes, on the available evidence.**

> **For `deep_mlp` deployment at any trainable depth, default to the synchronized
> schedule.**

- Synchronized is **near-lossless and depth-stable** through the deepest
  *trainable* rung (MNIST d8 0.9644 ±0.24pp, ~1.5pp from ANN; FMNIST d8 0.857,
  ~2.6pp from ANN) and near-deterministic across seeds.
- Cascaded is **strictly dominated at every trainable depth** — lower mean
  *and* far higher seed variance (sd up to 3.25pp), with a sharp d8 collapse on
  both datasets. Its only standing rationale is single-spike traffic economy, not
  accuracy; nothing here suggests that economy is worth a 9–16pp, high-variance
  hit by d8.
- This call is **MNIST/FMNIST-, w64-, d ≤ 8-scoped.** It does not extend a
  recommendation to d ≥ 12 — there is no trainable network there to recommend
  anything for.

### 4.2 Is the θ-cotrain (per-channel firing-gain) gate-fix worth testing on the trainable cascaded rungs? — **Yes, this is the one scoped, justified next experiment.**

The gate-fix (`ttfs_theta_cotrain` / `ttfs_staircase_ste` / per-channel
`per_source_scales`, all merged and present in `src/`; the coalescing-capability
gate fix is in `main` at commit `2b8dc2f`) is the per-channel knob that prior
toy/harness work localized as able to *recover* the cascaded collapse:

- Oracle per-layer θ recovered a d=3 toy cascade 0.074 → 0.909 (≈ staircase
  ceiling) — the lost accuracy is **recoverable by a gain knob**, not gone
  (artifacts `docs/research_artifacts_for_cascaded_ttfs_tuning/50–52`).
- A hedged staircase-backward STE reached d=9, S=32 ≈ 0.966 on the toy harness,
  halving the deep-cascade gap.

**The trainable cascaded rungs are now the right, in-scope place to test it.** We
have a *reproduced, trainable, degrading-but-not-dead* cascaded signature exactly
where the toy result says the knob applies: MNIST d8 cascaded sits at 0.8717
(neurons firing but mis-gained — it is 0.87, not chance — i.e. a non-dead,
gradient-bearing firing-gain deficit), with a clean 10.8pp gap to the ANN to
close. **Recommended scoped trial:** turn the θ-cotrain gate-fix on for MNIST
(and FMNIST) **d6/d8 cascaded @ w64, 3 seeds**, and ask whether the per-channel
firing-gain knob arrests the collapse — specifically whether cascaded d8 recovers
from 0.872 toward synchronized's 0.964.

**Bounded expectations (do not oversell):** the knob's prior plateau was
~0.95–0.966, **still ≤ synchronized's 0.964–0.97**; and naive gradient
θ-calibration *fails on dead neurons*, so the working recipe is value-domain
revive → cascade refine, not flipping one flag. The decision rule:

- If gate-fix **recovers cascaded d8 to ≈ synchronized** (closes most of the
  10.8pp gap) with acceptable cost → the per-channel firing-gain knob is a real
  lever; cascaded becomes viable where its single-spike traffic economy matters.
- If gate-fix **narrows but stays clearly below synchronized** (the toy plateau
  repeats) → synchronized remains the unconditional deep default; the gate-fix is
  a research curiosity, not a product path. Retire cascaded for deep `deep_mlp`.

### 4.3 Link to WS2 — the deep test needs a residual backbone.

The single biggest limitation of WS3 is **not** in the firing-gain mechanism — it
is the **backbone**. A plain Linear+ReLU `deep_mlp` cannot train past d ≈ 8 even
when widened to w128 (round-2 proved this directly: d16/d24 @ w128 ANN = chance).
**WS2** (modern ops: residual → LN → GELU → attention; see
`docs/research/RESEARCH_PROGRAM.md` §D/row "Modern ops (WS2)") owns delivering a
residual/norm backbone whose ANN actually trains at depth. **The deep
(d ≥ 12) cascaded-vs-synchronized firing-gain question is blocked on WS2.** Once
WS2 supplies a trainable deep backbone, re-run the d12/d16/d24 cascaded-vs-sync
sweep (and the §4.2 gate-fix trial) on it — that is the only way to learn whether
the d8 collapse keeps deepening or whether the firing-gain knob holds the
cascade together at genuine depth.

---

## 4b. Architecture × dataset dependence

**Question.** Is the cascaded firing-gain deficit (a) **architecture-dependent**
(small on a CNN, large on a deep MLP) and (b) **dataset-dependent** (does the same
architecture's cascaded gap swing from ~1pp on an easy dataset to many pp on a
harder one)? This section harvests the breadth sweep that holds the spiking knob
(`spiking_mode=ttfs_cycle_based`, `ttfs_cycle_schedule ∈ {cascaded,
synchronized}`) fixed while varying **architecture** (`lenet5` CNN vs `deep_mlp`)
and **dataset** (MNIST / FMNIST / KMNIST / SVHN).

Config sets: `experiments/campaign/sch_lenet_*` (LeNet5 CNN, `model_type=lenet5`,
all four datasets), `experiments/campaign/sch_dmlp_*` (deep_mlp, depth ∈ {4,6,8},
width 64), and `experiments/campaign/ws3cnn_lenet5_*` (a redundant **MNIST-only**
LeNet5 CNN cross-check). Deployed = soft-core spiking-sim metric (LeNet runs use
`max_simulation_samples=50`, so LeNet cascaded deployed values are quantized to
~±2pp — read the gaps, not the third decimal); ANN ref = Pretraining
`target_metric` (== final `Test accuracy`, re-verified).

### Coverage caveat (what is and is not in this batch)

- **deep_mlp:** MNIST and FMNIST are complete (3 seeds × d{4,6,8} × both
  schedules). **deep_mlp KMNIST and SVHN are still PENDING** on the runner
  (only 2 KMNIST-d4 sync seeds finalized; no cascaded KMNIST/SVHN deep_mlp
  deployed yet) — so the *dataset* axis is fully crossed only on **LeNet5**, and
  the *architecture* contrast is anchored at the deepest trainable deep_mlp rung
  **d8** vs the LeNet5 CNN.
- **LeNet5 SVHN cascaded — all 3 seeds FAILED (non-finalized).** The run crashed
  at **Soft Core Mapping** (stuck `running`, rc=1). The pre-failure trace is still
  informative: the ANN trained to ~0.896 and the cascaded **TTFS fine-tune already
  collapsed it to ~0.674** before the crash, vs synchronized's finalized 0.8605.
  The ~19pp number below is therefore a *flagged lower-confidence* cell (partial
  deployed_acc, not a clean SCM finalize), but the direction — a severe cascaded
  collapse on the hardest dataset — is unambiguous.
- **ANN-untrained confound: NONE here.** Unlike the deep_mlp d≥12 rungs of §2,
  *every* cell in this batch has an ANN well above its chance floor (MNIST 0.99,
  FMNIST 0.88–0.92, KMNIST 0.90–0.96, SVHN 0.895 ≫ SVHN chance ≈ 0.196). So every
  cascaded→sync gap below is genuine firing-gain signal, not a training-floor
  artifact.

### The table — deployed (3-seed mean ± population sd) and the cascaded→sync GAP

ANN ref shown is the synchronized cell's (cascaded/sync ANN agree to <0.4pp).
`casc→sync GAP` = sync deployed − cascaded deployed (pp). `ANN-gap(casc)` = ANN −
cascaded deployed (pp).

| model | dataset | d | sched | deployed (mean±sd) | ANN ref | ANN-gap(casc) | **casc→sync GAP** |
|:------|:--------|:--|:------|:-------------------|:--------|--------------:|------------------:|
| lenet5 (CNN) | mnist  | – | cascaded     | **0.9800 ± 0.00pp** | 0.9916 | 1.03 | — |
| lenet5 (CNN) | mnist  | – | synchronized | **0.9891 ± 0.10pp** | 0.9916 | — | **+0.91** |
| lenet5 (CNN) | fmnist | – | cascaded     | **0.8400 ± 1.63pp** | 0.9189 | 7.86 | — |
| lenet5 (CNN) | fmnist | – | synchronized | **0.8999 ± 0.05pp** | 0.9189 | — | **+5.99** |
| lenet5 (CNN) | kmnist | – | cascaded     | **0.8933 ± 1.89pp** | 0.9577 | 6.80 | — |
| lenet5 (CNN) | kmnist | – | synchronized | **0.9485 ± 0.39pp** | 0.9577 | — | **+5.52** |
| lenet5 (CNN) | svhn †FAILED | – | cascaded | **0.6698 ± 1.08pp** | 0.8946 | 22.58 | — |
| lenet5 (CNN) | svhn   | – | synchronized | **0.8605 ± 0.53pp** | 0.8946 | — | **+19.07** † |
| deep_mlp | mnist  | 8 | cascaded     | **0.8817 ± 1.31pp** | 0.9783 | 9.63 | — |
| deep_mlp | mnist  | 8 | synchronized | **0.9616 ± 0.24pp** | 0.9783 | — | **+7.99** |
| deep_mlp | fmnist | 8 | cascaded     | **0.7000 ± 4.32pp** | 0.8831 | 18.33 | — |
| deep_mlp | fmnist | 8 | synchronized | **0.8571 ± 0.26pp** | 0.8831 | — | **+15.71** |

† LeNet5 SVHN cascaded is a **non-finalized** cell (crashed at Soft Core Mapping);
the value is the pre-crash collapse and the 19.07pp gap is flagged
lower-confidence (see coverage caveat). It is recorded as
`cascaded_run_finalized=false` in the ledger.

deep_mlp shallower rungs for context (full per-depth ladder in §2/§3): MNIST
casc→sync gap +1.68pp (d4) → −0.29pp (d6) → +7.99pp (d8); FMNIST +1.46pp (d4) →
+6.32pp (d6) → +15.71pp (d8). The CNN MNIST cross-check `ws3cnn_lenet5` reproduces
`sch_lenet` MNIST **byte-for-byte** (casc 0.9800 / sync 0.9891, gap 0.91pp).

### Answer to the KEY question — BOTH, with dataset as the dominant axis

**(a) Architecture-dependent? — YES, at fixed dataset.** Hold the dataset and swap
architecture: on **MNIST**, the LeNet5 CNN cascaded gap is **0.91pp**
(near-lossless) while the deep_mlp d8 cascaded gap is **7.99pp** — an ~8× larger
deficit on the deep MLP. On **FMNIST**, CNN 5.99pp vs deep_mlp d8 15.71pp (~2.6×).
The CNN's shallow, conv-shared, residual-free-but-pooled structure suffers the
death-cascade far less than the 8-deep plain Linear+ReLU stack — consistent with
the depth law of §3 (the deficit is driven by the length of the greedy
partial-sum chain, and the CNN has fewer cascaded fully-connected hops).

**(b) Dataset-dependent? — YES, and this is the stronger axis.** Hold the
architecture (LeNet5 CNN) and swap dataset: the cascaded→sync gap sweeps
**0.91pp (MNIST) → 5.52pp (KMNIST) → 5.99pp (FMNIST) → 19.07pp (SVHN†)**. The
prompt's pre-registered hypothesis — *"LeNet cascaded gap ~0.9pp on MNIST but
~6pp on FMNIST"* — **lands essentially exactly** (0.91pp vs 5.99pp). The gap
tracks task difficulty / margin tightness: on MNIST the ANN sits at 0.99 with
huge logit margins that survive single-spike timing distortion; on SVHN (natural
images, tight 0.895 margins) the same cascaded timing-gain deficit shreds the
decision boundary (and additionally trips the deployment SCM gate). Same code,
same architecture — the dataset's margin structure decides whether cascaded is
near-lossless or catastrophic.

**Interaction.** The two axes compound: the worst cell is the *deep* MLP on a
*harder* dataset (deep_mlp FMNIST d8, 15.71pp), the best is the *shallow* CNN on
the *easiest* dataset (LeNet MNIST, 0.91pp). There is no architecture that is
universally safe (the CNN is fine on MNIST but loses ~6pp on FMNIST/KMNIST and
~19pp on SVHN) and no dataset that is universally safe for cascaded.

### Per-cell verdicts (ledger `kind="arch_dataset"`)

- `cascaded_near_lossless_on_cell` — **1** (lenet5 mnist, 0.91pp).
- `cascaded_firing_gain_degraded` (2–8pp) — **3** (lenet5 fmnist 5.99pp, lenet5
  kmnist 5.52pp, deep_mlp mnist d8 7.99pp).
- `cascaded_firing_gain_collapse` (>8pp) — **2** (deep_mlp fmnist d8 15.71pp;
  lenet5 svhn 19.07pp †non-finalized).

### Implication

Synchronized remains the **unconditional** schedule recommendation across **every
finalized cell** of this breadth sweep — it holds within ≤3.4pp of the ANN on all
four datasets and both architectures, with seed sd ≤0.53pp. Cascaded is only
near-lossless in the *one* easy corner (CNN × MNIST); everywhere else it costs
5.5–19pp and is high-variance. The breadth result **strengthens** the §4.1 call:
default to synchronized regardless of architecture or dataset. The
architecture-and-dataset structure of the deficit (worse with depth, worse with
task difficulty) is exactly the per-channel firing-gain signature the §4.2
gate-fix targets — but the gate-fix's prior plateau (~0.95–0.966) sits *below*
synchronized on the harder cells here, so cascaded's only standing rationale stays
single-spike traffic economy, not accuracy.

## 4c. The cascade collapse is MLP-architecture-specific — `deep_cnn` d4 carries NO deficit (2026-06-24)

**Question (the architecture axis, sharpened).** §4b's CNN row was the *shallow*
LeNet5. Does a **trainable, genuinely deep CNN** (`deep_cnn`, the same conv-stack
family used for the deep_mlp depth ladder) show the cascaded death-cascade that
deep_mlp does at the same depth? This isolates *architecture* at fixed depth (d4)
and fixed dataset (MNIST), against the deep_mlp baseline where d4 cascaded already
costs 4.3pp and d8 costs 9.3pp.

Runs: `dcnn_d4_{cascaded,synchronized}_s{0,1,2}` (6 runs, all `returncode==0`,
3 seeds/arm, `max_simulation_samples=200`). Ledger: `cluster:"WS3"`,
`kind:"arch_dataset"`, `model:"deep_cnn"`.

| model | dataset | d | sched | deployed (3-seed mean) | ANN ref | casc→sync GAP | ANN-gap(casc) |
|:------|:--------|:--|:------|:-----------------------|:--------|--------------:|--------------:|
| deep_cnn | mnist | 4 | cascaded     | **0.9883** (.97/.995/1.0) | 0.9931 | — | 0.47 |
| deep_cnn | mnist | 4 | synchronized | **0.9898** (.9911/.9865/.9918) | 0.9909 | **−0.15** | 0.11 |
| *(ref)* deep_mlp | mnist | 4 | cascaded | 0.9267 | 0.9817 | +4.32 | 5.50 |
| *(ref)* deep_mlp | mnist | 8 | cascaded | 0.8717 | 0.9783 | +9.27 | 9.63 |

**Verdict — SUPPORTED (collapse is MLP-specific), but BOUNDED.** On the trainable
`deep_cnn d4` the cascaded→sync gap is **−0.15pp** (cascaded is statistically *equal*
to synchronized, not worse) and cascaded tracks its 0.9931 ANN reference within
**0.47pp** — there is **no firing-gain death-cascade**, in sharp contrast to deep_mlp
where d4 cascaded already shows 4.3pp widening to 9.3pp at d8. ANN refs 0.989–0.995
(≫ 0.10 chance) confirm this is a genuine firing-gain comparison, not an
untrained-floor artifact. Together with §4b (shallow LeNet5 MNIST cascaded 0.91pp,
near-lossless) this **strengthens** the claim that the death-cascade is a property of
the deep plain-Linear+ReLU MLP stack, not a depth-universal law.

**Confounds / bounds.** (1) `max_simulation_samples=200` → read the ~0pp gap, *not*
the third decimals: cascaded s0=0.97 (≈6/200 misses) and s2=1.0 are small-N
variance that inflates the cascaded per-seed spread (0.97–1.0). (2) **The within-CNN
depth ladder cannot be closed in this batch — every `deep_cnn d6` run failed (rc=1,
a SANA-FE / mapping compile crash).** So the actual analogue of the MLP d4→d8
widening (does a *deeper* CNN eventually cascade?) is **untested**; absence of a gap
at d4 alone does not prove absence of a deeper cascade. The deep_mlp comparison
numbers are carried from §2/§3, not recomputed here. **Next:** re-attempt the
`deep_cnn` d5/d6/d7 rungs with SANA-FE/Loihi sim **off** (the crash surface) to fit a
within-CNN depth law and confirm/deny a deeper CNN cascade (backlog `plan_stage:4`).

---

## 5. Ledger

One verdict record per (dataset, depth, width, schedule) cell — **24 cells** —
appended to `runs/campaign/ledger.jsonl`, cluster `WS3`,
`kind="depth_firing_gain_final"`, each carrying `deployed_acc_mean/std`,
`ann_test_acc_mean/std`, `gap_pp`, `trainable`, `run_ids`. Verdict tally:

- `cascaded_firing_gain_degraded` — **5** (mnist d4/d6/d8, fmnist d4/d8 cascaded).
- `synchronized_holds_near_ann` — **5** (mnist d4/d6/d8, fmnist d4/d8 sync).
- `training_floor_confound` — **14** (mnist d12/d16/d32 @ w64, d16/d24 @ w128;
  fmnist d16/d32 @ w64; both schedules each).

Plus **6** breadth records (`kind="arch_dataset"`, §4b), one per (model,dataset)
cell, each carrying `cascaded/synchronized_deployed_mean/std_pp`,
`ann_test_acc_mean`, `cascaded_to_sync_gap_pp`, `cascaded_run_finalized`, and
`run_ids`: `cascaded_near_lossless_on_cell` ×1 (lenet5 mnist),
`cascaded_firing_gain_degraded` ×3 (lenet5 fmnist/kmnist, deep_mlp mnist d8),
`cascaded_firing_gain_collapse` ×2 (deep_mlp fmnist d8; lenet5 svhn †non-finalized).

---

## 6. One-line takeaway

Over the trainable depth band (MNIST/FMNIST, d ≤ 8, w64), synchronized `deep_mlp`
deploys **near-lossless and depth-stable** (MNIST d8 0.9644 ±0.24pp) while
cascaded is **strictly dominated, degrading and noisy** (MNIST d8 0.8717, FMNIST
d8 0.725; d4→d8 cascaded→sync gap 4.3→9.3pp, though the d6 midpoint refutes a
*clean* monotone) — so **default deep models to synchronized now**; the round-2
deep-rescue at w128 **failed to train** (d16/d24 @ w128 ANN = chance), so the
deeper firing-gain law is **bounded to d ≤ 8 and gated on WS2's residual
backbone**; meanwhile the merged per-channel θ-cotrain gate-fix is **worth a
scoped trial on the trainable d6/d8 cascaded rungs** — the one place we have a
reproduced, non-dead, recoverable firing-gain deficit to fix.
