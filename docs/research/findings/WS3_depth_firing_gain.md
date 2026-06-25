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

### 4c.1 The no-collapse law extends ONE RUNG DEEPER — `deep_cnn` d5 (closes the d4→d5 cell) (2026-06-24)

The §4c batch (`plan_stage:4`) returned its first deeper rung: **`deep_cnn` d5**
(the d6/d7 rungs of the same grid are still pending/crashing). 6 runs
(`pdcnnladder_d5_{cascaded,synchronized}_s{0,1,2}`), all `rc=0`, 3 seeds/arm, paired
by seed, `max_simulation_samples=200`. Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`,
`model:"deep_cnn"`, `depth:5`.

| model | dataset | d | sched | deployed (3-seed mean) | ANN ref | casc→sync GAP | ANN-gap(casc) |
|:------|:--------|:--|:------|:-----------------------|:--------|--------------:|--------------:|
| deep_cnn | mnist | 5 | cascaded     | **0.9917** (.99/.99/.995)   | 0.9913 | — | −0.04 |
| deep_cnn | mnist | 5 | synchronized | **0.9924** (.9918/.9911/.9942) | 0.9937 | **+0.07** | — |
| *(ref, §4c)* deep_cnn | mnist | 4 | cascaded vs sync | 0.9883 vs 0.9898 | ~0.992 | **−0.15** | 0.47 |

**Verdict — `no_cascade_collapse`, the no-collapse law extends d4→d5.** On the
trainable `deep_cnn d5` the cascaded→sync gap is **+0.07pp** (within ~1.2pp seed sd:
cascaded sd 0.24pp, sync sd 0.13pp), and cascaded tracks its **own** 0.9913 ANN to
**−0.04pp**. Together with §4c's d4 (−0.15pp) the within-CNN gap is **flat / near-zero
at both rungs** — there is **no within-CNN death-cascade through d5**, in sharp
contrast to deep_mlp (d4 +4.3pp → d8 +9.3pp). All 6 ANN refs ~0.99 (≫ MNIST chance
0.1135), so this is a genuine firing-gain comparison, not an untrained-floor artifact.
This **closes the d4→d5 cell §4c flagged open**.

**Confounds / bounds.** (1) `max_simulation_samples=200` → read the ~0pp gap, *not*
the third decimals: cascaded s2=1.0 vs s0=s1=0.99 is small-N variance inflating the
cascaded per-seed spread (0.99–0.995). (2) **d6/d7 remain untested** — the prior
`deep_cnn d6` crashed `rc=1` (soft-core mapping compile) and only d5 of the
`plan_stage:4` grid finalized this round, so absence of a gap *through d5* still does
**not** prove absence of a *deeper* within-CNN cascade. (3) the d4 numbers are carried
from §4c, not recomputed. **Next:** land the d6/d7 rungs (re-attempt the mapping-compile
crash surface) to push the within-CNN ladder past d5, and the paired deeper-convnet
`plan_stage:9` (d8/d10 @ n=1000) to stress the cascade at genuine convnet depth.

---

## 4d. The lenet5 cascade gap at n=1000 — MNIST washes to noise, FashionMNIST is small-and-real (2026-06-24)

**Question (the resolution axis).** §4b read the LeNet5 cascade gap off n=50
(`sync_full` tag, 2-decimal-rounded; the flat MNIST cascaded `[.98,.98,.98]` was
suspicious). Does the cascaded gap survive a paired **n=1000** re-measure on the
**VALID on-chip-majority `lenet5`** (vs the retired host-majority deep_mlp)?

Runs: `csr_lenet_{MNIST,FashionMNIST}_DataProvider_cascaded_n1000_s{0,1,2}` (6 runs,
all `rc=0`, `artifact_ok`, not timed out, 3 seeds, `max_simulation_samples=1000`,
`ttfs_cycle_based`). Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`.

| dataset | cascaded n=1000 (3-seed mean ± sd_pp) | n=50 sync_full baseline | casc→sync GAP | ANN ref | n50→n1000 casc shift | verdict |
|:--------|:--------------------------------------|:------------------------|--------------:|--------:|---------------------:|:--------|
| mnist   | **0.9873** (.988/.990/.984; ±0.31) | 0.9891 | **+0.18** | 0.9912 | **+0.73** | washes to noise |
| fmnist  | **0.8397** (.851/.831/.837; ±1.03) | 0.8999 | **+6.02** | 0.9183 | **−0.03** | small-and-real, hardens |

**Verdict — MIXED, and it sharpens §4b/§4c.** On **MNIST/lenet5** the cascade gap is
**+0.18pp < seed std 0.31pp** — the flat n=50 `[.98,.98,.98]` was 50-sample rounding
noise that lifted **+0.73pp** to 0.987 at n=1000, washing the apparent ~0.9pp
`cnn_mode_compare` gap to noise (cascaded tracks the 0.9912 ANN within 0.39pp). On
**FashionMNIST/lenet5** the gap is **+6.02pp ≫ seed std 1.03pp** and barely moved
(**−0.03pp**) from the n=50 baseline `[.86,.84,.82]` — a **real firing-gain residual on
a VALID CNN vehicle**. This **hardens the architecture×dataset-dependence** of the
death cascade (closeout S6 depth-risk): the cascade is *not* uniformly MLP-specific —
a valid shallow CNN carries a real, dataset-gated deficit on the harder dataset while
being lossless on the easier one.

**Confounds / bounds.** (1) The synchronized arm is the **recorded n=50 `sync_full`
tag, NOT a paired n=1000 synchronized re-run** (so `synchronized_run_ids` is empty and
the casc→sync gap mixes n1000-cascaded against n50-synchronized); the cleanest
*within-arm* cross-resolution check is the cascaded n50→n1000 shift quoted above. (2)
n=50 cascaded baselines are 2-decimal-rounded — read them as gaps, not third decimals.
(3) No confound on the cascaded n=1000 runs themselves: all rc=0, artifact_ok, 3 seeds,
ANN refs ≫ chance (not untrained). **Next:** a paired n=1000 **synchronized** lenet5
re-run on both datasets would close the only remaining mixed-resolution confound
(backlog `plan_stage:5`).

### 4d.1 The lenet5 KMNIST cell at n=1000 — MILD and dataset-stable, completing the 4-dataset CNN table (2026-06-24)

§4d re-measured MNIST/FMNIST at n=1000; this adds the **KMNIST** cell, completing the
4-dataset cascaded CNN table on the **VALID `lenet5`** vehicle. Runs:
`csr_lenet_KMNIST_DataProvider_cascaded_n1000_s{0,1,2}` (3 runs, all `rc=0`,
artifact_ok, not timed out, `max_simulation_samples=1000`, `ttfs_cycle_based`). Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`, `dataset:"kmnist"`.

| dataset | cascaded n=1000 (3-seed mean ± sd_pp) | synchronized (n=50) | casc→sync GAP | ANN ref | ANN-gap(casc) | round-1 n=50 casc | verdict |
|:--------|:--------------------------------------|:--------------------|--------------:|--------:|--------------:|:------------------|:--------|
| kmnist  | **0.934** (.924/.937/.941; ±0.73) | 0.9485 (±0.39) | **+1.45** | 0.9646 | 3.06 | 0.8933 (.92/.88/.88) | MILD, dataset-stable |

**Verdict — `cascaded_mild_dataset_stable`.** KMNIST cascaded 0.934 sits **1.45pp**
below synchronized — a **MILD** gap that places KMNIST squarely **between MNIST
(0.18pp, lossless, §4d) and FMNIST (6.02pp, §4d)** on the dataset-hardness axis. The
n=1000 cascaded mean **LIFTS +4.07pp** over the round-1 n=50 cascaded mean (0.8933),
collapsing the round-1 5.52pp gap (§4b) to 1.45pp — **most of the n=50 KMNIST gap was
2-decimal/50-sample subsample quantization**, the same washout §4d found for MNIST.
A residual **cascaded→ANN gap of 3.06pp > seed sd 0.73pp** remains, so a
*small-but-real* firing-gain residual persists (not pure noise). ANN ~0.965 (≫ KMNIST
chance 0.10) ⇒ genuine firing-gain measurement.

**Confounds / bounds.** (1) The synchronized arm is the **finalized n=50
`sch_lenet_KMNIST_synchronized` run** (`max_simulation_samples=50`; no paired n=1000
sync re-run), so the 1.45pp gap mixes **n1000-cascaded vs n50-synchronized** — read
gaps, not third decimals, on the sync arm (n=50 → 0.02 granularity). Unlike the §4d
MNIST/FMNIST rows (which left `synchronized_run_ids` empty), the finalized n=50 sync
runs exist here so they are paired/populated. (2) the **companion SVHN cascaded@n1000
arm all failed `rc=1`** → not harvestable this round; the SVHN cell remains the round-1
non-finalized row (§4b †). (3) depth-axis stress is modest (lenet5 IR max-latency ~3 /
2 neural segments), as for the §4b/§4e cnn cells. **Next:** paired n=1000 synchronized
KMNIST/SVHN (backlog `plan_stage:10`) to de-confound the resolution mix and to recover
the crashed SVHN arm.

---

## 4e. Matched-ANN cascaded-vs-synchronized on lenet5 — full-test SCM gap is 0.56pp, NOT a death-cascade (2026-06-24)

A **paired** cascaded-vs-synchronized run on a well-trained LeNet-5 (the §4d
n=1000 arm only had a cascaded leg). 6 runs, 3 seeds/mode, **matched** (lenet5,
MNIST, `ttfs_cycle_based`, S=4, 5 trainable layers / IR max-latency 3 / 2 neural
segments). All rc=0; ANN well-trained (~0.991, far from chance), so this is a
**genuine firing-gain vehicle**, not the invalid untrained deep_mlp.

**Read the apples-to-apples primary metric — the full-test-set SCM identity-mapped
accuracy** (`"Soft-core (identity-mapped) Spiking Simulation Test"`), computed
identically by **both** pipelines on all 10000 test samples:

| metric | cascaded (3-seed) | synchronized (3-seed) | cascaded→sync gap |
|:-------|:------------------|:----------------------|------------------:|
| **full-test SCM identity** | **0.9835** (0.9846/0.9828/0.9830) | **0.9891** (0.9898/0.9906/0.9869) | **0.56pp** |
| ANN ref | 0.9913 (0.9919/0.9909/0.9909) | 0.9913 (0.9932/0.9912/0.9896) | — |
| deployed→ANN gap | +0.78pp | +0.22pp | — |
| raw `__target_metric.json` (CONFOUNDED) | 0.98/1.0/0.96 (n=50 nevresim) | 0.9898/0.9906/0.9869 (full-test SCM) | 0.91pp (do NOT cite) |

**Verdict — FIRING-GAIN RISK NOT OBSERVED on a valid trainable CNN.** Cascaded
single-spike TTFS deploys **near-losslessly** at 0.9835 full-test SCM, only
**0.56pp** below the synchronized baseline at matched ANN, and within 0.78pp of
its own ANN. Cascaded full-test **SCM == HCM (0.9846, s0)**, so the mapping is
lossless and the sub-pp loss is **mode-intrinsic**. This answers the closeout-v2
§6 depth/firing-gain risk on a **real convnet**: no depth-driven death-cascade
collapse — the deep_mlp panic was an artifact of an untrained/invalid
host-majority vehicle.

**Confounds.** (1) **Asymmetric metric provenance** — the bare
`__target_metric.json` floats are *not* the same quantity across modes: cascaded's
(0.98/1.0/0.96, exact 1/50 multiples) is a genuine cascaded nevresim Simulation on
only **50/10000 subsampled** samples, whereas synchronized's is the full-test SCM
value. The naïve raw-target gap (0.91pp) is dominated by ±2% subsample
quantization and is **not load-bearing**; use the full-test SCM gap (0.56pp). (2)
`max_simulation_samples=50` on the genuine nevresim sim — read gaps, not third
decimals. (3) **Modest depth axis** — 5 trainable layers but IR max-latency=3 / 2
neural segments, so the depth stress is mild; a *deeper* convnet (backlog
`plan_stage:6`) would test the depth axis harder. Ledger: `cluster:"WS3"`,
`kind:"cnn_mode_compare"`, run ids `ws3cnn_lenet5_{cascaded,synchronized}_s{0,1,2}`.

---

## 4f. The within-CNN death-cascade DOES appear with depth — the no-collapse law breaks at d6 and the gap blows out by d10/d12 (2026-06-24)

**Question (the within-CNN depth law, closed).** §4c/§4c.1 found the cascaded→sync gap
*flat/near-zero* on `deep_cnn` through d4 (−0.15pp) and d5 (+0.07pp) and flagged the
deeper rungs open (the d6 retry kept crashing). This batch lands the full deeper ladder
— **d6, d8** (`dcnn_depth_cascade`, `plan_stage` legacy grid) **and d10, d12**
(`plan_dcnn_deep`, `plan_stage:2`) — and answers it: **a deeper CNN *does* eventually
cascade.** The §4c "no within-CNN cascade" reading was a *shallow-depth* artifact, not a
depth-universal CNN immunity.

All cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based`, S=4, 3 seeds/arm paired by
seed, `max_simulation_samples=200`. Ledger: `cluster:"WS3"`, `kind:"depth_firing_gain"`,
`model:"deep_cnn"`.

### The full within-CNN depth ladder — cascaded vs synchronized

| d | cascaded deployed (3-seed mean) | synchronized deployed (3-seed mean) | **casc→sync GAP** | ANN ref | casc→ANN gap | verdict |
|--:|:--------------------------------|:------------------------------------|------------------:|:--------|-------------:|:--------|
| 4  | 0.9883 (§4c)                | 0.9898 (§4c)                | **−0.15** | ~0.992 | 0.47 | no collapse |
| 5  | 0.9917 (§4c.1)              | 0.9924 (§4c.1)              | **+0.07** | 0.9913 | −0.04 | no collapse |
| 6  | **0.9664** (.976/.977/.946) | **0.9913** (.9918/.9892/.9928) | **+2.49** | 0.9917 | 2.69 | gap emerges (mild) |
| 8  | **0.9564** (.964/.965/.940) | **0.9904** (.9887/.9901/.9923) | **+3.40** | 0.9928 | 3.80 | widens (mild) |
| 10 | **0.8531** (.928/.856/.775) | **0.9917** (.9917/.9916/.9917) | **+13.86** | 0.9897 | 13.66 | **death-cascade** |
| 12 | **0.8780** (.884/.848/.902) | **0.9923** (.9919/.9916/.9934) | **+11.43** | 0.9921 | 11.41 | **death-cascade** |
| *(ref)* deep_mlp d8 | 0.8717 | 0.9644 | +9.27 | 0.9783 | 9.63 | death-cascade (INVALID host-majority) |

**Verdict — the depth × firing-gain risk (closeout-v2 §6) is REAL on a VALID CNN
vehicle.** The cascaded→sync gap **widens monotonically** d4→d5→d6→d8→d10
(−0.15 → +0.07 → +2.49 → +3.40 → +13.86pp): flat through d5, *opening* by d6, *mild*
through d8, then a **sharp collapse by d10** where cascaded drops to 0.853 with high
seed variance (0.775–0.928) while synchronized stays pinned at the 0.9917 ANN ceiling
(−0.20pp). d12 confirms it (cascaded 0.878, gap +11.43pp). Every cell's ANN is
well-trained (~0.99 ≫ MNIST chance 0.1135) so **this is a genuine firing-gain
result, not an untrained-floor artifact** — the §2 deep_mlp d≥12 training-floor
confound does *not* apply here (the `deep_cnn` backbone trains cleanly at every depth).
Synchronized holds the ANN ceiling at **every** depth (gap to ANN ≤0.20pp, seed sd
≤0.15pp) — it is the unconditional deep default on the CNN exactly as on the MLP.

**This corrects the §4c/§4c.1 bound.** "The death-cascade is MLP-architecture-specific;
`deep_cnn` carries no deficit" held only because §4c/§4c.1 reached *only* d4/d5. With the
ladder closed, the honest law is: **the cascade is depth-driven on the CNN too, just with
a higher onset depth than the MLP** — the deep_mlp shows +4.3pp already at d4 and +9.3pp
at d8, whereas the CNN stays <0.1pp through d5, is still only +3.4pp at d8, and reaches a
comparable collapse (+11–14pp) only at d10–d12. The conv-shared/pooled structure *delays*
but does **not** abolish the death-cascade; the deficit tracks the length of the greedy
single-spike partial-sum chain, and the CNN simply needs more layers to build a chain as
long as the d8 plain-Linear+ReLU stack.

**Confounds / bounds.** (1) **DOMINANT (validity):** *every* queue-recorded run finalized
with `returncode==1` and lives in `runs/campaign/q/failed/`, **none in `done/`** — by the
strict `returncode==0` rule **zero runs are formally valid**. The crash is a downstream
**`HardCoreMappingStep` "No more hard cores available"** (`greedy_pack_softcores`) chip
capacity/packing **infrastructure** failure (head_pool at d6, features_13 at d8), raised
**after** `SoftCoreMappingStep` wrote `__target_metric.json` and **after** its parity gates
passed (NF↔SCM cascaded agreement **1.0**, torch↔deployed-sim parity **0.9961–1.0**). So
the deployed values are the **genuine full-test-set SCM accuracies captured pre-crash**
(each matches the final log "Test accuracy" line: d10_cascaded_s0=0.928,
d10_sync_s0=0.9917), **not** a training/firing-gain failure — but they are **not clean
finalized deployments**. The `d12_cascaded` seeds additionally have **no queue JSON at
all** (`returncode==None`, never enumerated), though their logs show no packing crash and
their artifacts exist. (2) `max_simulation_samples=200` → **read the gaps (10+pp at d10/12
is robust), not the third decimals**; the cascaded d6 s2=0.9463, d8 s2=0.94 are small-N
seed outliers inflating cascaded spread (sd 1.2–1.4pp vs sync's 0.15pp). (3) n_seeds=3 per
arm. **Next:** the d10/d12 crash means the *physical-core packing*, not the science, is the
blocker — re-run d6–d12 with a **larger `cores_config`** (or coalescing-on) so they
finalize `rc=0` and the death-cascade ladder becomes a clean VALID-vehicle result, and add
the **dataset axis** (FMNIST/KMNIST) and the **θ-cotrain / conversion_policy gate-fix** at
the d10 collapse rung (backlog `plan_stage:14/15/16`).

---

## 4g. The deep_cnn cascade re-opens off easy-MNIST and WIDENS with dataset margin — d4 AND d8 dataset axis (2026-06-24)

**Question (the dataset axis, on the deep CNN).** §4f closed the within-CNN *depth*
ladder on **MNIST only**. §4c found the cascade *absent* at the shallow `deep_cnn` d4
on easy MNIST. Does that no-collapse corner survive a **harder dataset**, and does the
deep-d8 death-cascade widen with dataset margin the way §4b's law predicts? This batch
runs `deep_cnn` (width 16, S=4, `ttfs_cycle_based`) at **d4 and d8** on
**FashionMNIST and KMNIST**, paired cascaded-vs-synchronized, 3 seeds/arm.

### The deep_cnn dataset table — cascaded→sync gap by (depth, dataset)

ANN refs are well above 10-class chance (0.10) on every cell, so each gap is genuine
firing-gain signal, **not** an untrained-floor artifact (unlike the §2 deep_mlp d≥12 rungs).

| d | dataset | cascaded (3-seed mean ± sd) | synchronized (3-seed mean ± sd) | **casc→sync GAP** | ANN ref | ANN-gap(casc) | verdict |
|--:|:--------|:----------------------------|:--------------------------------|------------------:|:--------|--------------:|:--------|
| 4 | *(ref §4c)* mnist | 0.9883 | 0.9898 | **−0.15** | ~0.992 | 0.47 | no collapse |
| 4 | fmnist | **0.8700 ± 0.71pp** | **0.9090 ± 0.36pp** | **+3.90** | 0.9276 | 5.76 | degraded |
| 4 | kmnist | **0.8867 ± 1.31pp** | **0.9486 ± 0.31pp** | **+6.19** | 0.9684 | 8.17 | degraded |
| 8 | *(ref §4f)* mnist | 0.9564 | 0.9904 | **+3.40** | 0.9928 | 3.80 | widens (mild) |
| 8 | kmnist | **0.9153 ± 1.59pp** | **0.9650 ± 0.24pp** | **+4.96** | 0.9684 | 5.23 | degraded |
| 8 | fmnist | **0.7802 ± 1.07pp** | **0.9000 ± 0.53pp** | **+11.98** | 0.9328 | 15.36 | collapse |

**Verdict — the dataset axis DRIVES the cascade on the CNN too, and it compounds with
depth.** Two findings, each on three paired seeds:

1. **The d4 "no-collapse" corner does NOT generalize off MNIST.** §4c's −0.15pp at
   d4/MNIST flips to **+3.90pp (FMNIST)** and **+6.19pp (KMNIST)** at the *same* depth —
   the cascaded deficit re-opens the moment the dataset margin tightens, even on the
   shallow trainable CNN. The gap already scales with dataset margin at d4
   (KMNIST 6.19 > FMNIST 3.90pp).

2. **At d8 the cascade widens with dataset margin: MNIST +3.40 < KMNIST +4.96 < FMNIST
   +11.98pp.** The deep CNN tracks task hardness exactly as §4b's LeNet5 law did, but the
   *depth* makes the hardest dataset's hit (FMNIST 11.98pp) much sharper than its d4 hit
   (3.90pp). **Synchronized holds within ≤3.2pp of the ANN on every dataset and depth**
   (FMNIST-d8 sync→ANN 3.18pp; KMNIST ≤0.42pp) — the unconditional default is unchanged.

Together this **completes the picture** §4f opened: the deep_cnn death-cascade is driven
by *both* depth (§4f, MNIST d6→d12) and dataset margin (this section, d4/d8 ×
{FMNIST, KMNIST}), and the two compound — the worst cell is *deep × hard*
(d8/FMNIST, 11.98pp), the best is *shallow × easy* (d4/MNIST, −0.15pp), mirroring the
deep_mlp interaction of §4b but on a VALID convnet.

**Confounds / bounds.** (1) **VALIDITY split by depth.** The **d4** cells are all
**finalized `rc=0`** (12 runs, no crash) — clean evidence. The **d8** cells carry the
**SAME `NON_FINALIZED_rc1` infra-crash confound as the consolidated §4f MNIST CNN cells**:
all 12 d8 runs finalized `returncode==1`, crashed downstream at **`HardCoreMappingStep`
"No more hard cores available"** (`greedy_pack_softcores`, segment
`neural_segment_until:features_13`) *after* SoftCoreMapping wrote `__target_metric.json`
and *after* its parity gates passed — so the d8 deployed values are the **genuine
pre-crash full-test SCM accuracies** (each matches its log's final "Test accuracy" line,
e.g. FMNIST_casc_s0 0.7684 == last 0.7684), **not** a training/firing-gain failure but
**not** a clean finalized deployment. (2) `max_simulation_samples=200` → read the gaps
(3.9–12pp robust), **not** third decimals; the cascaded arm carries wide seed-spread
(FMNIST-d8 sd 1.07pp, KMNIST-d8 sd 1.59pp, KMNIST-d4 sd 1.31pp) vs synchronized's tight
0.24–0.53pp. (3) n_seeds=3/arm, paired by seed at matched (depth, dataset). **Next:** the
enlarged-`cores_config` re-run (backlog `plan_stage:14`) that clears the d8 packing crash
would lift these d8 cells to clean `rc=0` VALID evidence; the d10 collapse rung on the
harder datasets (`plan_stage:15`) is the deep × hard compound's worst-case test.

---

## 4h. The within-CNN death-cascade on a CLEAN-FINALIZED (`rc=0`) ladder — a sharp DEPTH-THRESHOLD onset (lossless ≤d5, ~5pp plateau ≥d6), not the deep_mlp smooth widening (2026-06-24)

**Question (the validity upgrade).** §4f established the within-CNN depth law but on a
**`rc=1`-confounded** vehicle: every `dcnn_`/`pdcnndeep_` d6–d12 run crashed downstream
at `HardCoreMappingStep` "No more hard cores available" *after* the SCM metric was
written, so those deployed values are genuine pre-crash SCM accuracies but **not clean
`rc=0` finalized deployments**. This batch re-runs the ladder on a **VALID,
clean-finalized** vehicle and asks: does the depth law survive when the runs actually
deploy `rc=0`, and is the onset *smooth* (like the deep_mlp d4→d8 4.3→9.3pp widening)
or a *threshold*?

All cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based`, S=4, 3 seeds/arm paired by
seed, `max_simulation_samples=200` (cascaded) vs FULL 10k test set (synchronized).
Ledger: `cluster:"WS3"`, `kind:"depth"`, `model:"deep_cnn"`. The valid ladder spans
3 distinct `deep_cnn` config families (d4=`dcnn_`, d5=`pdcnnladder_`, d6/d8=`pdcnnbc_`)
because the `dcnn_`/`pdcnnladder_` d6–d8 runs themselves crashed `rc=1` (excluded).

### The clean-finalized within-CNN depth ladder — cascaded vs synchronized

| d | vehicle | cascaded deployed (3-seed mean ± sd) | synchronized deployed (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN gap | verdict |
|--:|:--------|:-------------------------------------|:----------------------------------|------------------:|:--------|-------------:|:--------|
| 4 | `dcnn_`        | 0.9883 ± 1.31pp (.97/.995/1.0)  | 0.9898 ± 0.24pp | **−0.15** | 0.9931 | 0.47 | lossless (tied, within 200-sample noise) |
| 5 | `pdcnnladder_` | 0.9917 ± 0.24pp (.99/.99/.995)  | 0.9924 ± 0.13pp | **−0.07** | 0.9913 | −0.04 | lossless (tied) |
| 6 | `pdcnnbc_`     | **0.9383 ± 0.85pp** (.935/.95/.93) | 0.9904 ± 0.12pp | **−5.21** | 0.9923 | 5.39 | **firing-gain degraded (SHARP onset)** |
| 8 | `pdcnnbc_`     | **0.9425 ± 2.75pp** (.97/.915; n=2) | 0.9935 ± 0.04pp | **−5.10** | 0.9925 | 5.00 | **firing-gain degraded (PLATEAU ~5pp)** |

(Sign convention here is `casc − sync`; the negative numbers mean cascaded is *below*
synchronized. §4f's table used `sync − casc`, hence the opposite sign — same direction.)

**Verdict — DEPTH-THRESHOLD CONFIRMED, NOT smooth widening (closeout-v2 §6 ruling).** On
a VALID deep_cnn (98.9–99.5% on-chip, trained ANN ~0.99, parity gates NF↔SCM=1.0 /
torch↔sim=1.0) the cascaded mode is **lossless and tied to synchronized through d5**
(−0.15pp, −0.07pp) and then **collapses to a ~5pp deficit at d6 (−5.21pp) and d8
(−5.10pp)** while synchronized stays pinned at the ANN ceiling (≤0.18pp). The phenomenon
is a **sharp d5→d6 ONSET followed by a ~5pp PLATEAU** — qualitatively unlike the deep_mlp
**smooth** d4(4.3pp)→d8(9.3pp) widening, and even unlike §4f's `rc=1` ladder (which
blew out to 11–14pp by d10/d12). The cleaner `pdcnnbc_` vehicle deploys at a *lower*
gap (~5pp) than §4f's crashed d10/d12 (~11–14pp). **Ruling on the §6 depth-risk cell:
the death-cascade reproduces on a valid, clean-finalized vehicle as a depth-threshold
(lossless ≤d5, ~5pp deficit ≥d6), so the headline-gating risk is REAL but its severity
is bounded to ~5pp (not collapse) over d6–d8 on deep_cnn, and synchronized is the safe
deep-model default** — confirming closeout-v2 §6.2's "prefer synchronized for deep
models" recommendation on a valid vehicle.

**Confounds / bounds.** (1) **EVAL-SET MISMATCH (read gaps, not 3rd decimals):** every
cascaded run subsamples to `max_simulation_samples=200` (0.005 grid, ~1.5–3.5pp/seed
binomial noise) while every synchronized run reports the **FULL 10000-sample** test set
(4-decimal grid, the closing "Test accuracy" == deployed metric exactly, per commit
5568518). Means are unbiased so the casc→sync gap is valid; the d6/d8 ~5pp gaps are
>2× the noise band (real), the d4/d5 sub-0.2pp gaps are within noise (lossless). (2)
**INVALID/CRASHED RUNS EXCLUDED:** `dcnn_d6_*`/`dcnn_d8_*` and `pdcnnladder_d6_*`/
`pdcnnladder_d7_*` all FAILED `rc=1` (hard-core-packing "No more hard cores available",
*before* deploying) — their stale `__target_metric.json` values (e.g. dcnn_d6 0.9758) are
pre-deployment training metrics, NOT used. So valid d6/d8 evidence is from `pdcnnbc_`, and
the valid ladder is a mild cross-vehicle composite (3 deep_cnn families, all MNIST/w16,
on-chip 98.9–99.5%). (3) **<3 SEEDS at d8:** `pdcnnbc_d8_cascaded` has only 2 finalized
seeds (0.97/0.915; s1 still in `q/running/`), so the d8 cascaded mean (0.9425, sd 2.75pp)
is a 2-seed estimate. (4) **NO at-chance confound:** all ANN refs ~0.99 (10-class chance
0.10), parity gates clean → the d6/d8 drop is a genuine firing-gain deficit, not an
untrained/buggy-mapping artifact. **Next:** finalize the d8 s1 seed and push d10/d12 on
the `pdcnnbc_` (bigger-cores) vehicle so the *whole* ladder is `rc=0`-clean; add the
θ-cotrain gate-fix at the d6 onset rung (backlog `plan_stage:19`).

---

## 4i. The deep × hard COMPOUND WORST-CASE — `deep_cnn` d10 FMNIST/KMNIST blows out to +16–18pp on a CLEAN `rc=0` vehicle (2026-06-24)

**Question (the deep × hard worst-case, closed).** §4g opened the deep_cnn dataset
axis at d4/d8 but flagged the **d10 collapse rung on the harder datasets as still
open — the deep × hard compound's worst case** (it could only run d8, and the d8
cells were `NON_FINALIZED_rc1`). This batch lands the **deepest VALID `rc=0`** rung:
`deep_cnn` (width 16, S=4, `ttfs_cycle_based`) at **d10** on **FashionMNIST and
KMNIST**, paired cascaded-vs-synchronized, on the **enlarged `bigcores` config
(`cores.count = 480`, `plan_stage:14`)** that clears the §4g `HardCoreMappingStep`
"No more hard cores available" crash. All 10 done runs are **`rc=0`** and reach
`HardCoreMappingStep` *without* the crash — **CLEAN FINALIZED deployments**, not the
§4g pre-crash SCM reads.

### The d10 dataset cells — cascaded vs synchronized (`rc=0`, `FINALIZED_rc0`)

ANN refs are well above 10-class chance (0.10) on both cells, so each gap is a
genuine firing-gain deficit, **not** an untrained-floor artifact (unlike §2 deep_mlp d≥12).

| d | dataset | cascaded deployed (mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN | sync→ANN | verdict |
|--:|:--------|:------------------------------|:-------------------------|------------------:|:--------|---------:|---------:|:--------|
| 10 | fmnist | **0.7250 ± 1.50pp** (n=2: .71/.74) | 0.9041 ± 0.34pp (n=3) | **+17.91** | 0.9347 | 20.97 | 3.06 | **firing-gain collapse** |
| 10 | kmnist | **0.8025 ± 3.25pp** (n=2: .77/.835) | 0.9623 ± 0.25pp (n=3) | **+15.98** | 0.9663 | 16.38 | 0.40 | **firing-gain collapse** |

### The dataset-axis depth table — casc→sync gap (pp) by (depth, dataset)

| dataset | d4 | d8 | **d10 (NEW)** | trend |
|:--------|---:|---:|--------------:|:------|
| mnist  | −0.15 | +3.40 | *(cross-vehicle ‡)* | — |
| fmnist | +3.90 | +11.98 | **+17.91** | **monotone widening** |
| kmnist | +6.19 | +4.96 | **+15.98** | d8 dip, **then blows out** |

**Verdict — the deep × hard compound WORST-CASE is CONFIRMED at d10.** On a VALID,
clean-finalized `rc=0` convnet the cascaded mode **collapses 16–18pp below
synchronized** on the harder datasets — **the largest cascaded deficits in the entire
deep_cnn table**. FMNIST widens **monotonically with depth** (d4 +3.90 → d8 +11.98 →
d10 **+17.91pp**); KMNIST has a d8 non-monotone dip (+4.96) but **blows out to +15.98pp
at d10**. **Synchronized stays pinned within 0.40–3.06pp of the well-above-chance ANN
ceiling at d10** (KMNIST sync→ANN 0.40pp, FMNIST 3.06pp) — the unconditional deep-model
default is **reinforced, not threatened**. This is the deep × dataset-margin compound the
§4b/§4g law predicted: depth and task hardness **multiply**, and the worst corner
(deep × hard) is where the death-cascade is most severe.

**Confounds / bounds.** (1) **CASCADED ARMS n=2 (NOT 3):** FMNIST cascaded s1 finalized
`rc=-9` (killed/OOM) and KMNIST cascaded s0 finalized `rc=1`, both in `q/failed/` →
excluded; cascaded means are 2-seed (FMNIST {.71/.74}, KMNIST {.77/.835}), synchronized
arms are full 3 seeds. KMNIST cascaded per-seed spread is wide (sd 3.25pp). (2) **EVAL-SET
MISMATCH (read gaps, not 3rd decimals):** every cascaded run subsamples to
`max_simulation_samples=200` (0.005 grid; deployed bare floats are exact 1/200 multiples,
e.g. FMNIST casc_s0 0.71 ≈ HCM full-test 0.7087, KMNIST casc_s1 0.77 ≈ HCM 0.7852) while
synchronized reports the **FULL 10000-sample** test set (4-decimal SCM == deployed). Means
are unbiased so the 16–18pp casc→sync gaps are **>4–5× the per-seed binomial band** — robust.
(3) **NO at-chance confound:** ANN refs ~0.935 (FMNIST) / ~0.966 (KMNIST) ≫ 0.10 chance →
genuine firing-gain death-cascade, NOT an untrained-floor artifact. (4) **VALIDITY:** all
10 done runs are `rc=0` and reach `HardCoreMappingStep` **without** the "No more hard cores"
crash that confounded the §4g d8 cells — this is the enlarged `bigcores` (count=480,
`plan_stage:14`) vehicle, so these are **CLEAN FINALIZED `rc=0`** deployments
(`FINALIZED_rc0`), clearing the §4g `NON_FINALIZED_rc1` confound. (5) **DEPTH-AXIS NOTE:**
the MNIST d10 reference is rc-dependent/cross-vehicle (§4f `rc=1` +13.86pp vs §5b clean
`pdcnnbc_` −4.00pp ‡), so the cleanest depth comparison for these new FMNIST/KMNIST d10
cells is against the §4g d4/d8 **dataset** rows, not the MNIST depth ladder. **Next:**
finalize the 3rd cascaded seed on each cell (re-run FMNIST s1 / KMNIST s0); the
θ-cotrain / `ttfs_staircase_ste` firing-gain gate-fix on the d10 deep×hard collapse cell
is the highest-leverage rescue test (backlog `plan_stage:24`), and a d6/d8 FMNIST/KMNIST
gate-fix completes the dataset × depth × rescue cube (backlog `plan_stage:25`).

---

## 4j. The §4g d8 dataset cells are now CLEAN `rc=0` on `bigcores` — confound closed, dataset-margin death-cascade VALID on the convnet (2026-06-24)

**Question (close the §4g d8 confound).** §4g measured the deep_cnn d8 FMNIST/KMNIST
cascade *pre-crash* — every d8 run finalized `returncode==1` at `HardCoreMappingStep`
"No more hard cores available" *after* the SCM metric was written, so by the strict
`rc==0` rule they were `NON_FINALIZED_rc1` (the deployed values are genuine pre-crash
SCM reads, but not formally valid). This batch re-runs the **same d8 FMNIST/KMNIST cells
on the enlarged `bigcores` config** (`cores.count = 480`, 4×-enlarged hard cores,
backlog `plan_stage:17`) so they finalize **`rc=0`** — the d8 analog of what
`plan_stage:14` did for the MNIST depth ladder and §4i did for the d10 dataset rung.
`deep_cnn` (width 16, S=4, `ttfs_cycle_based`), paired cascaded-vs-synchronized.

### The d8 dataset cells — CLEAN `rc=0` (replaces the §4g `rc=1` reads)

ANN refs ≫ 10-class chance (0.10), so each gap is a genuine firing-gain deficit, not
an untrained-floor artifact.

| dataset | cascaded deployed (mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN | sync→ANN | verdict |
|:--------|:------------------------------|:-------------------------|------------------:|:--------|---------:|---------:|:--------|
| fmnist | **0.7900 ± 2.86pp** (n=3: .83/.765/.775) | 0.9034 ± 0.66pp (n=3) | **+11.34** | 0.933 | 14.28 | 2.98 | **firing-gain collapse** |
| kmnist | **0.8900 ± 1.0pp** (n=2: .88/.90) | 0.9619 ± 0.47pp (n=3) | **+7.19** | 0.9689 | 8.11 | 0.55 | **firing-gain degraded** |

### The dataset-axis depth table now reads on a single CLEAN-`rc=0` vehicle

| dataset | d4 (§4g, `rc=0`) | **d8 (NEW, `rc=0`)** | d10 (§4i, `rc=0`) | trend |
|:--------|---:|--------------:|---:|:------|
| mnist  | −0.15 | +3.40 ‡ | *(cross-vehicle)* | — |
| fmnist | +3.90 | **+11.34** | +17.91 | **monotone widening** |
| kmnist | +6.19 | **+7.19** | +15.98 | widens (d8 no longer dips) |

**Verdict — confound CLOSED, the dataset-margin death-cascade is VALID on the convnet
at d8.** The clean `rc=0` re-run **confirms the §4g pre-crash reads**: FMNIST
+11.98 → **+11.34pp**, KMNIST (the §4g +4.96pp `rc=1` 3-seed) → **+7.19pp** on the
clean 2-seed cascaded arm. The dataset-margin ordering at d8 is
**MNIST +3.40 < KMNIST +7.19 < FMNIST +11.34pp** — exactly the section-10.1 law,
now off the INVALID deep_mlp and onto a VALID, clean-finalized convnet.
**Synchronized HOLDS near its ANN on both cells** (sync→ANN 2.98pp FMNIST / 0.55pp
KMNIST) — the unconditional deep-model default is reinforced. With §4j (d8) and §4i
(d10) both clean, the **entire d4/d8/d10 × {FMNIST,KMNIST} dataset-axis cube is now
`rc=0`-valid**, and FMNIST widens monotonically with depth (+3.90 → +11.34 → +17.91pp).

**Confounds / bounds.** (1) **KMNIST cascaded n=2 (NOT 3):** the third seed
`pdcnnd8databc_KMNIST_DataProvider_cascaded_s2` is still in `q/running/`
(NON-FINALIZED) and excluded per the strict `rc==0` rule → KMNIST cascaded arm is
2-seed (s0=.88/s1=.90, sd 1.0pp); FMNIST cascaded + both sync arms are full 3-seed.
(2) **EVAL-SET MISMATCH (read gaps, not 3rd decimals):** cascaded subsamples to
`max_simulation_samples=200` (0.005 grid; FMNIST .83/.765/.775 carries small-N
variance, sd 2.86pp) — the 7–11pp gaps are several× the per-seed binomial band.
deployed = the bare float in `generated/<id>_phased_deployment_run/__target_metric.json`
(the 200-sample SCM metric), which differs slightly from each log's final
"Test accuracy" line (FMNIST_casc_s0 target 0.83 vs log-last 0.7989) — the
`__target_metric.json` convention governs. (3) **NO at-chance confound:** ANN refs
~0.933 (FMNIST) / ~0.969 (KMNIST) ≫ 0.10 chance → genuine firing-gain. (4)
**VALIDITY:** all 11 finalized runs are `rc=0`, reaching `HardCoreMappingStep`
*without* the §4g "No more hard cores" crash (`VALID_on_chip_majority_rc0`). **Next:**
finalize the KMNIST cascaded s2 seed; the θ-cotrain firing-gain gate-fix on the d6/d8
FMNIST/KMNIST grid (backlog `plan_stage:25`) maps the recovery surface across the
3.9–18pp deficit range (gated on the d10 gate-fix `plan_stage:24`).

---

## 4k. The MISSING d6 dataset rung is now filled — the FMNIST monotone-widening ladder is complete and continuous on the CLEAN `rc=0` convnet (2026-06-24)

**Question (fill the dataset-axis depth cube).** The §4j table read `d4 / d8 / d10`
on a single clean-`rc=0` vehicle but **left d6 empty** — exactly the inflection where
§4h located the within-CNN onset threshold (lossless ≤d5, ~5pp plateau ≥d6). This
batch lands the **d6 FMNIST/KMNIST dataset cells on the enlarged `bigcores` config**
(`cores.count = 480`, backlog `plan_stage:14`) so they finalize **`rc=0`** at
`HardCoreMappingStep` *without* the §4g "No more hard cores available" crash.
`deep_cnn` (width 16, S=4, `ttfs_cycle_based`), paired cascaded-vs-synchronized by seed.

### The d6 dataset cells — CLEAN `rc=0`

ANN refs ≫ 10-class chance (0.10), so each gap is a genuine firing-gain deficit, not
an untrained-floor artifact.

| dataset | cascaded deployed (mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN | sync→ANN | verdict |
|:--------|:------------------------------|:-------------------------|------------------:|:--------|---------:|---------:|:--------|
| fmnist | **0.8400 ± 2.00pp** (n=3: .84/.82/.86) | 0.9011 ± 0.49pp (n=3) | **+6.11** | 0.930 | 9.11 | 2.78 | **firing-gain degraded** |
| kmnist | **0.9100** (n=1: s0 only) | 0.9686 ± 0.30pp (n=2) | **+5.85 ‡** | 0.9753 | 6.28 | 0.80 | **degraded (n=1 provisional)** |

### The dataset-axis depth ladder is now CONTINUOUS (d4→d6→d8→d10) on one CLEAN-`rc=0` vehicle

| dataset | d4 (§4g, `rc=0`) | **d6 (NEW, `rc=0`)** | d8 (§4j, `rc=0`) | d10 (§4i, `rc=0`) | trend |
|:--------|---:|--------------:|---:|---:|:------|
| fmnist | +3.90 | **+6.11** | +11.34 | +17.91 | **monotone widening, no gaps** |
| kmnist | +6.19 | **+5.85 ‡** | +7.19 | +15.98 | widens (d6 single-seed) |

**Verdict — the d6 rung CONFIRMS the FMNIST monotone-widening law and closes the last
cube gap.** FMNIST now reads a smooth, gapless **+3.90 → +6.11 → +11.34 → +17.91pp**
ladder (d4→d6→d8→d10) on a single clean-finalized convnet — the within-CNN
death-cascade widens *monotonically* with depth and the d6 cell slots cleanly between
the §4g d4 (+3.90) and §4j d8 (+11.34) anchors. KMNIST's d6 (+5.85) sits between its d4
(+6.19) and d8 (+7.19), consistent with the gentler KMNIST ladder (n=1 caveat below).
**Synchronized HOLDS near its ANN on both cells** (sync→ANN 2.78pp FMNIST / 0.80pp
KMNIST) — the unconditional deep-model default is reinforced at the inflection depth.

**Confounds / bounds.** (1) **KMNIST cascaded n=1 (PROVISIONAL):** only
`pdcnnd6databc_KMNIST_DataProvider_cascaded_s0` finalized `rc=0`; s1/s2 are still in
`q/running/` (NON-FINALIZED) and excluded per the strict `rc==0` rule, and the 3rd sync
seed (`synchronized_s2`) is also still running → KMNIST is a single-seed cascaded point
vs a 2-seed sync arm; the +5.85pp gap is provisional (finalize s1/s2 to firm it).
FMNIST d6 is full 3-seed on both arms. (2) **EVAL-SET MISMATCH (read gaps, not 3rd
decimals):** cascaded subsamples to `max_simulation_samples=200` (0.005 grid; deployed
bare floats are exact 1/200 multiples, e.g. FMNIST .84/.82/.86, KMNIST .91, log
"[SimulationRunner] Subsampled 200 / 10000") while synchronized reports the FULL
10000-sample test set (4-decimal SCM, no subsample line, per commit 5568518) — the ~6pp
gaps are >2–3× the per-seed binomial band. deployed = the bare float in
`generated/<id>_phased_deployment_run/__target_metric.json`. (3) **NO at-chance
confound:** ANN refs ~0.930 (FMNIST) / ~0.973 (KMNIST) ≫ 0.10 chance → genuine
firing-gain. (4) **VALIDITY:** all 9 done runs are `rc=0`, reaching
`HardCoreMappingStep` without the §4g crash (`VALID_on_chip_majority_rc0`,
`plan_stage:14`); the only paired-arm config diff is `ttfs_cycle_schedule`
cascaded-vs-synchronized. **Next:** finalize the KMNIST cascaded s1/s2 seeds to firm
the +5.85pp d6 read; the θ-cotrain firing-gain gate-fix grid (`plan_stage:25`, now
spanning d6/d8) maps the recovery surface across the now-continuous 3.9–18pp ladder.

---

## 4m. The `pdcnnbcclean_` vehicle UPGRADES §4h to a FULL 3-seed `rc=0` d8 plateau AND lands a synchronized-LOSSLESS d10 rung (`item_id=dcnn_clean_depth_ladder_d8_d10`, 2026-06-24)

**Question (firm the plateau, extend one rung).** §4h read the within-CNN depth
plateau on the `pdcnnbc_` vehicle but its d8 cascaded arm was **n=2** (s1 still
running) — a 2-seed estimate. This batch re-runs the ladder on the explicitly
clean-named **`pdcnnbcclean_`** `bigcores` vehicle (`cores.count = 480`, MNIST, w16,
S=4, `ttfs_cycle_based`, paired cascaded-vs-synchronized by seed,
`max_simulation_samples=200`) and asks: (a) does the d8 plateau hold at a **full
3-seed `rc=0`** read, and (b) does synchronized stay lossless **one rung deeper at
d10**?

### The clean d8/d10 rungs — cascaded vs synchronized (`rc=0`)

ANN refs (mean 0.9926 d8 / 0.992 d10) ≫ 10-class chance (0.1135) on every cell, so each
gap is a genuine firing-gain deficit, not an untrained-floor artifact.

| d | cascaded deployed (3-seed mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN | sync→ANN | validity | verdict |
|--:|:-------------------------------------|:-------------------------|------------------:|:--------|---------:|---------:|:---------|:--------|
| 8 | **0.9517 ± 2.04pp** (.925/.975/.955; n=3) | 0.9933 ± 0.01pp (n=3) | **+4.16** | 0.9926 | 4.07 | −0.06 | `VALID_3seed_both_arms` (all 6 `rc=0`) | **firing-gain PLATEAU (~4–5pp, not widening)** |
| 10 | *0.9555 prov.* (.9427/.9545/.9694; **NON-FINALIZED**) | 0.9932 ± 0.10pp (n=3) | *+3.77 prov.* | 0.992 | — | **−0.12** | `SYNC_VALID_rc0`, cascaded pending | **synchronized LOSSLESS at d10; cascaded provisional ⇒ plateau** |

**Verdict — PLATEAU CONFIRMED at d8 (full 3 seeds), synchronized LOSSLESS through d10.**
On the clean `pdcnnbcclean_` convnet the d8 cascaded→sync gap is **+4.16pp** (all six
runs `rc=0`), squarely in the ~5pp plateau band of §4h (d6 −5.21, d8 −5.10) and
§4k (d6 KMNIST +5.85) — and **far from** the §4f `rc=1`-confounded 11–14pp collapse.
This upgrades §4h's 2-seed d8 to a **full 3-seed `rc=0`** anchor and *holds the
plateau*: the within-MNIST death-cascade **plateaus, it does not widen** on a valid
trainable convnet. Synchronized holds the ANN ceiling **lossless at both d8 (−0.06pp)
and d10 (−0.12pp)**, extending the deep-model-default verdict one rung past d8. The
d10 cascaded provisional (mean 0.9555, gap +3.77pp) is **consistent with the d8
plateau** but is NON-FINALIZED and uncountable.

**Confounds / bounds.** (1) **d10 cascaded NON-FINALIZED (at verifier snapshot):**
`pdcnnbcclean_d10_cascaded_s0/s1/s2` were in `q/running/` with `result=NONE`, EXCLUDED
per the strict `rc==0` rule, so `cascaded_to_sync_gap_pp` at d10 is `null`; their
mid-pipeline `__target_metric.json` (0.9427/0.9545/0.9694) is provisional only.
**Progress (2026-06-24):** s0+s2 have since finalized `rc=0` (`q/done/`, deployed
0.95/0.96), only s1 remains running → 2/3 finalized, the plateau strengthening; lock
when s1 hits `rc=0`. (2) **EVAL-SET MISMATCH (read gaps, not 3rd decimals):** cascaded
subsamples to `max_simulation_samples=200` (0.005 grid; d8 .925/.975/.955 are exact
1/200 multiples) while synchronized reports the FULL 10000-sample test set — the d8
+4.16pp gap is >2× the per-seed binomial band; the sub-0.2pp sync→ANN gaps are within
noise (lossless). (3) **SEED SPREAD:** cascaded d8 sd 2.04pp (.925/.975/.955) vs
synchronized sd 0.01pp, so the +4.16pp mean carries seed variance — but every cascaded
seed (≥0.925) is far above the 11–14pp collapse band. (4) **NO at-chance confound:**
ANN refs 0.989–0.994 ≫ 0.1135 → genuine firing-gain. (5) **SCOPE (d12):** the verifier
snapshot saw NO d12 runs; as of 2026-06-24 `pdcnnbcclean_d12_cascaded/synchronized_*`
ARE now in `q/running/` (in flight, not yet finalized) — the deepest decisive rung is
running, not absent. **Next:** finalize d10 cascaded s1 and the d12 arms to lock the
plateau-vs-widen verdict at the deepest rungs; layer the θ-cotrain firing-gain gate-fix
(`plan_stage:25`) on the d8 plateau anchor.

---

## 4n. The §4k d6 KMNIST n=1 PROVISIONAL is UPGRADED to a full 3-seed cell — d6 dataset-margin ordering CONFIRMED on a first-fully-finalized `rc=0` vehicle (`item_id=ws3_dcnn_d6_onset_dataset_axis`, 2026-06-24)

**Question (firm the §4k d6 rung).** §4k filled the d6 dataset rung but left **KMNIST
cascaded at n=1 (PROVISIONAL, gap +5.85pp)** — a single surviving seed from the
`pdcnnd6databc_*` family while s1/s2 were still running. This batch lands the **first
fully-finalized (12/12 `rc=0`) d6 dataset-axis** on the `pdcnnbcd6data_*` `bigcores`
family (`cores.count = 480`, `plan_stage:14`): `deep_cnn` (width 16, S=4,
`ttfs_cycle_based`, `max_simulation_samples=200`), **3 seeds per arm** on FMNIST and
KMNIST, paired cascaded-vs-synchronized by seed. All 12 runs are in `runs/campaign/q/done/`
with `rc=0` (NOT the §4f/§4g `rc=1` `HardCoreMappingStep`-crash batch).

### The d6 dataset cells — now FULL 3-seed, CLEAN `rc=0`

ANN refs ≫ 10-class chance (0.10) on both cells → genuine firing-gain, not an
untrained-floor artifact.

| dataset | cascaded deployed (3-seed mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN | sync→ANN | verdict |
|:--------|:-------------------------------------|:-------------------------|------------------:|:--------|---------:|---------:|:--------|
| fmnist | **0.8183 ± 1.89pp** (.81/.84/.805) | 0.8962 ± 0.66pp (.8888/.8984/.9015) | **+7.79** | 0.9293 | 11.09 | 3.31 | **firing-gain degraded** |
| kmnist | **0.9167 ± 1.61pp** (.91/.935/.905) | 0.9619 ± 0.23pp (.9598/.9616/.9644) | **+4.53** | 0.9647 | 4.81 | 0.28 | **firing-gain degraded, dataset-stable** |

**Verdict — the d6 dataset-margin ordering is CONFIRMED on a clean `rc=0` vehicle, and
the KMNIST n=1 provisional is UPGRADED.** At matched d6 the harder dataset carries the
larger cascade deficit: **FMNIST +7.79pp ≫ KMNIST +4.53pp** (ANN ~0.929 vs ~0.965) —
exactly the §4b/§4g dataset-margin law. The full-seed KMNIST cell stays **MILD and
dataset-stable** (sync within **0.28pp** of ANN, cascaded sd 1.61pp), confirming and
firming the §4k n=1 provisional. **Synchronized HOLDS near its ANN on both cells**
(sync→ANN 3.31pp FMNIST / 0.28pp KMNIST) — the deep-model default is reinforced at the
onset depth.

**Confounds / bounds.** (1) **TWO KMNIST run-id families DISAGREE on the absolute gap.**
The authoritative cell above is the `pdcnnbcd6data_KMNIST_*` family (gap **+4.53pp**,
casc 0.9167 sd 1.61pp, **sync sd 0.23pp** — the cleaner read, and the family whose
seeds .91/.935/.905 match the item). A **separate** `pdcnnd6databc_KMNIST_*` family —
the one that supplied the §4k n=1 provisional (s0=0.91) — also finalized 3-seed this
round but reads **WIDER (gap +7.94pp, casc 0.8900 sd 3.04pp** driven by the s1=0.855
outlier, sync 0.9694). Both are recorded in the ledger; the two families agree on the
**mild/dataset-stable** conclusion and the **FMNIST > KMNIST ordering**, but not on the
3rd-decimal gap — read the ordering, not the absolute KMNIST gap. (2) The FMNIST d6 here
(+7.79pp, casc 0.8183) reads **WIDER** than the older §4k FMNIST d6 cell (+6.11pp, casc
0.8400 from an earlier batch) — a fresh-batch shift, both in the degraded band. (3)
**EVAL-SET MISMATCH (read gaps, not 3rd decimals):** cascaded subsamples to
`max_simulation_samples=200` (0.005 grid; deployed bare floats are exact 1/200 multiples)
while synchronized reports the FULL 10000-sample test set — the 4.5–7.8pp gaps are
several× the per-seed binomial band. (4) **NO at-chance confound:** ANN refs ~0.929
(FMNIST) / ~0.965 (KMNIST) ≫ 0.10 chance → genuine firing-gain. (5) **VALIDITY:** all 12
runs `rc=0` in `q/done/`, reaching `HardCoreMappingStep` without the §4g crash
(`VALID_on_chip_majority_rc0`); the only paired-arm config diff is `ttfs_cycle_schedule`.
**Next:** layer the θ-cotrain / `ttfs_staircase_ste` firing-gain gate-fix on the d6
FMNIST/KMNIST cells (backlog `plan_stage:26`) to map recovery at the onset depth; the
WS7 §4f-flagged convnet θ-cotrain `rc=1` crash must be fixed before that gate-fix can run.

---

## 4o. The §4d.1 lenet5 KMNIST gap is RE-MEASURED at matched resolution (both arms n=1000) — MILD, dataset-stable CONFIRMED; SVHN sync-only this round (`item_id=ws3_lenet_paired_n1000_kmnist_svhn`, 2026-06-24)

**Question (de-confound the §4d.1 resolution mix).** §4d.1 read the lenet5 KMNIST
cascade gap (+1.45pp) by pairing an **n=1000 cascaded** arm against an **n=50
synchronized** arm — a resolution mismatch flagged there as the open confound. This
batch supplies the **paired n=1000 synchronized** arm (`plncpair_lenet_*_synchronized_n1000_*`)
so both arms read at `max_simulation_samples=1000`, and attempts the **SVHN** cell on the
same vehicle. Runs: lenet5, `ttfs_cycle_based`, thresholding `<=`. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`. Pairing axis =
`deployment_parameters.ttfs_cycle_schedule`.

### The matched-resolution KMNIST cell (both arms n=1000) + the SVHN sync-only baseline

ANN refs ≫ chance (KMNIST 0.10, SVHN 0.196) → genuine firing-gain, not at-chance.

| dataset | cascaded n=1000 (3-seed mean ± sd) | synchronized n=1000 (mean ± sd) | **casc→sync GAP** | prior §4d.1 (n50 sync) | ANN ref | residual casc→ANN | verdict |
|:--------|:-----------------------------------|:--------------------------------|------------------:|:-----------------------|:--------|------------------:|:--------|
| kmnist  | **0.934 ± 0.73pp** (.924/.937/.941) | **0.9519 ± 0.30pp** (.9476/.9537/.9543) | **+1.79** | +1.45 (sync 0.9485) | 0.9646 | 3.06 | **MILD, dataset-stable CONFIRMED** |
| svhn    | *cascaded all `rc=1` — UNAVAILABLE* | **0.8593 ± 0.36pp** (.8542/.8616/.8619) | *null* | — | 0.8945 | sync→ANN **3.52** | **sync-only baseline (cascaded recovery pending)** |

**Verdict — the §4d.1 MILD, dataset-stable verdict is CONFIRMED at matched resolution.**
With both arms at n=1000 the KMNIST cascade gap is **+1.79pp** (cascaded 0.934 vs sync
0.9519) — squarely in the **~1.4–1.8pp MILD band**, between MNIST-lossless and the
FMNIST ~6pp. The n=50 sync arm (0.9485) was only **0.34pp BELOW** the n1000 sync
(0.9519), so the §4d.1 resolution-mix confound was **SMALL** and moved the verdict
**toward (not away from) MILD** (1.45pp → 1.79pp, both in band). A residual
**cascaded→ANN gap of 3.06pp > cascaded seed sd 0.73pp** persists ⇒ a *small-but-real*
firing-gain residual, not pure noise. **SVHN is sync-only this round:** synchronized
n1000 0.8593 (sync→ANN 3.52pp, ANN 0.8945 ≫ SVHN chance 0.196) is a valid sync baseline,
but **all 6 SVHN cascaded n1000 seeds** (both `plncpair_*` and `csr_*` prefixes)
finalized `rc=1` in `q/failed/`, so **no matched cascaded→sync gap is computable** for
SVHN.

**Confounds / bounds.** (1) **SVHN cascaded UNAVAILABLE:** all 3 seeds of BOTH cascaded
prefixes failed `rc=1` → `cascaded_to_sync_gap_pp = null`; the SVHN cell remains
sync-only (and the §4b/§4d.1 SVHN cascaded reads stay non-finalized). (2) The matched
KMNIST gap (1.79pp) is **slightly LARGER** than the §4d.1 confounded 1.45pp purely
because the n1000 sync reads +0.34pp over the n50 sync — both numbers stay in the MILD
band, so the verdict is unchanged. (3) all 9 harvested runs (6 KMNIST + 3 SVHN sync) are
`rc=0`, finalized, artifact present, `max_simulation_samples=1000` on both arms → the
round-1 n=50 (0.02-granularity) sync confound is removed for the KMNIST cell. (4) both
arms share `model_type=lenet5`, TTFS, thresholding `<=`. **Next:** recover the SVHN
cascaded n=1000 arm (backlog `plan_stage:27`) to complete the matched 4-dataset CNN
cascaded→sync table (MNIST lossless / KMNIST +1.79 / FMNIST ~6 / SVHN open).

---

## 4r. The d8 deep_cnn dataset cells RE-MEASURED at genuine n=1000 (BOTH arms) — the dataset-margin death-cascade HARDENS vs n=200, ordering holds at 4-decimal fidelity (`item_id=dcnn_d8_dataset_n1000`, 2026-06-25)

**Question (de-confound the §4j n=200 read).** §4j read the deep_cnn d8 FMNIST/KMNIST
cascade off `max_simulation_samples=200` (0.005-grid cascaded vs full-10k synchronized,
a resolution mismatch). Does the dataset-margin gap survive a **fully paired n=1000**
re-measure where **both arms** subsample to `max_simulation_samples=1000`, or was the
n=200 read grid-noise inflation?

Runs: `pdcnnd8datan1000_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`
(12 runs, **all `rc=0`**, 3 seeds/arm, paired by seed, `deep_cnn` w16, S=4,
`ttfs_cycle_based`, `max_simulation_samples=1000`). Ledger: `cluster:"WS3"`,
`kind:"arch_dataset"`, `model:"deep_cnn"`, `depth:8`.

### The d8 dataset cells — genuine n=1000, both arms paired

ANN refs ≫ 10-class chance (0.10) on both cells → genuine firing-gain, not an
untrained-floor artifact.

| dataset | cascaded n1000 (3-seed mean ± sd) | synchronized n1000 (mean ± sd) | **casc→sync GAP** | n=200 prior (§4j) | ANN ref | casc→ANN | sync→ANN | verdict |
|:--------|:----------------------------------|:-------------------------------|------------------:|------------------:|:--------|---------:|---------:|:--------|
| fmnist | **0.7677 ± 2.51pp** (.794/.765/.744) | 0.9015 ± 0.21pp (.9017/.8993/.9035) | **+13.38** | +11.34 | 0.9329 | 16.40 | 3.14 | **firing-gain collapse (HARDENS)** |
| kmnist | **0.8903 ± 1.58pp** (.904/.873/.894) | 0.9732 ± 0.37pp (.9721/.9701/.9773) | **+8.28** | +7.19 | 0.9756 | 9.06 | 0.24 | **firing-gain degraded (HARDENS)** |

**Verdict — VALID-CONFIRMED; the n=200 dataset-margin read was NOT noise-inflated, the
gap HARDENS at genuine resolution.** With both arms at `max_simulation_samples=1000`,
the d8 cascade gap **grows** vs the §4j n=200 read: FMNIST +11.34 → **+13.38pp**,
KMNIST +7.19 → **+8.28pp**. Higher resolution *deepens* the death-cascade (matching the
§5b depth-axis n=1000-hardens precedent), so the gap is **not** a small-N grid artifact.
The dataset-margin ordering **KMNIST (+8.28) < FashionMNIST (+13.38)** holds at
4-decimal fidelity. Cascaded carries all the spread and deficit (FMNIST sd 2.51pp / ANN
gap 16.40pp; KMNIST sd 1.58pp / ANN gap 9.06pp); **synchronized stays near its ANN
ceiling** (FMNIST sync→ANN 3.14pp, KMNIST 0.24pp). This is a **genuine firing-gain
result** — both ANN refs (FMNIST 0.9317, KMNIST 0.9809) are far above 10-class chance,
fully trained — and the pairing is clean (unlike the §4d/§4d.1 lenet5 precedent that
mixed n1000-cascaded vs n50-synchronized, **both arms here are n=1000**).

**Confounds / bounds.** (1) **RESIDUAL eval asymmetry (minor):** the bare deployed floats
are exact 1/1000 multiples on the cascaded arm and on the synchronized arm too
(both subsample to 1000), so the prior n200-vs-10000 mismatch is **fully removed** for
this batch — the only residual is ~±0.1pp 1000-sample binomial noise, far below the
8–13pp gaps. (2) **n_seeds=3 per arm** (full 2-dataset × 2-policy × 3-seed grid
finalized, all 12 `rc=0`). (3) **DEPTH-MONOTONE CROSS-REFERENCE:** this batch measures
only d8, so the depth-widening must be cross-referenced against the §5b depth-axis
n=1000 precedent (d8 MNIST +8.51pp, d10 +11.14pp) rather than measured within this batch;
the d8 dataset gaps here (+8.28/+13.38pp) are consistent with and exceed the d8 MNIST
+8.51pp baseline — dataset-margin **amplifies** the cascade beyond the MNIST anchor. (4)
**NO at-chance confound:** ANN refs FMNIST 0.93 / KMNIST 0.98 ≫ 0.10 → genuine. **Next:**
the θ-cotrain / `ttfs_staircase_ste` firing-gain gate-fix on the d8 FMNIST collapse cell
(the worst dataset-margin corner) is the highest-leverage rescue test.

---

## 4s. The d6 deep_cnn dataset cells at genuine n=1000 — the d6 rung HOLDS and the continuous FMNIST monotone-widening ladder SURVIVES (`item_id=dcnn_d6_dataset_n1000`, 2026-06-25)

**Question (firm the §4k/§4n d6 rung at high resolution).** §4k/§4n read the d6 dataset
cells off n=200 (and §4k's KMNIST was an n=1 provisional). Does the d6 cascade gap survive
a **genuine n=1000** read, and does the continuous FMNIST widening ladder
(d4 → d6 → d8 → d10) slot the d6 rung cleanly at 4-decimal resolution?

Runs: `pdcnnd6datan1000_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}`
(12 runs, **all `rc=0`**, full 3 seeds/arm, paired by seed, `deep_cnn` w16, S=4,
`ttfs_cycle_based`, cascaded `max_simulation_samples=1000`). Ledger: `cluster:"WS3"`,
`kind:"arch_dataset"`, `model:"deep_cnn"`, `depth:6`.

### The d6 dataset cells — genuine n=1000

ANN refs ≫ 10-class chance (0.10) → genuine firing-gain, not an untrained-floor artifact.

| dataset | cascaded n1000 (3-seed mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | prior read | ANN ref | casc→ANN | sync→ANN | verdict |
|:--------|:----------------------------------|:-------------------------|------------------:|:-----------|:--------|---------:|---------:|:--------|
| fmnist | **0.8247 ± 0.62pp** (.817/.832/.825) | 0.8979 ± 0.36pp (.9025/.8938/.8975) | **+7.33** | +6.11 (n200, §4k) | 0.9299 | 10.46 | 3.25 | **firing-gain degraded (HARDENS)** |
| kmnist | **0.917 ± 0.43pp** (.913/.923/.915) | 0.9598 (.953/.9594/.9671) | **+4.28** | +5.85 (n=1 prov., §4k) | 0.9631 | 5.28 | −0.34 | **firing-gain degraded (FIRMED to 3-seed)** |

**Verdict — SUPPORTED at high resolution; the d6 rung HOLDS and the continuous FMNIST
widening ladder SURVIVES.** FMNIST +6.11pp (n200) → **+7.33pp** (n1000) HARDENS — matching
the depth-axis precedent that n=1000 deepens, not shrinks, the gap. KMNIST +5.85pp (n=1
provisional) → **+4.28pp** (full 3-seed) confirms the degraded-but-gentler KMNIST ladder.
The **FMNIST monotone-widening ladder stays continuous and gapless**:
**d4 +3.90 → d6 +7.33 → d8 +11.34 → d10 +17.91pp**, with the d6 rung slotting cleanly
between the d4 and d8 anchors at 4-decimal resolution. KMNIST d6 +4.28pp sits below FMNIST
d6 +7.33pp (**dataset-margin ordering preserved**) and is consistent with the gentler KMNIST
ladder (d4 +6.19 / d8 +7.19). **Synchronized holds within ≤3.25pp of ANN on both datasets**
(FMNIST 3.25pp; KMNIST −0.34pp, statistically at/above its ANN) — the unconditional
synchronized default is reinforced at the within-CNN cascade-onset depth.

**Confounds / bounds.** (1) **0.005-grid noise fully deconfounded:** cascaded deployed
floats are now exact 1/1000 multiples (FMNIST .817/.832/.825, KMNIST .913/.923/.915), 5×
finer than the n=200 ladder. (2) **RESIDUAL eval asymmetry (minor):** cascaded eval is
n=1000 (1/1000 grid) while synchronized still reports the FULL 10000-sample test set
(its floats e.g. 0.9025/0.9594 are 1/10000 multiples) — read at the multi-pp gap scale
(7.33/4.28pp ≫ 0.1pp sampling noise), far less severe than the prior n200-vs-10000
mismatch, so the verdict is robust to it. (3) **All 12 runs `rc=0`, full 3-seed both arms**
(the prior KMNIST n=1 provisional point is retired). (4) **NO at-chance confound:** ANN
refs FMNIST ~0.93 / KMNIST ~0.96 ≫ 0.10 → genuine firing-gain. Distinguishing knob =
`ttfs_cycle_schedule` (cascaded vs synchronized). **Next:** the θ-cotrain firing-gain
gate-fix at the d6 onset rung on FMNIST (the harder dataset) maps recovery across the
now-continuous 3.9–18pp ladder.

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

## 5b. Clean bigcores ladder (rc=0) + genuine n=1000 resolution — 2026-06-24

This batch landed the **clean `pdcnnbc_` (480/480 bigcores, 4×-enlarged cores)
deep_cnn vehicle that finalizes `rc=0`** at the deep rungs, replacing the
`rc=1`-confounded `pdcnndeep_`/`dcnn_` cells whose `__target_metric.json` was
written *before* a downstream `HardCoreMappingStep` packing crash. It also adds a
**genuine high-resolution** read (nevresim n=1000, 5× the n=200 ladder, so the
0.005-grid noise is removed) on a paired `pdcnndeeppair_` vehicle.

| depth | vehicle | res | cascaded mean (seeds) | sync mean | ANN | c→sync gap | rc |
|------:|---------|-----|-----------------------|-----------|-----|-----------:|----|
| d8 | `pdcnnbc_` | n200 | 0.9450 (0.97/0.95/0.915) | 0.9935 | 0.9929 | **−4.85pp** | 0 |
| d10 | `pdcnnbc_` | n200 | 0.9517 (0.925/0.945/0.985) | 0.9917 | 0.9923 | **−4.00pp** | 0 |
| d12 | `pdcnnbc_` | n200 | 0.98 (s1 only, **n=1**) | 0.9917 | 0.9887 | −1.17pp † | mixed |
| d8 | `pdcnndeeppair_` | **n1000** | 0.9066 (0.930/0.964/0.826) | 0.9917 | 0.9918 | **+8.51pp** | 1 ‡ |
| d10 | `pdcnndeeppair_` | **n1000** | 0.8807 (0.738/0.926/0.978) | 0.9921 | 0.9932 | **+11.14pp** | 1 ‡ |

**Verdicts.**
- **Clean d8 cascaded cell COMPLETED to 3 seeds** by this batch's
  `pdcnnbc_d8_cascaded_s1` (0.95, `rc=0`), superseding the prior `n_seeds=2`
  ledger entry (0.9425). Mean 0.9450, sd 2.27pp.
- **The d10 death-cascade gap SHRINKS by ~10pp on the clean vehicle:** the
  `rc=1`-confounded `pdcnndeep_d10` read was ~13.86pp; the clean `rc=0`
  `pdcnnbc_d10` read is **−4.00pp**. The prior gap was inflated by the
  post-metric packing crash and cross-vehicle comparison — the real cascaded
  deficit is a **bounded ~4–5pp plateau, NOT a deepening collapse.**
- **Synchronized is LOSSLESS at the ANN ceiling through d12** (0.9917 vs ANN
  0.9887, +0.30pp, 3 seeds `rc=0`, sd 0.07pp) — it owns deep deployment.
- **d12 cascaded is UNMEASURED (†):** only `pdcnnbc_d12_cascaded_s1` finalized
  `rc=0` (0.98); s0 and s2 are `returncode=-9` (killed, `q/failed/`). The
  −1.17pp gap is an n=1 point, NOT a reliable cell. **OPEN: re-run d12 cascaded
  s0/s2.**

**Confounds.**
1. **Eval-set mismatch (read gaps, not 3rd decimals):** cascaded subsamples to
   200 (n200) / 1000 (n1000) samples; synchronized reports the FULL 10k test set
   (4-decimal). Means are unbiased so c→sync gaps are valid; all reported gaps are
   >2× the per-seed binomial band.
2. **(‡) The n=1000 `pdcnndeeppair_` runs are `rc=1`** — same documented
   post-metric `HardCoreMappingStep` "No more hard cores available" crash
   (features_13/features_16) AFTER `__target_metric.json` + NF↔SCM parity
   (agreement 1.0) + torch↔sim parity (1.0) were written. Read as
   **CONFIRMED-WITH-CONFOUND** ONLY under the `pdcnndeep_`/`dcnn_` precedent.
3. **Resolution does NOT shrink the gap toward zero:** the n=1000 reads
   (8.51 / 11.14pp) are LARGER than the clean n=200 `pdcnnbc_` reads
   (−4.85 / −4.00pp). Higher resolution *hardens* the depth-law; the gap is not a
   grid artifact. (The n=200 `pdcnnbc_` cells are the VALID `rc=0` lower-bound
   deployment reads; the n=1000 cells are the higher-fidelity confounded reads.)
   The d10 s0=0.7375 log shows a **genuine mid-pipeline SCM collapse**
   (0.9939 ANN → 0.1873 → recovers 0.7375) — death-cascade fragility, not noise.
4. **No at-chance confound:** all ANN refs ~0.99 ≫ 10-class chance 0.10 → every
   cascaded drop is a genuine firing-gain deficit.

---

## 5b. lenet5 arch×dataset breadth at n1000 — the cascaded CNN deficit is MILD and dataset-STABLE, ordering by dataset margin (2026-06-24)

The VALID on-chip-majority `lenet5` convnet (99.1% on-chip) is the clean vehicle
for the cascaded firing-gain question that the INVALID host-majority `deep_mlp`
could only suggest. n=1000 re-measure, 3 seeds, `ttfs_cycle_based` S=4. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `model:"lenet5"`.

| dataset | ANN ref | cascaded deployed (3-seed mean ± sd_pp) | deployed→ANN gap | verdict |
|:--------|--------:|:----------------------------------------|-----------------:|:--------|
| MNIST | 0.9912 | **0.9873** (±0.25) | **0.39pp** | near-lossless |
| KMNIST | 0.9646 | **0.9340** (±0.73) | **3.06pp** | mild |
| KMNIST (re-pair `plncpair`) | 0.9657 | **0.9303** (±0.78) | **3.54pp** | reproduces csr KMNIST |
| FashionMNIST | 0.9183 | **0.8397** (±0.84) | **7.86pp** | largest (hardest dataset) |

**Verdict — the cascaded CNN deficit is MILD and dataset-stable; the gap orders
monotonically by dataset margin** (easier dataset / higher ANN → smaller gap:
MNIST 0.39 < KMNIST 3.06 < FMNIST 7.86pp). All three are far from the MLP-style
death-cascade collapse (`deep_mlp` d8 cascaded gaps were 10.8pp MNIST / 16.0pp
FMNIST). Seed psd ≤ 0.84pp → low-variance, not a fragile high-variance collapse.
The `plncpair` KMNIST set is byte-identical to `csr_lenet` KMNIST except
`experiment_name`, so it is a genuine replicate: 3.54pp vs 3.06pp reproduces the
cell (combined 6-seed KMNIST: deployed 0.9322, ANN 0.9651, gap 3.30pp).

**Confounds.**
1. **Matched-resolution cascaded→sync gap is NOT yet computable.** All n1000
   synchronized counterparts (`plnsync_lenet_{MNIST,FashionMNIST}_synchronized_n1000`,
   `plncpair_lenet_KMNIST_synchronized_n1000`) remain PENDING (0 finalized), so
   `cascaded_to_sync_gap_pp` is **null** in every record — only the deployed→ANN
   gap is reported. The only sync reference present is the 50-SAMPLE `sync_full`
   tag (MNIST ~0.9891, FMNIST ~0.8999) which is UNMATCHED resolution (n=50 vs
   n=1000): it implies MNIST 0.18pp / FMNIST 6.02pp casc→sync **for context only**,
   not a valid 3rd-decimal gap.
2. **No chance confound.** All 12 cascaded ANN refs ≫ chance (MNIST 0.990–0.992,
   KMNIST 0.961–0.969, FMNIST 0.916–0.920); every drop is a genuine firing-gain
   deficit.
3. All 12 cascaded runs VALID (rc=0, not timed_out, artifact_ok, 3 seeds/cell).
   SVHN cascaded/sync n1000 runs exist but are OUT OF SCOPE for this
   MNIST→FMNIST→KMNIST item (and the SVHN cascaded set is in `q/failed/`).

Run ids: `csr_lenet_{MNIST,KMNIST,FashionMNIST}_DataProvider_cascaded_n1000_s{0,1,2}`,
`plncpair_lenet_KMNIST_DataProvider_cascaded_n1000_s{0,1,2}`.

---

## 4p. The genuine n=1000 deep-ladder reads are now CLEAN `rc=0` — the §4f / AC §1g (‡) `rc=1` crash confound is CLOSED, and the death-cascade is NOT depth-monotone (`item_id=dcnn_n1000_deathcascade_finalize`, 2026-06-25)

**Question (close the last n=1000 confound).** AC §1g landed the genuine high-resolution
(nevresim **n=1000**, 5× the n=200 ladder) deep_cnn d8/d10 reads, but only on the
**`rc=1`-confounded `pdcnndeeppair_` vehicle** (d8 +8.51pp, d10 +11.14pp — read as
CONFIRMED-WITH-CONFOUND because every run crashed downstream at `HardCoreMappingStep`
"No more hard cores available" *after* `__target_metric.json` + the NF↔SCM/torch↔sim
parity gates were written). This batch re-runs the **same genuine n=1000 d8/d10 paired
ladder on the proven CLEAN `rc=0` `bigcores` (`cores.count = 480`) vehicle** — the
`plan_stage:23` proposal — and asks: does the death-cascade survive when the runs
actually finalize `rc=0`, and is the depth ordering what §4f assumed (d8 mild, d10
collapse)?

All cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based` S=4, **`max_simulation_samples=1000`**,
3 seeds/arm paired by seed, `cores.count = 480`. **ALL 12 runs finalized `rc=0` in
`q/done/`** (`artifact_ok=true`, ZERO "No more hard cores" crash lines). Ledger:
`cluster:"WS3"`, `kind:"depth"`, `model:"deep_cnn"`.

### The CLEAN `rc=0` genuine-n=1000 deep ladder — cascaded vs synchronized

| d | cascaded deployed (3-seed mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | casc→ANN gap | sync→ANN gap | validity | verdict |
|--:|:-------------------------------------|:-------------------------|------------------:|:--------|-------------:|-------------:|:---------|:--------|
| 8  | **0.898 ± 6.00pp** (.814/.95/.93)    | 0.9928 ± 0.09pp (.993/.9917/.9938) | **−9.48** | 0.9923 | 9.43 | +0.05 | VALID `rc=0` (12/12) | firing-gain degraded |
| 10 | **0.9297 ± 2.47pp** (.907/.918/.964) | 0.9903 ± 0.18pp (.9895/.9886/.9928) | **−6.06** | 0.9907 | 6.10 | −0.04 | VALID `rc=0` (12/12) | firing-gain degraded |

(Sign convention `casc − sync`; negative = cascaded below synchronized.)

**Verdict — the §4f / AC §1g (‡) crash confound is CLOSED: the death-cascade is REAL,
VALID, and clean at n=1000 — but it is NOT depth-monotone, and the prior "d8 mild / d10
collapse" framing is REFUTED.** On a clean `rc=0` vehicle at genuine n=1000 resolution
the cascaded mode degrades **substantially at BOTH depths** (d8 −9.48pp, d10 −6.06pp)
while synchronized stays pinned at the ANN ceiling (d8 +0.05pp, d10 −0.04pp, sd ≤0.18pp,
LOSSLESS). The §4f-confound — "every d8–d12 cell finalized `rc=1` with values captured
only pre-crash" — is now CLOSED: all 12 runs are `rc=0`, `artifact_ok`, with NF↔SCM
cascaded decision agreement **1.0000** and torch↔deployed-sim parity **0.9922 (d8s0) /
1.0000 (d10s0)** → the degradation is a **genuine firing-gain deficit, not a parity/decode
bug**. ANN refs 0.988–0.9946 (mean d8 0.9923, d10 0.9907) ≫ MNIST chance 0.1135 → genuine
firing-gain, no untrained floor.

**The item's premise is CONTRADICTED on the clean reads (a FLAG, not a validity confound).**
The item assumed "d8 mild, d10 collapse." The clean rc=0 reads show the **opposite ordering**:
the d8 gap (−9.48pp) is **LARGER** than the d10 gap (−6.06pp). This **inverts** the prior
`rc=1` n=1000 ordering (where d10 +11.14pp > d8 +8.51pp). The d8>d10 ordering here is
**dominated by the d8 cascaded s0=0.814 outlier** (vs s1=0.95, s2=0.93; sd 6.00pp). Both
depths degrade by ~6–9pp; **neither is "mild" and the gap is NOT depth-monotone** on the
clean vehicle. The death-cascade *reproduces* at both depths; its *magnitude ordering* is
not stable across resolution/vehicle.

**Confounds / bounds.** (1) **EVAL-SET subsample:** the cascaded arm subsamples
**1000/10000** (~1pp/seed binomial spread; the d8 s0=0.814 sits well below s1/s2 and drives
the 6.00pp sd) while synchronized reports the full SCM eval — read the **GAPS** (−9.48pp,
−6.06pp ≫ ~1pp/seed noise), not 3rd decimals. `__target_metric.json` values track the
SCM/HCM chip eval within ~1pp (e.g. d8s0 0.814 vs SCM/HCM 0.8252); per convention the bare
`__target_metric.json` float is reported. (2) **NO at-chance confound** — ANN ≫ chance at
both depths. (3) **3 finalized seeds per cell** (`n_seeds=3`). (4) **schedule knob** =
`deployment_parameters.ttfs_cycle_schedule` cascaded-vs-synchronized; configs otherwise
byte-matched, paired by seed. This is the **clean `rc=0` + high-resolution upgrade** that
turns the §4f/§1g (‡) `rc=1` n=1000 reads into formally VALID evidence. Run ids:
`pdcnnbcn1000_d{8,10}_{cascaded,synchronized}_s{0,1,2}`. **Next:** the firing-gain gate-fix
on this clean d8/d10 n=1000 anchor — but note the gate-fix is REFUTED as a deep auto-rescue
(see WS7 §10, `dcnn_d10_gatefix_rescue`): θ-cotrain *crashes* the convnet and `cp:true`-only
deploys ~0.79 < the cascaded baseline. Synchronized remains the unconditional deep default.

---

## 4q. The genuine n=1000 deep ladder is EXTENDED to d6 and d12 — on the full `pdcnnbcn1000*` vehicle the cascaded→sync gap is DEPTH-MONOTONE (3.61→4.62→5.84→9.48pp), reconciling the §4p two-point d8>d10 read (`item_id=dcnn_mnist_depth_deathcascade_n1000`, 2026-06-25)

**Question (close the d6/d12 endpoints of the n=1000 ladder).** §4p firmed the genuine
n=1000 reads at d8/d10 but read them as *non-monotone* (d8 −9.48pp > d10 −6.06pp) on the
two-point `pdcnnbcn1000_` slice. This item adds the **d6 onset** and the **d12 collapse**
rungs and pools the per-depth seeds across the `pdcnnbcn1000plat_*` (s0–s2) and
`pdcnnbcn1000seed_*` (s3–s5) run-id families to ask: across the full d6→d12 band, is the
death-cascade depth-monotone, and does synchronized hold the ANN ceiling at every rung?

All cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based` S=4, `cores.count=480`,
3-or-6 seeds/arm paired by seed. **All 39 in-scope runs finalized `rc=0` in `q/done/`**
(the `rc=−9` OOM-killed `pdcnnbcclean_d12_cascaded` seeds in `q/failed/` are NOT the d12
cascaded source — d12 cascaded comes from the clean `pdcnnbcd12fin_*` family). Ledger:
`cluster:"WS3"`, `kind:"depth"` (4 rungs) + one `kind:"synthesis"` row.

### The full genuine-n=1000 within-CNN depth ladder — cascaded vs synchronized

| d | cascaded deployed (mean ± sd) | synchronized (mean ± sd) | **casc→sync GAP** | ANN ref | sync→ANN | n_seeds | resolution |
|--:|:------------------------------|:-------------------------|------------------:|:--------|---------:|--------:|:-----------|
| 6  | 0.9563 ± 1.16pp | 0.9924 ± 0.15pp | **+3.61** | 0.9914 | −0.10 | 3 | casc/sync n1000 |
| 8  | 0.9450 ± 2.09pp | 0.9912 ± 0.13pp | **+4.62** | 0.9914 | −0.06 | 6 (pooled) | casc/sync n1000 |
| 10 | 0.9328 ± 3.76pp | 0.9912 ± 0.15pp | **+5.84** | 0.9924 | +0.08 | 6 (pooled) | casc/sync n1000 |
| 12 | 0.8967 ± 4.40pp | 0.9915 ± 0.07pp | **+9.48** | 0.9916 | −0.15 | 3 | casc/sync **n200** |

(Sign convention `sync − casc`; positive = cascaded below synchronized.)

**Verdict — SUPPORTED: the death-cascade is depth-MONOTONE on this n=1000 vehicle, and
synchronized holds the float ANN ceiling at EVERY rung.** The cascaded→sync gap widens
monotonically 3.61→4.62→5.84→9.48pp while `|sync→ANN| ≤ 0.15pp` everywhere (lossless). The
deep_mlp depth-monotone widening law **reproduces on the VALID deep_cnn vehicle**. Every ANN
ref ≈ 0.99 ≫ MNIST chance 0.1135 → genuine firing-gain, not an untrained floor; the d12
cascaded outlier is FIRMED at 3 clean `rc=0` seeds (0.835/0.92/0.935, mean 0.8967).

**Reconciling §4p.** §4p read the SAME phenomenon as *non-monotone* because it used only the
two-point `pdcnnbcn1000_` slice (d8 s0=0.814 outlier drove d8>d10). Pooling the
`plat_*`+`seed_*` families here (d8 plat 0.9553 / seed 0.9347; d10 plat 0.9313 / seed 0.9343
— **families agree**) restores the monotone reading. The death-cascade **direction**
(cascaded ≪ sync, deepening overall) is invariant across §4f/§4h/§4m/§4p; only the
per-rung **magnitude ordering** is vehicle/resolution-dependent.

**Confounds / bounds.** (1) **Resolution split at d12:** d6–d10 are n1000 on BOTH arms; the
d12 cells are **n200 on both arms** (`pdcnnbcd12fin` cascaded + `pdcnnbcclean` synchronized)
— read the d12 +9.48pp as a *gap*, not a 3rd-decimal. A cross-res n1000 d12 sync family
(`pdcnnbcn1000seed_d12_synchronized`, mean 0.9920) gives a near-identical **+9.54pp**,
confirming the d12 sync arm is resolution-robust; no n1000 d12 cascaded arm exists in scope.
(2) **Pooled families at d8/d10** (6 seeds each) — justified, the two families agree and the
plat-only 3-seed gaps (d8 +3.65, d10 +6.04pp) tell the same monotone story. (3) **High
cascaded seed variance at deep rungs** (d10 sd 3.76pp incl. an s2=0.99 high outlier and
s1=0.869 low; d12 sd 4.40pp) inflates per-seed spread, but every gap is ≫ 2× the per-seed
noise. (4) **No at-chance confound** — every ANN ≈ 0.99; parity gates clean. **Next:** the
firing-gain gate-fix is REFUTED as a deep auto-rescue (WS7 §10) — synchronized remains the
unconditional deep default; the open question is whether an *in-loop* lever (depth-aware
surrogate / staircase-backward STE) recovers the deep cascaded rung where the schedule does
not. Run ids: `pdcnnbcn1000plat_d{6,8,10}_*`, `pdcnnbcn1000seed_d{8,10}_*`,
`pdcnnbcd12fin_cascaded_s{0,1,2}`, `pdcnnbcclean_d12_synchronized_s{0,1,2}`.

---

## 4l. SYNTHESIS — the two CONFIRMED deep_cnn items, with the closeout §6 "monotone" framing corrected (2026-06-24)

Cross-round consolidation of the verified deep_cnn death-cascade items into two
synthesis ledger rows (`kind="synthesis"`, citing every rung run_id). This
section records the CORRECTED verdicts; it does not restate the per-rung
sections 4f–4k.

### Item A — `deep_cnn` MNIST depth ladder (`item_id=deep_cnn_depth_ladder`)

VALID vehicle: ANN ~0.99 at every depth; synchronized ~lossless `==`ANN at every
depth. The cascaded single-spike-TTFS death-cascade is REAL with a **sharp
d5→d6 onset**, but it is **NOT monotonically widening** beyond d6.

| depth | cascaded mean | sync mean | ANN | casc→sync gap (pp) | n_casc seeds |
|------:|--------------:|----------:|----:|-------------------:|-------------:|
| 5 | 0.9917 | 0.9924 | 0.9925 | 0.07 | 3 |
| 6 | 0.9383 | 0.9904 | 0.9921 | **5.21** | 3 |
| 8 | 0.945  | 0.9935 | 0.9934 | 4.85 | 3 |
| 10 | 0.9517 | 0.9917 | 0.9923 | 4.00 | 3 |
| 12 | 0.98 | 0.9917 | 0.9887 | 1.17 | **1** |

- **Verdict: CONFIRMED-WITH-CONFOUND.** Robust claim = sharp d5→d6 ONSET of a
  SUSTAINED ~4–5pp gap. The closeout §6 "monotonically widening" framing is
  **REFUTED in literal form**: the gap is 5.21/4.85/4.00pp at d6/d8/d10 (shrinks,
  not grows), driven by high cascaded seed variance (per-seed range up to 0.06).
- **d12 INCONCLUSIVE:** cascaded n=1 — seeds s0,s2 are `rc=-9` timed_out at the
  3600s wall (OOM/timeout, NOT methodology). The lone surviving seed (0.98,
  1.17pp gap) is a survivor artifact and cannot be claimed.
- `max_simulation_samples=200` → cascaded deployed lands on the 0.005 grid; read
  gaps, not 3rd decimals. d6+ from the `pdcnnbc_` 4× hard-core re-run
  (`pdcnnladder_d6/d7` failed `rc=1`, excluded); d5 from `pdcnnladder_d5` (`rc=0`).

### Item B — `deep_cnn` dataset axis, FashionMNIST + KMNIST (`item_id=deep_cnn_dataset_axis`)

VALID vehicle: ANN ~0.93 FMNIST / ~0.97 KMNIST (≫0.10 chance); synchronized
near-lossless (FMNIST sync→ANN ~2.9pp, KMNIST sync→ANN ~0.5–0.8pp).

| depth | FMNIST casc→sync (pp) | KMNIST casc→sync (pp) |
|------:|----------------------:|----------------------:|
| 5  | 6.03 | 4.62 |
| 6  | 6.11 | 7.94 |
| 8  | 11.34 | 7.02 |
| 10 | 17.91 | 16.0 |

- **Verdict: SUPPORTED (with caveats).** The cascaded→sync gap widens with depth
  across BOTH datasets, and the dataset axis holds (sync higher on
  lower-margin KMNIST than FMNIST; cascaded gap large on both). **FMNIST is
  STRICT monotone** (6.03→6.11→11.34→17.91pp). **KMNIST widens overall** with a
  d6→d8 dip (7.94→7.02pp) that is **within 200-sample noise** (~0.5pp resolution).
- **Deepest cascaded rungs are n=2:** d10 FMNIST cascaded s1 `rc=-9` OOM-kill;
  d10 KMNIST cascaded s0 `rc=1` crash.
- NOT chance/firing-gain-at-chance — every ANN ≫ 0.10.

**Net:** the death-cascade depth-law reproduces on a valid deep_cnn vehicle and
tracks dataset margin, but the honest framing is **sharp depth-threshold onset +
dataset-margin amplification**, not a clean monotone curve — and the d12 MNIST
rung remains unmeasured.

---

## 4v. CONSOLIDATED — the full clean `rc==0` bigcores MNIST depth ladder, both resolutions, with the d12 rung CLOSED: sharp d6 onset → BOUNDED ~4–7pp plateau, NOT monotone-widening (`item_id=deep_cnn_depth_cascade_ladder_mnist`, 2026-06-25)

This single section consolidates **every clean `rc==0` `pdcnnbc*` bigcores rung**
of the within-CNN MNIST cascade ladder into one item, at **both** the n200
(0.005-grid) and the genuine **n1000** (0.001-grid, 5× resolution) read, and
**closes the d12 cascaded rung** that §4l Item A left at n=1 inconclusive. It
SUPERSEDES the per-rung 4f/4l reads for the MNIST depth ladder and corrects the
closeout §6 framing for the CNN.

All cells: `deep_cnn` (width 16), MNIST, `ttfs_cycle_based`, S=4, paired
cascaded-vs-synchronized by seed, on the **`bigcores` config (`cores.count = 480`,
480/480)** that clears the d10/d12 `HardCoreMappingStep` "No more hard cores
available" packing crash so the ladder finalizes `rc==0` with mapping numerics
unchanged. ANN ~0.99 at every depth (≫ MNIST chance 0.1135) ⇒ every gap is a
**genuine firing-gain** read, not an untrained-floor artifact. Ledger:
`cluster:"WS3"`, `kind:"depth"`, `item_id:"deep_cnn_depth_cascade_ladder_mnist"`.

### The full MNIST depth ladder — cascaded vs synchronized, both resolutions

| d | casc n200 | casc n1000 | sync (n200/n1000) | ANN | **casc→sync gap n200** | **casc→sync gap n1000** | verdict |
|--:|----------:|-----------:|:------------------|----:|-----------------------:|------------------------:|:--------|
| 4  | 0.9883 | — | 0.9898 | 0.9931 | **+0.15** | — | tied to sync (pre-onset) |
| 5  | 0.9917 | — | 0.9924 | 0.9913 | **+0.07** | — | tied to sync (last pre-onset) |
| 6  | 0.9383 | 0.9563 | 0.9904 / 0.9924 | 0.992 | **+5.21** | **+3.61** | sharp ONSET |
| 8  | 0.9483 | 0.9293 | 0.9934 / 0.9918 | 0.992 | **+4.50** | **+6.24** | degraded, bounded |
| 10 | 0.9525 | 0.9318 | 0.9925 / 0.9909 | 0.992 | **+4.00** | **+5.91** | degraded, bounded |
| 12 | 0.9175 ‡ | 0.9353 | 0.9916 / 0.9920 | 0.992 | **+7.41** ‡ | **+5.67** | degraded, bounded (d12 CLOSED) |

‡ d12 **n200** cascaded is a **cross-vehicle pool** (1 `pdcnnbc_` s1=0.980 + 3
`pdcnnbcd12fin_` 0.835/0.920/0.935) because the same-vehicle `pdcnnbc_`/`pdcnnbcclean_`
d12 cascaded OOM-crashed (`rc=-9`, NOT firing-gain). The **d12 n1000** read is the
**CLEANEST** d12 cell: a same-vehicle `pdcnnbcn1000seed_` s3/4/5 3-seed paired
cascaded-vs-sync (gap **+5.67pp**) — prefer it for the headline.

### Verdict — `cascaded_firing_gain_degraded_bounded_plateau_sharp_d6_onset`

The within-CNN cascade follows **pattern (b)**: **byte-tied to synchronized and the
ANN ceiling at d4/d5** (gap +0.15 / +0.07pp, near-lossless), then a **SHARP onset at
d6** drops cascaded to a **BOUNDED ~4–7pp plateau** that **does NOT widen
monotonically through d12** (n200 gaps 5.21→4.50→4.00→7.41; n1000 gaps
3.61→6.24→5.91→5.67pp), while synchronized holds ~0.991–0.993 (within ~0.1–0.3pp of
the ANN ceiling) **flat at every depth**. This is:

- **NOT the deep_mlp-style monotonic widening** (pattern (a): the INVALID
  host-majority deep_mlp shows +4.3pp at d4 → +9.3pp at d8, a smooth climb), and
- **NOT absent** (pattern (c): §4c's "no within-CNN cascade" reading was a
  *shallow-depth d4/d5 artifact*, corrected here and in §4f).

Synchronized is the **unconditional deep default on the CNN** (gap to ANN ≤0.3pp,
seed sd ≤0.16pp at every depth). The conv-shared/pooled structure **delays** the
onset (d6 vs the MLP's d4) but does **not** abolish it; the deficit tracks the
length of the greedy single-spike partial-sum chain.

### Confounds / bounds

1. **Resolution / metric provenance.** Cascaded subsamples to
   `max_simulation_samples` (n=200 → 0.005 grid; n=1000 → 0.001 grid) while
   synchronized uses the **FULL 10k** test set — so **read the GAPS, not the third
   decimals**. The n200 vs n1000 cascaded means differ ~1–2pp at matched depth, and
   cascaded seed sd is high (1–5pp) vs synchronized (~0.1–0.2pp) — a fragile,
   high-variance code consistent with the death-cascade framing. Higher resolution
   does **NOT** move the gap toward zero (n1000 ≥ n200 at d8/d10) → the depth-law
   **hardens** with resolution.
2. **d12 n200 cross-vehicle pool** (see ‡ above): its +7.41pp ladder-max is driven by
   the 0.835 `pdcnnbcd12fin_` outlier; the clean same-vehicle d12 n1000 read (+5.67pp)
   sits squarely **IN** the plateau and is the load-bearing d12 number.
3. **EXCLUDED from this ladder** (different sub-questions / infra confounds): the WS7
   gate-fix grids `pdcnnbcn1000fix_`/`pdcnnd6fix_` (cp/cot levers, all cascaded-only,
   no synchronized counterpart; `cotTrue` arms `rc=1`-crash on
   `Conv2DPerceptronMapper`); and the `rc=1`/`rc=-9` ladders `dcnn_d6/d8`,
   `pdcnnladder_d6/d7`, `pdcnnd67retry_`, `pdcnnbc_d12 s0/s2`, `pdcnnbcclean_d12`
   (the d10/d12 packing crash at `HardCoreMappingStep` — an infra/packing confound;
   the deployed metric there was written PRE-crash).
4. Every underlying clean run passed parity (NF↔SCM cascaded agreement 1.0,
   torch↔deployed-sim 0.9961–1.0) before finalizing `rc==0`.

**This corrects §4l Item A** (which read d12 as n=1 INCONCLUSIVE and the gap as
"shrinks d6→d10"): with the d12 n1000 rung closed, the honest law is a **sharp d6
onset followed by a bounded, non-monotone ~4–7pp plateau through d12** — the
closeout §6 "monotonically widening" framing is **REFUTED in literal form for the
CNN** (it holds only on the INVALID deep_mlp and on the *dataset-axis* FMNIST ladder
of §4l Item B).

---

## 4x. POOLED-BATCH confirmation of the §4v plateau — d8/d10 over n=9 (three independent seed batches base/plat/seed) re-confirms `cascaded→sync` SATURATES, and isolates the d8 base-batch outlier (`item_id=deep_cnn_depth_cascade_ladder_mnist`, synthesis row, 2026-06-25)

This subsection pools the §4v genuine-n1000 d8/d10 rungs over **three independent
seed batches** (`pdcnnbcn1000_` s0–2 *base* + `pdcnnbcn1000plat_` s0–2 *plat* +
`pdcnnbcn1000seed_` s3–5 *seed*, **n=9 per rung**) and re-anchors the shallow control
(d4 n200) and the deep close (d6/d12 n1000). It does **not** supersede §4v; it
hardens the plateau verdict against the d8 base-batch noise and records the
consolidated synthesis ledger row.

### The pooled MNIST depth ladder — cascaded vs synchronized

| d | n | mss | casc mean | sync mean | ANN | **casc→sync gap** | per-batch gap (base/plat/seed) |
|--:|--:|----:|----------:|----------:|----:|------------------:|:-------------------------------|
| 4  | 3 |  200 | 0.9883 | 0.9898 | 0.992  | **+0.15** | — (shallow control; read gap not 3rd dec) |
| 6  | 3 | 1000 | 0.9563 | 0.9924 | 0.9914 | **+3.61** | plat only |
| 8  | 9 | 1000 | 0.9293 | 0.9918 | 0.9917 | **+6.24** | 9.48 / 3.65 / 5.59 |
| 10 | 9 | 1000 | 0.9318 | 0.9909 | 0.9918 | **+5.91** | 6.06 / 6.04 / 5.63 |
| 12 | 3 | 1000 | 0.9353 | 0.9920 | 0.9904 | **+5.67** | seed only |

### Verdict — PLATEAU holds; the d8 pooled mean is INFLATED by one base-batch read

The cascaded→sync gap rises steeply **d4(0.15)→d6(3.61)→d8(~6.2)** then **saturates**
at d8/d10(5.91)/d12(5.67) — `d10 ≤ d8` and `d12 ≤ d10`, so **monotone-widening is
refuted regardless of how the d8 outlier is treated**. The pooled d8 gap **+6.24pp** is
dragged up by a single noisy base-batch seed (`pdcnnbcn1000_d8_cascaded_s0 = 0.814`,
verified artifact); the **clean per-batch d8 gaps are 3.65 (plat) / 5.59 (seed)pp**, and
d10's three batches agree tightly at **5.63–6.06pp**. The synchronized arm holds
**~lossless (==ANN 0.990–0.992)** at every depth. Cascaded per-seed variance is high
(d8 0.814–0.967, d10 0.869–0.99, d12 0.887–0.977), consistent with a fragile cascade,
but the depth-trend of the MEAN gap is **flat/saturating, not rising**.

### Confounds / bounds

1. **No untrained-floor confound:** ANN ~0.99 (0.985–0.996) at every cell ≫ MNIST
   chance 0.1135 → every gap is a genuine firing-gain read.
2. **Resolution:** d6/d8/d10/d12 at `max_simulation_samples=1000` (read 2nd–3rd decimal
   cautiously, but 3.6–6.2pp gaps are far above resolution noise); the **d4 shallow
   control is n200** → read its +0.15pp as ~lossless, not a 3rd-decimal claim.
3. **All 54 matched runs finalized `rc=0`** (done queue; zero failed-dir matches).
   d8/d10 pool n=9 across base/plat/seed; d4/d6/d12 are n=3.
4. This is a **consolidation** of the §4v per-rung reads — the per-cell rows already
   live in the ledger under `item_id=deep_cnn_depth_cascade_ladder_mnist`; the new
   synthesis row (`kind:"synthesis"`) cites all 54 run_ids for explicit coverage.

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

---

## 4t. The d6 deep_cnn gate-fix 2×2 DECOMPOSES — `ttfs_staircase_ste` is the dominant knob and roughly HALVES the deficit, but neither knob nor their combination reaches the ANN/sync ceiling (`item_id=dcnn_d6_ste_gatefix_decomposition`, 2026-06-25)

**Question (which gate-fix knob carries the rescue, and does it close the gap?).**
On the VALID on-chip-majority `deep_cnn` d6 MNIST cascaded vehicle, decompose the
firing-gain gate-fix into a 2×2 over `ttfs_staircase_ste ∈ {F,T}` ×
`conversion_policy ∈ {F=pure cascaded, T=synchronized/conversion route}`. Which
knob is dominant, and does the best combo reach the float ANN ceiling?

Runs: `pdcnnd6stefix_ste{True,False}_cp{True,False}_s{0,1,2}` (**12 runs, all `rc=0`**,
full 3 seeds/cell, `deep_cnn` w16, S=4, `ttfs_cycle_based`, cascaded
`max_simulation_samples=200`). Ledger: `cluster:"WS3"`, `kind:"escalation"`,
`item_id:"dcnn_d6_ste_gatefix_decomposition"`. NOTE: the prior round on this exact
onset cell found `ttfs_theta_cotrain` **rc=1-CRASHES** on the convnet
(`Conv2DPerceptronMapper` forward bug); this batch swapped in `ttfs_staircase_ste`,
which runs `rc=0` and IS the working (partial) lever where θ-cotrain was unusable.

### The d6 MNIST gate-fix 2×2 — deployed mean (3 seeds)

ANN refs ~0.992 ≫ 0.10 chance → genuine firing-gain deficit, not an untrained-floor artifact.

| cell | ste | cp | deployed/seed | **deployed mean** | ANN mean | **ANN gap** |
|:-----|:----|:---|:--------------|------------------:|---------:|------------:|
| steTrue_cpTrue (best) | T | T | .96/.96/.985 | **0.9683** | 0.9923 | **+2.40** |
| steTrue_cpFalse | T | F | .945/.99/.96 | 0.9650 | 0.9907 | +2.57 |
| steFalse_cpTrue | F | T | .97/.97/.94 | 0.9600 | 0.9930 | +3.30 |
| steFalse_cpFalse (worst, pure cascaded) | F | F | .96/.955/.93 | **0.9483** | 0.9920 | **+4.37** |

**Knob effects (pp):** STE lift = **+0.83** at cpTrue, **+1.67** at cpFalse;
conversion_policy lift = **+0.33** at steTrue, **+1.17** at steFalse.

**Verdict — `ttfs_staircase_ste` DOMINANT, partial rescue, NOT lossless.** The STE
gradient is the larger knob on both columns (+0.83 / +1.67pp) and roughly **halves**
the death-cascade deficit (worst pure-cascaded +4.37pp → best combo +2.40pp). But
the best combo (steTrue+cpTrue, 0.9683) does **NOT** reach the ~0.992 ANN/sync
ceiling — a **+2.40pp residual deficit remains**. With STE ON the conversion-route
adds only **+0.33pp** (WITHIN n=200 noise) — STE has already substituted for the
conversion-route rescue. conversion_policy alone is a **weak partial lever** (+1.17pp
at steFalse, still leaving +3.30pp).

**Confounds / bounds.** (1) **n=200 → 0.005 deployed-accuracy grid:** single-seed
swings (e.g. steTrue_cpFalse .945/.99/.96) are ~1–2 samples of noise; per-cell
sub-pp differences (notably the casc→sync +0.33pp under STE) are WITHIN-noise and
must be read at the multi-pp scale — but the +2.40pp best-combo ANN gap and the
+1.67pp STE lift at cpFalse **exceed** this resolution. (2) **No at-chance confound:**
ANN ~0.9915 (range 0.9875–0.9941) on every run ≫ 0.10 → genuine firing-gain. (3) All
12 runs `rc=0`, full 3 seeds/cell, none crashed. (4) **Eval-set asymmetry:** deployed
acc on n=200 cascaded-eval vs the n=10000 ANN test set. (5) θ-cotrain remains the
unusable lever on this convnet (rc=1); STE is the working swap. **Next:** lift the STE
arm to genuine n=1000 to firm the +2.40pp residual below the grid, and test whether a
deeper STE-mix or per-channel θ trim closes the residual on the harder FMNIST onset.

---

## 4u. The cascaded death-cascade reproduces on the VALID deep_cnn vehicle across {FMNIST,KMNIST}×{d6,d8} at n=200 — depth-law HOLDS on FMNIST, FLAT on KMNIST; dataset-margin separation only at d8 (`item_id=dcnn_dataset_depth_deathcascade_valid_vehicle`, 2026-06-25)

**Question (does the death-cascade reproduce on the VALID vehicle, replacing the retired INVALID deep_mlp §10.1 table?).**
The headline depth × firing-gain death-cascade table was originally measured on the
**INVALID host-majority** `deep_mlp` (retired, see VALIDITY_AUDIT). Does it reproduce on
the VALID on-chip-majority `deep_cnn` (w16) across {FMNIST,KMNIST}×{d6,d8}, and does the
cascaded→ANN firing-gain gap widen with depth and with dataset margin?

Runs: `pdcnndatafix_d{6,8}_{FashionMNIST,KMNIST}_DataProvider_cotFalse_s{0,1,2}`
(**11 valid `rc=0` cascaded runs**, `deep_cnn` w16, S=4, `ttfs_cycle_based`, cascaded
schedule, `ttfs_theta_cotrain=False`, `allow_coalescing=True`,
`max_simulation_samples=200`). Ledger: `cluster:"WS3"`, `kind:"arch_dataset"`,
`item_id:"dcnn_dataset_depth_deathcascade_valid_vehicle"`. **This batch has NO
synchronized arm** (the only axis is cotrain T/F), so `cascaded_to_sync_gap_pp` is NULL
for every cell and the gap reported is **cascaded→ANN** only.

### The dataset×depth death-cascade on the VALID vehicle — cascaded→ANN gap

ANN refs 0.93–0.97 ≫ 0.10 chance → genuine firing-gain, not an untrained-floor artifact.

| dataset | depth | cascaded n200 (mean ± sd) | ANN ref | **casc→ANN GAP** | verdict |
|:--------|:------|:--------------------------|:--------|-----------------:|:--------|
| fmnist | 6 | 0.855 ± 3.19pp (.81/.88/.875) | 0.9304 | **+7.54** | firing-gain degraded |
| fmnist | 8 | 0.7675 ± 0.75pp (.76/.775)† | 0.9356 | **+16.81** | firing-gain **collapse** (widens sharply) |
| kmnist | 6 | 0.8967 ± 1.93pp (.915/.87/.905) | 0.9698 | **+7.31** | firing-gain degraded |
| kmnist | 8 | 0.8967 ± 1.03pp (.885/.91/.895) | 0.9702 | **+7.35** | firing-gain degraded (**FLAT** vs d6) |

† FMNIST d8 is **n=2** (s2 failed `rc=1`); all other cells n=3.

**Verdict — SUPPORTED-WITH-CONFOUND. Depth-law HOLDS on FMNIST, FLAT on KMNIST.** The
cascaded→ANN firing-gain gap widens **sharply** with depth on FMNIST (**+7.54pp @ d6 →
+16.81pp @ d8**, deployed 0.855 → 0.768) and widens with **dataset margin at d8**
(FMNIST +16.81pp ≫ KMNIST +7.35pp). But on KMNIST the depth axis is **FLAT**
(+7.31 → +7.35pp, deployed 0.897 → 0.897) — the depth-widening law does NOT hold on
KMNIST. Dataset-margin separation is clear **only at d8** (at d6 the two datasets are
~7.3–7.5pp, indistinguishable at this resolution). The death-cascade reproduces on the
VALID vehicle, **replacing the retired INVALID host-majority deep_mlp §10.1 table**.

**Confounds / bounds.** (1) **NO synchronized arm** → `synchronized_deployed_mean` and
`cascaded_to_sync_gap_pp` are NULL for every cell; the gap is cascaded→ANN only. (2) The
**gate-fix cotTrue arm: all 12 runs failed `rc=1`** (Conv2DPerceptronMapper crash), not
consolidatable. (3) **FMNIST d8 n=2** (s2 failed rc=1); all other cells n=3. (4)
**n=200 → 0.005 grid:** read gaps not 3rd decimals; cascaded seed spread inflated. (5)
**KMNIST depth axis FLAT** (7.31 ≈ 7.35pp) → depth-widening fails on KMNIST; only FMNIST
shows it. (6) 3 duplicate d8_KMNIST queue JSONs (id == filename) excluded by strict rule.
(7) **NO at-chance confound:** ANN 0.93–0.97 ≫ 0.10. **Next:** add the synchronized arm
at these dataset×depth cells (to convert the cascaded→ANN gap into a cascaded→sync gap)
and fix the Conv2DPerceptronMapper forward bug to unlock the cotTrue gate-fix arm.

---

## 4w. The §4u open gap is CLOSED — matched-batch SYNCHRONIZED companions at d8/d10 convert cascaded→ANN into cascaded→SYNC, and the death-cascade law holds across the dataset-breadth × depth corpus (`item_id=dcnn_dataset_breadth_depth`, 2026-06-25)

**Question (does the cascaded firing-gain death-cascade law hold across the dataset-breadth
axis on the VALID deep_cnn vehicle, with a synchronized ceiling to separate firing-gain from
capacity?).** §4u reproduced the death-cascade on the VALID vehicle but had **NO synchronized
arm** at the dataset×depth cells, so the deficit was only measurable as cascaded→ANN. This
item lands the **matched-batch synchronized companions** at d8 (`pdcnnd8databc`) and d10
(`pdcnnd10data`) and pools the d6 and gate-fix grid cells — closing the §4u "add the
synchronized arm" gap and giving a clean **cascaded→synchronized firing-gain gap** at every
matched cell. `deep_cnn` (w16), S=4, `ttfs_cycle_based`, `max_simulation_samples=200`. Ledger:
`cluster:"WS3"`, `kind:"arch_dataset"`, `item_id:"dcnn_dataset_breadth_depth"` (9 rows).

### The dataset-breadth × depth death-cascade — cascaded vs synchronized (matched batch)

ANN refs 0.93–0.97 ≫ 0.10 chance → genuine firing-gain at every cell (NO untrained-floor).

| dataset | depth | cascaded (mean, n) | synchronized (mean, n) | **casc→sync GAP** | casc→ANN | sync→ANN | verdict |
|:--------|:------|:-------------------|:-----------------------|------------------:|---------:|---------:|:--------|
| FashionMNIST | 6  | 0.855  (n=3) | — | — | +7.54 | — | cascaded-only (no sync companion) |
| KMNIST       | 6  | 0.9183 (n=3) | — | — | +5.58 | — | cascaded-only (smallest deficit: shallow + easy) |
| FashionMNIST | 8  | 0.790  (n=3) | 0.9034 (n=3) | **+11.34** | +14.32 | 2.98 | death-cascade; harder margin = larger gap |
| KMNIST       | 8  | 0.8917 (n=3) | 0.9619 (n=3) | **+7.02**  | +7.57  | 0.55 | sync near-lossless (0.55pp) |
| FashionMNIST | 10 | 0.725  (n=2†) | 0.9041 (n=3) | **+17.91** | +20.86 | 2.95 | **WORST corner** (deep × hard) |
| KMNIST       | 10 | 0.8025 (n=2†) | 0.9623 (n=3) | **+15.98** | +15.91 | −0.07‡ | sync near-lossless |

† d10 cascaded is **n=2** (FMNIST s1 crashed `rc=-9`, KMNIST s0 crashed `rc=1`); sync arms full
n=3. ‡ KMNIST d10 sync 0.9623 marginally exceeds its 200-sample ANN ref 0.9616 (−0.07pp) — a
**sampling artifact** of the coarse n=200 ANN eval, not a super-ANN result.

### The gate-fix grid does NOT close the deep × hard deficit

| dataset | depth | gate-fix cascaded (cotF cpF) | sync ceiling (d10data) | casc→sync GAP | verdict |
|:--------|:------|:-----------------------------|:-----------------------|--------------:|:--------|
| FashionMNIST | 10 | 0.750 (n=2) | 0.9041 | **+15.41** | gate-fix knobs do NOT recover |
| KMNIST       | 10 | 0.8625 (n=2) | 0.9623 | **+9.98** | cpTrue s2=0.865 ~same; no recovery |

**Verdict — CONFIRMED. The death-cascade law holds across FashionMNIST + KMNIST × d6/d8/d10;
the harder dataset-margin carries the LARGER cascaded deficit at every matched depth, and
synchronized stays ~lossless.** The cascaded→sync firing-gain gap grows **monotonically with
depth** on BOTH datasets (FMNIST +11.34pp@d8 → +17.91pp@d10; KMNIST +7.02pp@d8 → +15.98pp@d10)
and is **consistently larger on the harder FMNIST margin** (lower ANN ceiling) than on KMNIST
at every matched depth. Synchronized deployment stays within **0.55–2.98pp** of the well-above-
chance ANN ceiling at every cell — so the cascaded deficit is a **firing-gain pathology, not a
capacity limit**. The §4u open gap ("add the synchronized arm") is **CLOSED at d8/d10**: the
cascaded→ANN gaps now carry a matched cascaded→sync decomposition, and the sync→ANN residual
is ≤2.98pp everywhere. The gate-fix cot/cp grid (cotFalse cpFalse) sits at the **same depressed
cascaded level** (~0.75 FMNIST, ~0.86 KMNIST), 10–15pp below the sync ceiling — it
**corroborates, does not close**, the deep × hard deficit (matching §2e/§2f: no config-level
firing-gain rescue on the convnet).

**Confounds / bounds.** (1) **No d5 runs exist in this corpus** (depths present: d6/d8/d10) —
the d5 leg is UNANSWERED here. (2) **d10 cascaded n=2** (one crash each: FMNIST s1 `rc=-9`,
KMNIST s0 `rc=1`); sync arms full n=3. (3) **d6 cells are cascaded-only** (no synchronized
companion in this batch) so the d6 casc→sync gap is not directly measurable — only casc→ANN
(~5.6–7.5pp). The duplicate d6 KMNIST cell (`pdcnndatafix_d6 cotFalse` = 0.8967, casc→ANN
+7.31pp) is consistent with `pdcnnd6kmfin` (0.9183). (4) **All 45 valid runs use
`max_simulation_samples=200`** — deployed accs are on a 0.005 grid; **trust pp-gaps, not 3rd
decimals**. (5) **NO at-chance confound:** every ANN ref 0.93–0.98 ≫ 0.10, so every cell is a
genuine firing-gain result, not an untrained floor. (6) The gate-fix `pdcnnd10datafix` cot/cp
grid (n≤2) is **cascaded-only** with the firing-gain gate knobs — corroborating, not closing,
the deficit. **This consolidates §4u + AC §1h/§1j into a single dataset-breadth × depth
death-cascade item with the synchronized ceiling attached; FashionMNIST × d10 is the worst
corner in the entire deep_cnn table.** (Run ids: see ledger
`item_id:"dcnn_dataset_breadth_depth"`.)

---

## 4y. The dataset-margin ORDERING law spans d5/d6/d8 (incl. the §4w-UNANSWERED d5 leg) — FMNIST > KMNIST cascaded deficit at EVERY depth, monotone on FMNIST, non-monotone on KMNIST (`item_id=deep_cnn_dataset_axis_death_cascade`, synthesis row, 2026-06-25)

This subsection consolidates the **`pdcnnd5data` / `pdcnnd6databc` / `pdcnnd8databc`**
dataset-axis corpus into one synthesis row spanning **d5/d6/d8** — supplying the **d5
leg that §4w confound #1 explicitly leaves UNANSWERED** — and states the dataset-margin
ordering law across all three depths. `deep_cnn` (w16), S=4, `ttfs_cycle_based`,
**`max_simulation_samples=200`**, paired cascaded-vs-synchronized by seed (only
`ttfs_cycle_schedule` differs). 36 matched runs, all `rc=0`, 3 seeds/cell.

### The dataset-margin × depth cascaded deficit — cascaded→ANN gap

ANN refs FMNIST ~0.93 / KMNIST ~0.97 ≫ 0.10 chance → genuine firing-gain (no untrained floor).

| dataset | depth | casc→ANN gap | sync→ANN gap | ordering |
|:--------|:------|-------------:|-------------:|:---------|
| FashionMNIST | 5 | **+8.89** | +2.87 | FMNIST > KMNIST |
| KMNIST       | 5 | **+5.29** | +0.48 | (easier = smaller) |
| FashionMNIST | 6 | **+9.00** | +2.96 | FMNIST > KMNIST |
| KMNIST       | 6 | **+8.42** | +0.68 | |
| FashionMNIST | 8 | **+14.30** | +2.90 | FMNIST > KMNIST (collapse) |
| KMNIST       | 8 | **+7.71** | +0.55 | |

### Verdict — CONFIRMED-WITH-CAVEAT: ordered by dataset margin, depth-monotone only on FMNIST

The cascaded firing-gain deficit **orders by dataset margin** — the harder FMNIST
(ANN~0.93) carries a **larger cascaded→ANN gap than KMNIST (ANN~0.97) at every depth**
(d5 8.89>5.29, d6 9.00>8.42, d8 14.30>7.71pp) — while the synchronized schedule holds
**near-ANN throughout** (FMNIST 2.87–2.96pp, KMNIST 0.48–0.68pp, no depth trend). Depth
growth is **clean and monotone on FMNIST** (degraded d5/d6 → collapse d8: 8.89→9.00→14.30)
but **non-monotone on KMNIST** (5.29→8.42→7.71; d6 > d8). The reproduction on this VALID
deep_cnn vehicle confirms the death-cascade is a dataset-margin-modulated, depth-amplified
**cascaded-only** deficit, not a training-floor/untrained-ANN artifact.

### Confounds / bounds

1. **n=200 → 0.005 grid:** read GAPS, not 3rd decimals (cascaded values land on the grid).
2. **KMNIST non-monotonicity sits in the noise:** KMNIST cascaded seed-std is elevated
   (d6 2.48pp, d5 1.03pp), which fully explains the non-monotone d6>d8 — depth-monotonicity
   is only **clean on FMNIST**.
3. **No at-chance confound:** ANN FMNIST ~0.93 / KMNIST ~0.97 ≫ 0.10; `ann_test_acc_mean`
   = mean of the first `Test accuracy:` log line across all 6 runs per (depth,dataset) cell.
4. **Complement, not duplicate:** §4w lands the d6/d8/d10 **cascaded→SYNC** decomposition
   but states "No d5 runs exist in this corpus"; this row supplies the **d5 leg** and the
   cross-depth cascaded→ANN ordering at matched n200. Per-cell rows already live in the
   ledger (`dcnn_d5_dataset_axis` / `ws3_dcnn_d6_onset_dataset_axis` / `dcnn_dataset_breadth_depth`
   / `deep_cnn_dataset_axis`); the new `kind:"synthesis"` row cites the 36 d5/d6/d8 run_ids.

---

## 5c. The §5b PENDING synchronized arms are now FINALIZED — the lenet5 cascaded→sync gap is computed at matched n=1000, completing the 4-dataset CNN cascaded table; SVHN cascaded FAILS the parity gate (`item_id`s `lenet_sync_n1000_complete_cnn_gap` + `lenet_cascade_kmnist_rung_svhn_parityfail`, 2026-06-25)

The §5b confound #1 ("matched-resolution cascaded→sync gap is NOT yet computable —
all n1000 synchronized counterparts remain PENDING") is **CLOSED**. The
`plnsync_lenet_*_synchronized_n1000` (MNIST/FMNIST) and
`plncpair_lenet_*_synchronized_n1000` (KMNIST/SVHN) arms have finalized `rc=0`,
so every n1000 cascaded cell now carries a **paired, same-resolution** synchronized
companion and a real `cascaded_to_sync_gap_pp` (the §5b records had `null`). Configs
are byte-identical except `ttfs_cycle_schedule` (cascaded vs synchronized);
`ttfs_cycle_based`, S=4, `simulation_steps=4`, `max_simulation_samples=1000`.

### The 4-dataset lenet5 cascaded→sync table (paired n=1000, 3 seeds)

| dataset | ANN ref | cascaded (3-seed ± sd_pp) | synchronized (3-seed ± sd_pp) | casc→sync GAP | casc→ANN | verdict |
|:--------|--------:|:--------------------------|:------------------------------|--------------:|---------:|:--------|
| MNIST | 0.9922 | **0.9873** (±0.25) | **0.9894** (±0.11) | **+0.21pp** | 0.48pp | near-lossless / MILD (gap < seed sd) |
| KMNIST | 0.9600 | **0.9310** (±0.10) | **0.9519** (±0.37) | **+2.09pp** | 2.90pp | mild firing-gain residual |
| FashionMNIST | 0.9176 | **0.8397** (±0.84) | **0.8911** (±0.48) | **+5.14pp** | 7.79pp | real MODERATE residual (>> seed sd; not mild) |
| SVHN | 0.8945 | **PARITY-GATE FAIL** (rc=1) | **0.8593** (±0.44) | **null** | — | cascaded sync-only/unavailable |

**Verdict — the cascaded CNN deficit orders monotonically by dataset margin**
(MNIST +0.21 < KMNIST +2.09 < FMNIST +5.14pp). MNIST is near-lossless (gap below the
seed sd), KMNIST is genuinely mild, but **FashionMNIST is a real MODERATE firing-gain
residual at +5.14pp — above the 1–2pp "mild" band** on a VALID on-chip-majority CNN
(99.1% on-chip), correcting the §5b "MILD and dataset-stable" framing for the hardest
greyscale dataset. The clean paired n1000 sync arm **tightens** the §5b context-only
mixed-resolution estimates: FMNIST drops from the 6.02pp (n1000-casc vs n50-sync)
context figure to a true **5.14pp**, because real n1000 synchronized FMNIST (0.8911)
sits BELOW the n50 sync baseline (0.8999). This is far milder than the `deep_mlp`
death-cascade (d8 cascaded gaps 10.8pp MNIST / 16.0pp FMNIST) — the convnet does not
collapse; it carries a bounded, dataset-margin-ordered residual.

**SVHN cascaded is a deployment-fidelity FAILURE, NOT a firing-gain result.** All 3
SVHN cascaded seeds (`plnmargin_lenet_SVHN_*_cascaded_n1000`) return `rc=1` and sit in
`q/failed/`: they crash in TTFS Cycle Fine-Tuning's `_run_nf_scm_parity_gate`
(`soft_core_mapping_step.py:312` → `nf_scm_parity.py:176` `NfScmParityError`) with
cascaded decision agreement **0.8906 / 0.7812 / 0.8750 < `min_agreement`=0.98**. The
post-crash deployed floats (~0.69/0.69/0.66) are gate-fail artifacts of a wrong-NF-
dynamics incident class, **not** a deployment metric, so the apparent ~17.9pp SVHN
casc→sync "gap" is spurious and EXCLUDED (`cascaded_to_sync_gap_pp=null`,
`cascaded_run_finalized=false`). The SVHN synchronized arm
(`plncpair_lenet_SVHN_*_synchronized_n1000`) is fully valid (rc=0, 0.8593 ± 0.44pp,
sync→ANN 3.52pp) and is the only valid SVHN deployment number. A parallel `plncpair`
cascaded SVHN arm ALSO returns rc=1, corroborating.

**Confounds / bounds.** (1) **No at-chance confound:** every ANN ref ≫ chance (MNIST
0.991–0.993, KMNIST 0.957–0.966, FMNIST 0.916–0.919, SVHN 0.893–0.897 vs 0.10), so
every reported drop is a genuine firing-gain measurement, not an untrained floor.
(2) `max_simulation_samples=1000` (not ≤50) → pp-gaps and 2–3 sig-fig deployed reads
are trustworthy; KMNIST cascaded NF↔SCM agreement = 1.0000 and torch↔sim = 1.0000 on
all 3 seeds. (3) **Depth-axis stress is modest** — lenet5 IR max-latency ~3, 2 neural
segments — so this is the breadth/dataset-margin axis, not the deep death-cascade axis
(that lives on `deep_cnn`, §4). (4) The MNIST cascaded `csr_lenet` rows are the §5b
arm re-paired here against the now-finalized `plnsync` sync companion; no cascaded
re-run. **This closes the §4d / §5b empty-synchronized-arm gap for the lenet5 cascaded
CNN table and flags SVHN cascaded as sync-only.** (Run ids: see ledger
`item_id:"lenet_sync_n1000_complete_cnn_gap"` and
`item_id:"lenet_cascade_kmnist_rung_svhn_parityfail"`.)

---

## 4z. The bigcores-gatefix `deep_cnn` cascaded→ANN deficit WIDENS d8→d10 against a full-eval trained-ANN reference, and `conversion_policy` is NET-NEGATIVE at BOTH rungs (`item_id=dcnn_deep_n1000_gatefix_d8_d10`, 2026-06-25)

The §4v MNIST depth ladder measured `cascaded→synchronized`; the WS7 §12 d8 escalation
measured `conversion_policy` against an *in-log* ANN (~0.9744). This row re-frames the
**n=1000 bigcores-gatefix `deep_cnn` MNIST cascaded** cell at d8 **and** the next rung d10
against the **full-eval trained ANN** (0.9949 / 0.9916) and pairs the `conversion_policy`
lever at both depths. `deep_cnn` (w16), `ttfs_cycle_based`, `ttfs_cycle_schedule=cascaded`,
S=4, `max_simulation_samples=1000`, on-chip-majority VALID. Ledger: `cluster:"WS3"`,
`kind:"depth"`, `item_id:"dcnn_deep_n1000_gatefix_d8_d10"` (2 rows).

| depth | cpFalse cascaded baseline (3-seed) | seeds | trained ANN | casc→ANN gap | cpTrue rescue | cp lift | rescue n |
|:-----:|-----------------------------------:|:------|------------:|-------------:|--------------:|--------:|:--------:|
| **d8**  | **0.9723** | .96/.981/.976 | 0.9949 | **2.26pp** | 0.9477 (.978/.954/.911) | **−2.46pp** | 3 |
| **d10** | **0.9433** | .892/.96/.978 | 0.9916 | **4.83pp** | 0.925 (.94/.91) | **−1.83pp** | 2 (s2 `rc=1`) |

**Verdict — `cascaded_firing_gain_degraded` (depth-widening 2.3→4.8pp), `conversion_policy`
rescue REFUTED (net-negative at both rungs, NOT a no-op).** Against the full-eval trained
ANN the cascaded baseline is degraded but **near-lossless, not collapse** — the conv
inductive bias caps severity at d8 (2.26pp) and d10 (4.83pp), and the deficit **widens with
depth** (consistent with the §4v sharp-onset → bounded-plateau ladder). The
`conversion_policy` escalation lever **HURTS at both rungs** (d8 −2.46pp, d10 −1.83pp) and
at d10 additionally **trips the NF↔SCM parity gate** (s2 `rc=1`), so it is a genuine
net-negative lever, not a benign no-op. This is the **depth-extension of the WS7 §12 d8-only
result** (cp net-negative) and confirms there is **no working `conversion_policy` firing-gain
rescue at depth on the convnet**; synchronized remains the unconditional deep_cnn default.

**Confounds / bounds.** (1) **No at-chance confound** — trained ANN ~0.99 at every cell
(d8 .9961/.993/.9955, d10 .9888/.9956/.9904, all ≫ 0.1135 chance) ⇒ genuine firing-gain
regime, not an untrained floor. (2) `max_simulation_samples=1000` → adequate resolution
(3rd decimals usable, but read the pp gaps). (3) **No synchronized arm in this batch** — the
pairing axis is `conversion_policy` (cpFalse baseline vs cpTrue rescue), so
`cascaded_to_sync_gap_pp=null`; the ~0.99 lossless reference is the **trained ANN** plus the
§4h/§4v synchronized deep_cnn ceiling (0.990–0.994). (4) **d10 cpTrue is n=2** — the third
seed `pdcnnbcn1000fix_d10_cotFalse_cpTrue_s2` finalized `rc=1` (`NfScmParityError`: NF↔SCM
cascaded agreement 0.9531 < 0.98 — a wrong-NF-dynamics incident **induced by the
conversion_policy lever**); excluded per `rc==0`, its 0.9054 `__target_metric` is a pre-crash
value not counted. (5) The companion `ttfs_theta_cotrain` lever is **not** analyzed — all
cotTrue runs crashed `rc=1` (`Conv2DPerceptronMapper features_3` tensor-shape break) and are
excluded (same break as WS7 §9–§12). (6) The d8 cpFalse/cpTrue run_ids are also cited in the
WS7 §12 `dcnn_deep_controller_escalation` row (in-log ANN 0.9744); this row uses the
full-eval ANN 0.9949 and the depth-pairing framing. Run ids: cpFalse
`pdcnnbcn1000fix_d{8,10}_cotFalse_cpFalse_s{0,1,2}`; cpTrue
`pdcnnbcn1000fix_d{8,10}_cotFalse_cpTrue_s{0,1,2}` (d10 s2 `rc=1`).

---

## 4aa. SYNTHESIS — the two CONFIRMED `deep_cnn` death-cascade items consolidated: a BOUNDED depth ladder on MNIST and a depth-first dataset-breadth re-opening off MNIST, both with the synchronized ceiling attached (`item_id`s `dcnn_deep_death_cascade_ladder` + `dcnn_dataset_breadth_cascaded`, 2026-06-25)

This round folds the scattered clean-`rc=0` `deep_cnn` cascaded evidence into **two
CONFIRMED consolidated items** carrying the synchronized arm at every cell, replacing the
retired INVALID host-majority `deep_mlp` §6 framing. Both ride the established VALID
on-chip-majority (99.6% on-chip) `deep_cnn` w16 bigcores vehicle, `ttfs_cycle_based`, S=4;
every ANN reference is ~0.92–0.99 (≫ 0.1135 chance), so these are genuine firing-gain
results, not untrained floors. Ledger: `cluster:"WS3"`, `kind:"depth"` (4 rows) +
`kind:"arch_dataset"` (4 rows).

**(A) `dcnn_deep_death_cascade_ladder` — MNIST depth ladder (n=1000 lower-noise family).**

| depth | cascaded (mean ±sd_pp) | synchronized (mean ±sd_pp) | ANN ref | cascaded→sync gap | n_seeds |
|:-----:|-----------------------:|---------------------------:|--------:|------------------:|:-------:|
| **d6**  | 0.9563 (±1.16) | 0.9924 (±0.15) | 0.9914 | **3.61pp** | 3 |
| **d8**  | 0.9450 (±2.09) | 0.9912 (±0.13) | 0.9921 | **4.62pp** | 6 |
| **d10** | 0.9328 (±3.76) | 0.9912 (±0.15) | 0.9928 | **5.84pp** | 6 |
| **d12** | 0.9353 (±3.70) | 0.9920 (±0.11) | 0.9923 | **5.67pp** | 3 |

(d5 baseline = lossless/tied, gap 0.07pp, from the §4v n=200 family.) Synchronized **holds
the ANN ceiling at every depth** (all sd ≤ 0.15pp). The cascade is **near-lossless through
d5, opens at d6, and widens to a BOUNDED ~4–6pp plateau** on this lower-noise n=1000 family —
**NOT a monotone collapse**. The n=200 family reads a sharper d12 (gap 9.48pp, casc 0.8967,
s0=0.835); read the ladder direction (widens with depth), not the d12 absolute. **Verdict —
`cascaded_firing_gain_degraded`; synchronized is the unconditional deep-model default.**

**(B) `dcnn_dataset_breadth_cascaded` — dataset spectrum off MNIST (n=200).**

| dataset | depth | cascaded (mean) | synchronized (mean) | ANN ref | cascaded→sync gap | sync→ANN | n_seeds |
|:--------|:-----:|----------------:|--------------------:|--------:|------------------:|---------:|:-------:|
| FashionMNIST | d5  | 0.8383 | 0.8986 | 0.9283 | **6.03pp** | +2.97pp | 3 |
| KMNIST       | d5  | 0.9167 | 0.9629 | 0.9696 | **4.62pp** | +0.67pp | 3 |
| FashionMNIST | d10 | 0.7250 | 0.9041 | 0.9336 | **17.91pp** | +2.95pp | 2 |
| KMNIST       | d10 | 0.8025 | 0.9623 | 0.9616 | **15.98pp** | +0.00pp | 2 |

**Verdict — `cascaded_death_cascade_reopens_offmnist`: the cascade does NOT stay closed off
MNIST.** It orders by **DEPTH first** (d10 gaps ~16–18pp ≫ d5 ~5–6pp), then by dataset, while
synchronized stays near-lossless (sync→ANN +0.0 to +3.0pp) everywhere. Notably the cascaded
deficit is **LARGER on FashionMNIST (ANN ~0.93) than on the nominally-harder KMNIST (ANN
~0.96–0.97)** at both depths (d5 FMNIST +6.03 > KMNIST +4.62; d10 FMNIST +17.91 > KMNIST
+15.98) — so on this CNN vehicle the cascade does **NOT** reproduce the `deep_mlp`
"harder-dataset = bigger-gap" ordering.

**Headline-gating ruling.** The §6 depth × firing-gain (death-cascade) risk is **REAL on the
VALID vehicle** but **BOUNDED** relative to the retired `deep_mlp`: delayed onset (~d6 vs MLP
d4) and a smaller MNIST deficit (~4–6pp at n=1000 vs MLP 4.3→9.3pp). The `deep_mlp`
catastrophic-collapse magnitude was inflated by its invalid host-majority + training-floor
confounds.

**Confounds / bounds.** (1) **Two resolution families — read gaps, not 3rd decimals.** The
n=200 ladder (0.005 grid) and n=1000 ladder (0.001 grid) both pair a **subsampled-cascaded**
arm against a **full-10000-sample synchronized SCM** (per commit 5568478/5568f); means are
unbiased so the multi-pp gaps are robust (>2–4× the per-seed band), but sub-0.2pp d5 gaps are
within noise (lossless). (2) **d12 severity is resolution/seed-sensitive** (n=200 9.48pp vs
n=1000 5.67pp) — read the ladder direction. (3) **Crash exclusions (strict `rc==0`).** Item A:
`pdcnnbc_d12_cascaded` s0/s2 (`rc=-9` OOM, `q/failed`), `dcnn_d6/d8_*` and `pdcnnladder_d6/d7_*`
(`rc=1` "No more hard cores available") excluded; the valid d6/d12 cascaded evidence is the
`pdcnnbcn1000*` family. Item B: FMNIST d10 cascaded s1 (`rc=-9`) and KMNIST d10 cascaded s0
(`rc=1`) excluded → those cells are n=2; including the consistent crashed artifacts (FMNIST
0.7416, KMNIST 0.8903) would not change the verdict direction. (4) **No at-chance confound**
(every ANN ref ≫ chance; on-chip parameter majority 99.6% confirmed, parity gates clean).
(5) Item B `pdcnndatafix_d6/d8` runs are **off-axis** (conversion-policy probe with no
synchronized counterpart) and are excluded from these cells. Run ids — Item A: cascaded
`pdcnnbcn1000plat_d{6,8,10}_cascaded_s{0,1,2}` + `pdcnnbcn1000seed_d{8,10,12}_cascaded_s{3,4,5}`,
synchronized the matching `*_synchronized_*`; Item B: `pdcnnd5data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}` (d5) +
`pdcnnd10data_{FashionMNIST,KMNIST}_DataProvider_{cascaded,synchronized}_s{0,1,2}` (d10).
**(REFUTED this round, not consolidated: `dcnn_deep_controller_cp_rescue_n1000` — the
`conversion_policy` rescue is net-negative at depth, see §4z.)**

## 4ab. The shallow d4 rung reads ZERO depth-risk on the VALID `deep_cnn` vehicle — cascaded HOLDS at the synchronized/ANN ceiling, replacing closeout v2 §10.1's INVALID `deep_mlp` shallow-rung death-cascade datapoint (`item_id=dcnn_d4_mnist_cascaded_vs_sync_ci`, 2026-06-25)

The §4v/§4aa MNIST ladder placed the death-cascade **onset** at ~d6. This rung pins the
**floor below onset**: `deep_cnn` **d4** w16, MNIST, `ttfs_cycle_based` S=4, the sole config
diff being `ttfs_cycle_schedule` cascaded vs synchronized. It directly **replaces** the
INVALID host-majority `deep_mlp` shallow-rung datapoint that closeout v2 §10.1 used to assert a
nonzero depth floor.

### The d4 MNIST rung — cascaded vs synchronized (deployed, ANN ref)

| schedule | deployed (mean ± sd_pp) | n_seeds | ANN ref | deployed→ANN | casc→sync gap |
|:---------|:------------------------|:-------:|--------:|-------------:|--------------:|
| cascaded | **0.9867** (±1.89) | 3 | 0.9922 | −0.55pp | — |
| synchronized | **0.9901** (±0.12) | 5 | 0.9924 | −0.23pp | **−0.34pp** |

**Verdict — `cascaded_near_lossless_on_cell`: NO death-cascade at d4.** The cascaded forward
holds **at/above** synchronized; the −0.34pp casc→sync "gap" is **within the 50-sample
resolution and reads as ZERO**, so the §6 depth-risk floor is **nonzero=FALSE at d4**. This is
the VALID-vehicle replacement for closeout v2 §10.1's INVALID `deep_mlp` shallow-rung
datapoint — the catastrophic shallow collapse there was a host-majority + training-floor
artifact, not a real depth floor.

### Confounds / bounds

(1) **Read gaps, not 3rd decimals (DOMINANT).** `max_simulation_samples=50` → deployed
granularity is **2.0pp/sample**. Cascaded values are exactly `{1.0, 0.96, 1.0}` =
`{50/50, 48/50, 50/50}`; the lone 0.96 is a **single** misclassified sample (2 errors), and the
1.89pp std is small-sample noise. The synchronized 4-decimal values come from the **same**
50-sample eval and are equally point-estimates. The −0.34pp casc→sync and −0.55pp cascaded→ANN
"gaps" are **not statistically resolvable** at n=50 and must be read as **zero**, not as
cascaded beating sync. (2) **Crash exclusions (strict `rc==0`).** Cascaded seeds **s1, s4
FAILED `rc=1`** (in `runs/campaign/q/failed/`) and are excluded → cascaded is n=3
(`cascaded_run_finalized=true`); synchronized has 5 finalized seeds. (3) **No at-chance
confound** — ANN ≈ 0.992 ≫ 0.10 chance for all 8 runs, so the cells are genuinely trained and
this is a legitimate firing-gain comparison. Run ids — cascaded
`f1_deep_cnn_mnist_ci_MNIST_DataProvider_cascaded_d4_s{0,2,3}`, synchronized
`...synchronized_d4_s{0,1,2,3,4}`, failed-excluded `...cascaded_d4_s{1,4}`.

## 4ac. Neither firing-gain rescue lever recovers the d6 cascaded deficit off MNIST — `theta_cotrain` is UNMEASURABLE (all `cotTrue` crash `rc=1`) and `conversion_policy=controller` REGRESSES the cascade; the best arm never reaches the synchronized ceiling, contradicting closeout §10.2's positive controller-rescue on the INVALID `deep_mlp` d8 (`item_id=dcnn_d6_theta_cotrain_cp_rescue_fmnist_kmnist`, 2026-06-25)

The §4aa dataset-breadth ladder showed the cascade re-opens off MNIST. This batch tests the two
candidate **firing-gain rescue levers** at the d6 onset rung on the VALID on-chip-majority
`deep_cnn` (w16, S=4, `ttfs_cycle_based` cascaded) vehicle, a 2×2 over
`ttfs_theta_cotrain` × `conversion_policy=controller`, on FashionMNIST and KMNIST.

### The d6 off-MNIST rescue 2×2 — deployed mean (3 seeds), with synchronized ceiling

| dataset | arm | deployed (mean ± sd_pp) | n_seeds (rc=0) | ANN | →sync ceiling |
|:--------|:----|:------------------------|:--------------:|----:|--------------:|
| FashionMNIST | cotFalse_cpFalse (**best**) | **0.8283** (±2.78) | 3 | 0.9312 | **+6.79pp below** |
| FashionMNIST | cotFalse_cpTrue | 0.8217 (±2.01) | 3 | 0.9307 | +7.45pp below |
| FashionMNIST | cotTrue_cp{False,True} | **CRASH rc=1** | 0/3 each | — | unmeasurable |
| FashionMNIST | synchronized (ceiling) | 0.8962 | 3 | — | — |
| KMNIST | cotFalse_cpFalse (**best**) | **0.9167** (±1.03) | 3 | 0.9654 | **+4.53pp below** |
| KMNIST | cotFalse_cpTrue | 0.8583 (±4.52) | 3 | 0.9711 | +10.36pp below |
| KMNIST | cotTrue_cp{False,True} | **CRASH rc=1** | 0/3 each | — | unmeasurable |
| KMNIST | synchronized (ceiling) | 0.9619 | 3 | — | — |

**Verdict — `neither_firing_gain_lever_rescues_cascade`.** (a) `theta_cotrain` is
**UNMEASURABLE**: all 12 `cotTrue` runs crash `rc=1`. (b) `conversion_policy=controller` does
**not rescue but REGRESSES** the cascade: cpFalse→cpTrue lift **−0.67pp** (FMNIST), **−5.83pp**
(KMNIST). (c) The best arm (`cotFalse_cpFalse`) stays **+6.79pp** (FMNIST) / **+4.53pp**
(KMNIST) **below** the synchronized ceiling and **never reaches it**. This **directly
contradicts** closeout §10.2's positive controller-auto-rescue lift, which was measured on the
**INVALID host-majority** `deep_mlp` d8 — the rescue does not transfer to the VALID vehicle.

### Confounds / bounds

(1) **`cotTrue` crash (DOMINANT).** All 12 `theta_cotrain=True` runs FAIL `rc=1` (in
`q/failed/`) with `RuntimeError '[ModelRepresentation] forward failed at node
Conv2DPerceptronMapper(name=features_3)'` (proximate torch error: `size of tensor a (28) must
match tensor b (16) at non-singleton dim 3`) at the **start of TTFS Cycle Fine-Tuning** — i.e.
inside the cotrain firing-gain forward. The high `__target_metric.json` floats on disk for
`cotTrue` (~0.928–0.932) are **STALE pre-deployment ANN/pretraining-stage artifacts** (carried
through; not deployed) and must NOT be read as cotrain deployed accuracy. (2) **Controller
high-variance, not a crash.** KMNIST `cotFalse_cpTrue` s0 = 0.80 is a **GENUINE `rc=0`
finalized collapse** (99.41% on-chip, agreement 1.0, parity 1.0) — the controller policy is
legitimately high-variance (sd 4.52pp), not a harvesting artifact. (3) **No at-chance
confound** — ANN ≈ 0.931 (FMNIST) / 0.965 (KMNIST) ≫ chance. (4) **Sync-ceiling choice.**
Primary synchronized ceiling = `pdcnnbcd6data_*_synchronized_s{0,1,2}` (matches the FMNIST batch
vehicle exactly); an alt KMNIST sync batch `pdcnnd6databc` reads 0.9694 (gap +5.27pp) — verdict
direction unchanged. Run ids — `pdcnnd6datacotfix_{FashionMNIST,KMNIST}_DataProvider_cot{False,True}_cp{False,True}_s{0,1,2}`
(cotTrue all `rc=1`-excluded), synchronized
`pdcnnbcd6data_{FashionMNIST,KMNIST}_DataProvider_synchronized_s{0,1,2}`.


## 4ad. At the SHALLOWEST off-MNIST cascaded rung (`deep_cnn` d5, S=4), the `ttfs_staircase_ste` gradient only PARTIALLY and unevenly closes the dataset-margin firing-gain gap — a clean +1.4pp lift on KMNIST, a wash (+0.67pp, sign-flips by seed) on FashionMNIST; neither reaches the prior-item synchronized ceiling (`item_id=dcnn_d5_ste_onset`, 2026-06-25)

The §4t d6 STE decomposition showed `ttfs_staircase_ste` is the dominant gate-fix knob but
not lossless. This batch (`pdcnnd5stefix_*`) probes the **same STE lever one rung shallower**,
at the **d5 cascaded onset off MNIST**, on the VALID on-chip-majority `deep_cnn` vehicle.

**The in-batch lever is `ttfs_staircase_ste` (steTrue vs steFalse), BOTH at
`ttfs_cycle_schedule=cascaded`** — this is NOT a cascaded-vs-synchronized contrast. The
synchronized ceilings and canonical cascaded baselines below come from the PRIOR
`pdcnnd5data_` item (`item_id=dcnn_d5_dataset_axis`), NOT from any run in this batch.

### The d5 STE-on/off contrast — deployed mean, with prior-item sync ceiling

| dataset | arm | deployed mean | per-seed | n (rc=0) | STE Δ (pp) | →prior sync ceiling |
|:--------|:----|:-------------:|:---------|:--------:|:----------:|--------------------:|
| FashionMNIST | steFalse (cascaded) | 0.8067 | 0.82 / 0.775 / 0.825 | 3 | — | +9.19pp below 0.8986 |
| FashionMNIST | steTrue | 0.8133 | 0.875 / 0.80 / 0.765 | 3 | **+0.67** (sign-flips +5.5/+2.5/−6.0) | +8.53pp below 0.8986 |
| KMNIST | steFalse (cascaded) | 0.8775 | 0.88 / 0.875 | **2** | — | +8.54pp below 0.9629 |
| KMNIST | steTrue | 0.8917 | 0.935 / 0.85 / 0.89 | 3 | **+1.42** (+5.5/+1.5 shared seeds) | +7.12pp below 0.9629 |

**Verdict — PARTIAL / MIXED.** STE gives a **clean partial lift on KMNIST** (+1.42pp mean,
best steTrue 0.935 toward the 0.9629 sync ceiling) and is a **wash on FashionMNIST** (+0.67pp
mean, within 200-sample noise, per-seed sign-flips +5.5/+2.5/−6.0pp). **Neither dataset reaches
the prior-item synchronized ceiling** (residual +7.12pp KMNIST / +8.53pp FashionMNIST). The
staircase-STE lever **narrows but does not close** the d5 dataset-margin gap — consistent with
the §4t d6 finding (dominant but not lossless), now confirmed dataset-dependent one rung
shallower. Firing-gain origin confirmed (ANN ≫ chance).

### Confounds / bounds

(1) **Lever is STE-on/off, NOT cascaded-vs-sync.** Both arms run `ttfs_cycle_schedule=cascaded`;
in the ledger the schema's `cascaded_deployed_mean`←steFalse and `synchronized_deployed_mean`←steTrue,
so `cascaded_to_sync_gap_pp` is repurposed as the STE delta (steTrue−steFalse), NOT a sync gap.
(2) **`max_simulation_samples=200`** → ~0.005 grid; read gaps in pp, not 3rd decimals. The
200-sample steFalse means (FMNIST 0.807, KMNIST 0.878) do not exactly reproduce the prior
cascaded baselines (0.8383, 0.9167). (3) **KMNIST steFalse is n=2:** seed s1
(`pdcnnd5stefix_KMNIST_DataProvider_steFalse_s1`) is in `q/failed/` — its log stops at *TTFS
Cycle Fine-Tuning* BEFORE deployment, and its on-disk `__target_metric.json` 0.9559 is a
**STALE pre-deployment ANN/torch-mapping artifact** (the `[PROFILE]` trace shows 0.9559 = ANN
test acc, Δ=0), NOT a deployed metric → excluded. The other 11 runs are `rc=0`. (4) **No
at-chance confound** — ANN ≈ 0.927 (FMNIST) / 0.969 (KMNIST) ≫ chance → genuine firing-gain
gap. (5) **FashionMNIST per-seed STE delta is unstable** (+5.5/+2.5/−6.0pp) → its +0.67pp mean
is not significant at n=3. Run ids —
`pdcnnd5stefix_{FashionMNIST,KMNIST}_DataProvider_ste{False,True}_s{0,1,2}`
(KMNIST steFalse s1 `rc=-9`-excluded).

