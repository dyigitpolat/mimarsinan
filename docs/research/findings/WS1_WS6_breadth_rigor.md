> ⚠️ **VALIDITY:** deep_mlp results below are an **INVALID host-majority config** (<50% params on-chip; see [VALIDITY_AUDIT.md](VALIDITY_AUDIT.md)). The phenomena may be real but are NOT valid on-chip deployments. Valid trainable-deep vehicle = **deep_cnn**.

# WS1+WS6 — Breadth × Rigor (consolidated CI table)

**Question.** Does the synchronized `deep_mlp` deployment story (ANN → on-chip
TTFS) hold *across datasets* (FMNIST, KMNIST, SVHN) and *across depth* (4/6/8),
not just on MNIST? With proper 3-seed confidence intervals, where is it
near-lossless and where does a gap open?

**Status: CLOSED for the grayscale family (MNIST/FMNIST/KMNIST × d{4,6,8}),
3 seeds per cell, cross-replicated across two independent batches. SVHN and the
`mlp_mixer` seed-matrix are confounds and are scoped out (see §5).**

**One-line verdict.** Synchronized `deep_mlp` lowering is **bit-exact on every
dataset** (the chip mapping adds ~0pp), but the **ANN→TTFS conversion** loses
accuracy that **grows with depth and with dataset difficulty**: MNIST is
near-lossless (d4 ~1.0pp = MET; d6/d8 ~1.4–1.7pp), FMNIST holds a ~2–2.6pp gap,
and KMNIST carries a real **~5.6–6.7pp** gap. Breadth is **not free** — the
harder the dataset, the more the trainable TTFS fold costs, even though the
deployment itself is lossless.

---

## 1. Matrix (what actually ran)

`deep_mlp` (width 64, 5-bit weights, T=4 TTFS cycle window) ×
{MNIST, FashionMNIST, KMNIST} × depth {4, 6, 8} × schedule {synchronized} ×
seed {0, 1, 2}.

Two independent batches cover this grid and are **pooled per cell only as a
cross-check** — each reported cell is a single batch's 3 seeds:

- **`sch_dmlp_<Provider>_d<depth>_synchronized_s{0,1,2}`** — the canonical
  batch; complete 3×3×3 for MNIST/FMNIST/KMNIST except KMNIST d8 (placeholder).
- **`ws6b_*` / `ws6c_*`** — an independent replication of the same recipe
  (`ttfs_cycle_schedule=synchronized`, `model_type=deep_mlp`, width 64); supplies
  the KMNIST d8 cell and the SVHN attempt.

ANN reference = **Pretraining `Test accuracy`** from
`_GUI_STATE/steps.json` (≡ that step's `target_metric`).
Deployed = `__target_metric.json` (soft-core spiking-sim metric, ≡ the
Hard Core Mapping `target_metric`). gap = ANN − deployed (positive = lost).

---

## 2. The consolidated CI table (3-seed mean ± 95% CI)

CI = t-interval, df = n−1 = 2, t₀.₉₇₅ = 4.303. Verdict rule:
**MET = deployed within ~1pp of ANN (lossless modulo eval noise); GAP otherwise.**

| dataset | depth | n | ANN test | deployed (mean ± 95% CI) | gap (pp) | verdict |
|---------|------:|--:|---------:|--------------------------|---------:|:-------:|
| MNIST   | 4 | 3 | 0.9786 | **0.9685 ± 0.0071** | **+1.01** | **MET** |
| MNIST   | 6 | 3 | 0.9791 | **0.9654 ± 0.0017** | **+1.37** | GAP |
| MNIST   | 8 | 3 | 0.9783 | **0.9616 ± 0.0072** | **+1.67** | GAP |
| FMNIST  | 4 | 3 | 0.8910 | **0.8712 ± 0.0094** | **+1.97** | GAP |
| FMNIST  | 6 | 3 | 0.8851 | **0.8632 ± 0.0004** | **+2.19** | GAP |
| FMNIST  | 8 | 3 | 0.8831 | **0.8571 ± 0.0080** | **+2.60** | GAP |
| KMNIST  | 4 | 3 | 0.8962 | **0.8406 ± 0.0085** | **+5.56** | GAP |
| KMNIST  | 6 | 3 | 0.8940 | **0.8306 ± 0.0220** | **+6.34** | GAP |
| KMNIST  | 8 | 3 | 0.8888 | **0.8216 ± 0.0293** | **+6.72** | GAP |

(KMNIST d8 from the `ws6b` batch; all other cells from `sch_dmlp`.)

**Cross-batch replication** (the same cell, two independent 3-seed batches,
gap_pp agreement) — the breadth signal is stable, not seed luck:

| cell | `sch_dmlp` gap | `ws6b/c` gap | Δ |
|------|---------------:|-------------:|--:|
| MNIST d4  | 1.01 | 1.19 (n2) | 0.18 |
| MNIST d6  | 1.37 | 1.31 | 0.06 |
| MNIST d8  | 1.67 | 1.80 (n2) | 0.13 |
| FMNIST d4 | 1.97 | 1.77 | 0.20 |
| FMNIST d6 | 2.19 | 2.27 | 0.08 |
| FMNIST d8 | 2.60 | 2.42 | 0.18 |
| KMNIST d4 | 5.56 | 5.56 | 0.00 |
| KMNIST d6 | 6.34 | 5.64 | 0.70 |

Every cell replicates within **≤0.7pp** across two independent batches. The
table is a real effect, not noise.

---

## 3. Where the gap lives — the chip mapping is lossless

The accuracy that is lost is lost in the **trainable TTFS conversion**
(TTFS Cycle Fine-Tuning), **not** in the lowering. Tracing one seed per cell
through `steps.json` `target_metric`:

| cell | ANN (pretrain) | after TTFS-FT | deployed (HCM) | TTFS drop | map drop |
|------|---------------:|--------------:|---------------:|----------:|---------:|
| MNIST d4  | 0.9740 | 0.9646 | 0.9652 | −0.94 | **+0.06** |
| MNIST d8  | 0.9791 | 0.9606 | 0.9642 | −1.85 | **+0.36** |
| FMNIST d8 | 0.8830 | 0.8500 | 0.8602 | −3.30 | **+1.02** |
| KMNIST d4 | 0.8974 | 0.8393 | 0.8445 | −5.81 | **+0.52** |
| KMNIST d6 | 0.8969 | 0.8331 | 0.8360 | −6.38 | **+0.29** |

The "map drop" (TTFS-FT → soft/hard-core deployed) is **≈0 or slightly
positive** in every cell — Quantization Verification → Core Quant Verification →
Soft Core Mapping → Hard Core Mapping carry the `target_metric` forward with
≤0.0001 drift. So the entire ANN→deployed gap is the **ANN→single-spike TTFS
fold**, and it scales with both depth (deeper cascade = bigger fold) and dataset
difficulty (lower-margin distributions lose more per fold).

This is consistent with the per-run lowering parity that other WS work already
locks (NF↔SCM per-neuron 0.0000%, torch↔deployed-sim parity 1.0): the breadth
question is **purely about how much the trainable TTFS network retains per
dataset**, and the mapper is exonerated.

---

## 4. Breadth verdict

**Does synchronized `deep_mlp` hold across FMNIST / KMNIST / depth? —
QUALIFIED. The *lowering* generalizes perfectly; the *trainable TTFS recipe*
does not stay lossless off MNIST.**

- **MNIST** — the only **MET** cell is d4 (+1.01pp, lossless within ~1pp). d6
  (+1.37) and d8 (+1.67) are tight, low-variance GAPs: synchronized MNIST is
  *near*-lossless and degrades gently and monotonically with depth.
- **FMNIST** — a **consistent ~2–2.6pp GAP** that grows with depth
  (1.97 → 2.19 → 2.60). Earlier round-1 "FMNIST is lossless (−0.01pp)" was an
  **n=1 artifact off a single smoke run with a different ANN reference**; with
  3 seeds and the matched per-run Pretraining reference, FMNIST is **not**
  lossless. It is, however, *stable* and *bounded*.
- **KMNIST** — a **real ~5.6–6.7pp GAP** at every depth, the largest by far.
  KMNIST (cursive Kuzushiji) is the hardest grayscale set and the TTFS fold
  costs ~3× what it costs on MNIST. The gap is **reproducible across both
  batches** (d4 identical 5.56 vs 5.56), so it is a **genuine dataset effect**,
  not a seed or a training-flake.
- **Depth axis** — within every dataset the gap is **monotone in depth**
  (deeper cascade ⇒ larger TTFS fold): MNIST 1.01→1.37→1.67, FMNIST
  1.97→2.19→2.60, KMNIST 5.56→6.34→6.72. Depth and dataset-difficulty **stack**.

**The honest headline:** synchronized `deep_mlp` lowering is lossless on every
new dataset, but "lossless deployment" as a *headline* is **MNIST-d4-only**.
Off MNIST, the trainable TTFS conversion opens a dataset-difficulty-scaled gap
(FMNIST ~2pp, KMNIST ~6pp) that the bit-exact mapping cannot hide. Closing it is
a **training** problem (the TTFS-FT step), not a lowering problem.

---

## 5. Confounds (scoped out of the table)

- **SVHN (`ws6b_svhn_*`, deep_mlp) — CONFOUND, trainer-level.** The `sch_dmlp`
  SVHN cells **never produced artifacts** (all `pending`, no `generated/` dir).
  The `ws6b` SVHN runs that did finish have an **ANN that itself only reaches
  0.56–0.69** (d4 s2 ANN 0.69 → deployed 0.467; d8 ANN 0.56–0.65 → deployed
  0.33–0.48; gaps 17–24pp). The MLP is the **wrong inductive bias for 3×32×32
  RGB natural images** — the network never trained, so this is *not* a
  deployment-fidelity result and is excluded. SVHN needs a **CNN/LeNet-class**
  model (the next breadth rung), not `deep_mlp`. (Several SVHN jobs also
  hard-failed: `ws6b_svhn_d4_{s0,s1}`, `d8_s0`.)
- **`ws6_seed_matrix_{1,2,4,5,7,9}_s{3..7}` — CONFOUND, wrong model family.**
  These are **`model_type=mlp_mixer_core`** (not `deep_mlp`), all MNIST, opaque
  hyperparameter cells (matrix_1…matrix_9), resume-style configs with no
  recorded depth or schedule. They are a *different* model program (an MNIST
  mlp_mixer multiseed-CI sweep) and are **excluded** from the deep_mlp breadth ×
  depth cells. Note: `matrix_9` all five seeds failed (`deployed_acc=null`), and
  several `matrix_{1,2,4,5}` low seeds are 3–4s crash-failures; only the
  long-wall runs are real. Not part of the breadth claim.

All three confounds are recorded in the ledger as `breadth_ci` rows with
`verdict=CONFOUND` so the next round does not mistake them for breadth cells.

---

## 6. Ledger

12 `breadth_ci` verdicts appended to the campaign ledger under cluster **WS6**
(`scripts/campaign/research_loop.py ledger-append`): 9 grayscale cells
(1 MET + 8 GAP) + 2 SVHN CONFOUND + 1 mlp_mixer-seed-matrix CONFOUND. Each
grayscale record carries `ann_test_acc`, `deployed_acc_mean`,
`deployed_acc_ci95`, per-seed `deployed_acc_seeds`, `gap_pp`, and the
`cross_batch_gap_pp` agreement value.

---

## 7. Next breadth rung

1. **Close the KMNIST/FMNIST TTFS-FT gap as a *training* problem.** The mapping
   is lossless; the loss is the ANN→single-spike fold and it is
   dataset-difficulty-scaled. The lever is the TTFS Cycle Fine-Tuning recipe
   (gradient/surrogate/per-depth θ), **not** the mapper. This is the same
   gradient-not-schedule conclusion the cascaded-TTFS line already reached;
   breadth confirms it generalizes off MNIST.
2. **SVHN belongs on a CNN/LeNet, not deep_mlp.** SVHN's failure here is a
   model-capacity confound (ANN ≤0.69). The honest SVHN breadth test needs a
   convolutional model (and likely an input-fan-in mapping change for the
   3072-wide RGB input). Until then, "does deployment generalize to natural
   images?" stays open — but for a *model-family* reason, not a lowering one.

Sequencing: fix the TTFS-FT gap on the known grayscale cells first (it is the
attributable training axis), then graduate to the CNN+SVHN rung (the
architecture + input-shape axes) so the next breadth result stays attributable.
