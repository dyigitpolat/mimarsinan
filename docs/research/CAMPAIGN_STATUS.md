# CAMPAIGN_STATUS — living synthesis & coordinator hand-off

**Owner:** Research Director (coordinator). **Last synthesized:** 2026-06-24
(HEAD `18eee5d`, branch `main`; runner alive at re-synthesis: 252 done / 51
failed / 8 running / 23 pending). **Inputs:** the ledger
(`scripts/campaign/research_loop.py ledger-read`, 67 records), the three cluster
findings under `docs/research/findings/`, the live queue (`runs/campaign/q/`), and
the freshly-drained `generated/` artifacts harvested at re-synthesis time
(`ws7esc_*`, `sch_lenet_SVHN_*`, `pm_sync_*`, `ws6b_*`, `cp_d8h_SVHN_*`).

This is the standing synthesis that replaces re-reading every artifact. It states
what is **proven**, what is **in-flight**, what is **open**, and what must be
**built** next — rigorously and honestly. Where a number is single-seed,
single-run, or harvested from a `generated/` artifact **not yet consolidated into a
ledger verdict record**, it is labelled `[harvested]`. The charter is
`docs/research/RESEARCH_PROGRAM.md`; the program north-star is
`docs/mimarsinan_closeout_analysis_v2.md`; the GPU substrate is
`docs/PUBLICATION_CAMPAIGN.md`.

> **What changed since the last synthesis (the headlines).** Four large sweeps
> drained and reframe the campaign:
> 1. **SVHN LeNet landed and KILLS the "CNN makes cascaded free" claim.** On RGB
>    SVHN the cascaded↔synchronized gap is **19.1 pp** (cascaded 0.670 vs sync
>    0.860), not ~0.9 pp. The CNN advantage was an easy-grayscale artifact.
> 2. **The keystone ESCALATION path NEVER fires** — across **18** `conversion_policy`
>    runs (incl. the trainable off-distribution `deep_mlp d8 cascaded` re-test
>    specified in WS7 §7) every run took the **fast ladder**; `cpTrue` and `cpFalse`
>    traces are structurally identical. Escalation is effectively dead code on real
>    cells.
> 3. **The WS3 "d8 cascaded collapse to 0.872" is recipe-dependent, not intrinsic.**
>    The *same* `deep_mlp d8 cascaded` architecture under the conversion-policy
>    recipe deploys **0.963** (3 seeds). The collapse is a property of the WS3
>    training recipe, not of cascaded-at-d8.
> 4. **WS6 has real multi-seed breadth CIs now** (clean `ws6b_*` re-queue,
>    synchronized): MNIST/FMNIST hold to ~1.2–2.4 pp, **KMNIST is a genuine ~6 pp
>    breadth limit** (multi-seed confirmed), SVHN `deep_mlp` is hard (14–25 pp, and
>    the ANN itself barely trains).

---

## 0. The one-paragraph state of the campaign

The **synchronized** TTFS-cycle schedule remains the established, near-lossless,
depth-stable, multi-seed deployment default on every trainable cell we have
(MNIST/FMNIST `deep_mlp` d≤8 within ~2.4 pp; LeNet CNNs; mixer). The **cascaded**
single-spike schedule carries a real firing-gain deficit, but the new wave sharply
**narrows its scope**: (a) the deficit is **dataset/architecture-dependent and
re-opens badly on RGB** — SVHN LeNet cascaded loses **19 pp** to sync, the opposite
of the earlier "CNN makes cascaded free" read; and (b) the WS3 d8 cascaded
**collapse to 0.872 is a recipe artifact** — under the conversion-policy recipe the
same model deploys 0.963. The **deep-MLP depth law is still bounded to d≤8** by the
ANN-untrained plain-Linear backbone (a backbone limit), gating the deep firing-gain
question on a residual/norm backbone (**WS2/`deep_cnn`**, unbuilt). The **keystone
MATCHes** but its **escalation path has now been shown not to fire on any of 18 real
runs** — it is a no-op selector that keeps the fast path regardless; conversion-policy
buys only a small recipe-default lift, not adaptive escalation. Three capabilities
still block the next phase: a **trainable deep CNN backbone** (`deep_cnn`),
**baseline conversion methods** (for any publishable claim), and the **per-layer-S
Pareto allocator** (WS5).

---

## 1. Findings so far (the proven core, with the new-wave updates folded in)

### WS3 — Depth × firing-gain (HEADLINE; recipe-confounded at the headline rung)
Findings doc: `docs/research/findings/WS3_depth_firing_gain.md`. Ledger:
`WS3 kind=depth_firing_gain_final` (24 consolidated cells).

- **Synchronized holds near the ANN ceiling, depth-stable, tight.** MNIST sync
  deploys 0.9699 (d4) → 0.9640 (d6) → 0.9644 (d8), essentially flat, ≤1.5 pp from
  the float ANN at every depth, seed sd ≤ 0.24 pp (3 seeds). Verdict
  `synchronized_holds_near_ann` on all 5 trainable cells (MNIST d4/d6/d8 + FMNIST
  d4/d8).
- **Cascaded degrades with depth and is strictly dominated — under the WS3 recipe.**
  MNIST cascaded 0.9267 (d4) → 0.94 (d6) → **0.8717 (d8)**; ANN-gap 5.5→10.8 pp.
  FMNIST cascaded falls harder (0.845 d4 → **0.725 d8**). Verdict
  `cascaded_firing_gain_degraded` on all 5 trainable cells. The d6 midpoint
  **refutes a clean monotone law** (0.94 > 0.9267, then d8 collapses); the
  defensible claim is *cascaded is strictly dominated by synchronized on every
  trainable rung, with a sharp seed-stable collapse appearing by d8 under this
  recipe.*
- **NEW — that d8 collapse is recipe-dependent, not intrinsic.** `[harvested]` The
  keystone re-test (`ws7esc_deepmlp_d8_cascaded`, same `deep_mlp d8 w64 cascaded
  S=4`, but with `conversion_policy:True` and the fast-ladder fine-tune recipe)
  deploys **0.955 / 0.965 / 0.97** (3 seeds, ANN ref 0.978). The fast-ladder
  fine-tune holds post-acc ≈ 0.97 through rate=1.0. So "cascaded collapses at d8" is
  **a property of the WS3 cluster's training recipe, not of cascaded-single-spike at
  d8**. This must be reconciled before the WS3 headline ships: *which* recipe knob
  (LR-find vs fixed spanning-cosine, ramp schedule, θ defaults) produces the 9 pp
  swing is now the open mechanism question.
- **The scope confound stands:** plain `Linear+ReLU` `deep_mlp` is ANN-untrained at
  d≥12 (chance for both schedules; the d16/d24 @w128 rescue failed). 14 cells are
  `training_floor_confound` and carry zero firing-gain information. The deep (d≥12)
  firing-gain test is **GATED on a trainable deep backbone (WS2)**.

### WS3 extension — architecture-dependence (REVISED: convolution does NOT save cascaded)
Ledger: `WS3 kind=cnn_mode_compare` (consolidated MNIST). LeNet schedule sweep:
15 grayscale `sch_lenet_*` runs + **6 new SVHN `sch_lenet_SVHN_*` runs** — both
`[harvested]`, not yet consolidated into per-cell ledger verdicts.

The prior synthesis read "the death-cascade is architecture-specific; on a LeNet CNN
the cascaded penalty is nearly free." **The SVHN rung overturns the universal form
of that claim.** Consolidated LeNet picture across all four datasets:

| dataset | cascaded (mean ±sd) | synchronized (mean ±sd) | ANN ref | casc↔sync gap |
|:--------|:--------------------|:------------------------|--------:|--------------:|
| MNIST   | 0.980 (n3)          | 0.989 (n3)              | ~0.991  | **0.9 pp**    |
| FMNIST  | 0.840 (n3)          | 0.900 (n3)              | ~0.919  | **6.0 pp**    |
| KMNIST  | 0.900 (n2)          | 0.948 (n1)              | ~0.96   | ~4.8 pp       |
| **SVHN**| **0.670 ±1.1 (n3)** | **0.860 ±0.5 (n3)**     | ~0.895  | **19.1 pp**   |

> **The corrected claim.** The cascaded↔synchronized gap on a LeNet CNN is **not a
> fixed architectural property** — it tracks task difficulty/colour: ~0.9 pp on
> easy grayscale MNIST, ~5–6 pp on FMNIST/KMNIST, and a **19 pp blowout on RGB
> SVHN**. "Convolution makes cascaded nearly free" was an MNIST artifact.
> Synchronized, by contrast, is near-lossless on **every** LeNet rung including SVHN
> (gap to ANN: MNIST −0.0, FMNIST +1.8, KMNIST +0.6, SVHN +3.4 pp). The robust,
> defensible headline is now: **synchronized is the universal near-lossless default;
> cascaded's penalty grows with task difficulty regardless of architecture.**

### WS1+WS6 — Breadth × rigor (now has real multi-seed CIs)
Findings doc: `docs/research/findings/WS1_WS6_breadth_rigor.md`. Ledger: `WS6`
(18 records, all from the **broken** round-1). The clean re-queue is the new
`ws6b_*` family (synchronized, ttfs_cycle), **`[harvested]`, not yet consolidated**.

- **The lowering is bit-faithful on every new provider** (round-1 fact that
  survives): FMNIST d4 NF↔SCM per-neuron parity 0.0000%, torch↔deployed-sim parity
  1.0; MNIST/KMNIST d4 parity 1.0. The chip mapping adds nothing — breadth is purely
  how much the trained TTFS network retains per dataset.
- **NEW — first multi-seed breadth CIs (synchronized `ws6b_*` re-queue, clean tree):**

  | dataset | depth | deployed (mean ±sd) | ANN ref | gap | n |
  |:--------|------:|:--------------------|--------:|----:|--:|
  | MNIST   | 4 | 0.9692 ±0.14 pp | 0.9811 | 1.19 pp | 2 |
  | MNIST   | 8 | 0.9593 ±0.05 pp | 0.9773 | 1.80 pp | 2 |
  | FMNIST  | 4 | 0.8709 ±0.17 pp | 0.8887 | 1.77 pp | 3 |
  | FMNIST  | 8 | 0.8588 ±0.36 pp | 0.8830 | 2.42 pp | 3 |
  | KMNIST  | 4 | 0.8420 ±0.60 pp | 0.8976 | **5.56 pp** | 3 |
  | KMNIST  | 8 | 0.8216 ±0.96 pp | 0.8888 | **6.72 pp** | 3 |
  | SVHN    | 4 | 0.467 (n1 valid) | 0.690 | ~22 pp | 1 |
  | SVHN    | 8 | 0.40 (n2: 0.328/0.475) | ~0.61 | ~21 pp | 2 |

- **Reads:** synchronized breadth **holds** on MNIST/FMNIST (≤2.4 pp at d8,
  tight). **KMNIST is a genuine breadth limit** — the n=1 6.16 pp flag from round-1
  is now **multi-seed confirmed (~6–7 pp)**, and it *worsens with depth*. **SVHN
  `deep_mlp` is a non-result**: the *ANN itself* only trains to ~0.61–0.69 (a
  backbone/task limit), so the deployed 0.33–0.48 is a training-floor-adjacent
  confound, not a clean firing-gain measurement — SVHN needs the CNN backbone, not
  a plain MLP.
- **Round-1 provenance trap (resolved by re-queue).** 13/18 round-1 WS6 cells were
  `BROKEN` by an unresolved **git merge-conflict marker** in
  `src/mimarsinan/models/builders/__init__.py`; the runner's `artifact_ok` checked
  **existence not freshness**. The `ws6b_*` re-queue ran against a clean tree and
  produced the CIs above — but **C6 (the freshness/conflict-marker gate) is still
  unbuilt**, so the trap can recur.

### WS7 — The keystone: automatic recipe selection (MATCH proven; ESCALATE shown DEAD)
Findings doc: `docs/research/findings/WS7_keystone_automatic.md`. Ledger:
`WS7 kind=keystone_decision` (3 cells) + the new `ws7esc_*` escalation sweep
(18 runs, `[harvested]`, not yet consolidated).

- **MATCH works end-to-end** (unchanged): in-distribution `mlp_mixer_core` (0.96)
  and near-distribution no-bias mixer (0.9606) both correctly MATCH→keep-fast. The
  dtype-leak crash is fixed in HEAD.
- **NEW — the ESCALATE path does not fire on ANY real run.** The WS7 §7 re-test was
  run (`ws7esc_deepmlp_d8_cascaded`, trainable off-distribution cascaded d8) plus a
  full off-distribution A/B (`ws7esc_nb_*` = `deep_mlp d8 w64 cascaded S=4` on a
  *constrained core board* forcing neuron-split/coalescing, MNIST/FMNIST/mixer ×
  {cpTrue,cpFalse} × 3 seeds). **All 18 runs took the 8-rung fast spanning-cosine
  ladder** (≤2 distinct LRs, no multi-probe controller search, no rollback-driven
  adaptive loop). The escalation path is effectively **dead code on real cells**.
- **Why it doesn't matter for accuracy (but does for the claim):** the
  off-distribution cells **did not death-cascade** — they deploy 0.94–0.97 (MNIST),
  0.81 (FMNIST), 0.95 (mixer). A healthy cascade has `cold_cascade_live=True` and
  `firing_gain≈1`, so the death-cascade probes correctly do not trip. **We still
  have no real run where a true firing-gain deficit forces escalation** — the
  death-cascade probe remains validated only by an artificially-pruned (~92%
  sparsity) unit test.
- **The `conversion_policy` A/B isolates its actual effect.** `[harvested]`
  cpTrue vs cpFalse, same cells, same fast path:

  | cell | cpTrue (mean) | cpFalse (mean) | Δ |
  |:-----|--------------:|---------------:|--:|
  | MNIST nb d8 casc | 0.948 | 0.945 | +0.3 pp |
  | FMNIST nb d8 casc | 0.812 | 0.794 | +1.8 pp |

  So `conversion_policy:true` buys a **small, consistent recipe-default lift
  (≈+0.3 to +1.8 pp)** — *not* adaptive escalation. This is the honest
  characterization: the keystone is currently a **recipe-default applier**, not a
  selector.
- **SVHN deep_mlp keystone A/B** (`cp_d8h_SVHN_*`, `[harvested]`): cpTrue 0.46 vs
  cpFalse 0.39 (mean) — same small lift, but on a model whose ANN barely trains
  (confounded, as above).
- **Thresholds remain MNIST-mmixcore-calibrated absolute constants**
  (`_FIRING_GAIN_FLOOR=0.1`, `_LIVE_DEPTH_FRACTION=0.75`, `_RAMP_BUDGET=0.02`,
  `_LIVENESS_FLOOR=1e-4`). To *demonstrate* escalation we now need a cell that
  **actually trips the probes** — i.e. a model with a real death-cascade (the WS3
  d8 collapse *under the WS3 recipe*, or a deeper backbone once built) routed through
  `conversion_policy`. The escalation re-test is now **better specified**: re-run
  the WS3-recipe (not fast-ladder) d8 cascaded cell *with* `conversion_policy:true`
  and `escalation_reason`/`probes` instrumentation (C7), so the characterizer sees
  the collapsing cascade WS3 actually produced.

---

## 2. The verdict landscape (per model × dataset × mode)

Consolidated verdicts + new-wave harvested rows. **TRAINABLE** = ANN ≫ chance (the
only firing-gain evidence). Deployed = soft-core spiking-sim metric; gap = ANN −
deployed. `[h]` = harvested, not yet a consolidated ledger verdict.

### `deep_mlp` (plain Linear+ReLU, w64) — TRAINABLE band
| dataset | d | cascaded (WS3 recipe) | cascaded (conv-policy recipe) | sync | verdict |
|:--------|--:|---------------------:|------------------------------:|-----:|:--------|
| MNIST  | 4 | 0.9267 ±2.66 (n3) | — | 0.9699 ±0.09 (n3) | casc degraded / sync holds |
| MNIST  | 6 | 0.9400 ±1.08 (n3) | — | 0.9640 ±0.14 (n3) | casc degraded / sync holds |
| MNIST  | 8 | **0.8717 ±1.25 (n3)** | **0.963 (n3) `[h]`** | 0.9644 ±0.19 (n3) | **recipe-dependent**: WS3 recipe collapses, conv-policy recipe holds |
| FMNIST | 4 | 0.845 (n1) | — | 0.8705 (n1) | casc degraded / sync holds |
| FMNIST | 8 | **0.725 (n1)** | 0.81 (n3) `[h]` | 0.8574 (n1) | recipe-dependent |

### `deep_mlp` — CONFOUNDED band (ANN ≈ chance, NO firing-gain info)
MNIST d12/d16/d32 @w64, d16/d24 @w128; FMNIST d16/d32 @w64 — `training_floor_confound`,
both schedules. **SVHN `deep_mlp` d4/d8 join this band** (ANN only ~0.61–0.69; needs
a CNN). Excluded from every firing-gain claim.

### `lenet5` CNN — TRAINABLE (deployed harvested; consolidation pending)
| dataset | cascaded | synchronized | casc↔sync gap | note |
|:--------|---------:|-------------:|--------------:|:-----|
| MNIST  | 0.980 (n3) | 0.989 (n3) | 0.9 pp | cascaded ~free on easy grayscale |
| FMNIST | 0.840 (n3) | 0.900 (n3) | 6.0 pp | gap re-opens |
| KMNIST | 0.900 (n2) | 0.948 (n1) | ~4.8 pp | gap re-opens |
| **SVHN** | **0.670 ±1.1 (n3)** | **0.860 ±0.5 (n3)** | **19.1 pp** | **RGB blowout — kills "CNN frees cascaded"** |

### Synchronized breadth (`ws6b_*` re-queue, multi-seed, `[h]`) — see §1 table
MNIST/FMNIST hold (≤2.4 pp); **KMNIST is a real ~6 pp breadth limit (multi-seed)**;
SVHN `deep_mlp` confounded.

### `mlp_mixer_core` — keystone cells + 5-mode sweep (pm, now COMPLETE)
| dataset | cell / mode | deployed (mean ±sd, n3) | verdict |
|:--------|:------------|------------------------:|:--------|
| MNIST | matrix_6 (in-dist) cascaded | 0.96 | keystone MATCH (correct) |
| MNIST | matrix_8 (near, no-bias) cascaded | 0.9606 | keystone MATCH (correct) |
| MNIST | pm `ttfs_analytic` | **0.9805 ±0.15 pp** | tightest, highest |
| MNIST | pm `ttfs_quantized` | 0.9773 ±0.37 pp | second |
| MNIST | pm `cascaded` | 0.9652 ±1.51 pp | **highest variance** |
| MNIST | pm `synchronized` | 0.9607 ±0.20 pp | tight |
| MNIST | pm `lif` | 0.9631 ±0.30 pp | tight |

**pm read (now full 5-mode, `[h]`):** on `mlp_mixer_core`/MNIST the analytical/quantized
TTFS modes deploy tightest and highest (~0.98); LIF and synchronized land ~0.96 with
small sd; **cascaded is again the highest-variance mode** (sd 1.5 pp, range
0.95–0.986) — the fragile-cascaded-code story carries onto the mixer. Note cascaded's
*mean* (0.965) edges sync (0.961) here, but its variance makes it the unreliable
choice. Ready for a consolidated `mode_compare` verdict.

### Tally across consolidated ledger + harvested
- `synchronized_holds_near_ann`: 5 consolidated + LeNet ×4 + `ws6b` ×6 `[h]`.
- `cascaded_firing_gain_degraded`: 5 consolidated; **re-scoped** by the recipe and
  SVHN findings.
- `training_floor_confound`: 14 consolidated + SVHN `deep_mlp` `[h]`.
- keystone `match_keep_fast`: 2 correct + **18 fast-ladder escalation runs that never
  escalated** `[h]`.
- WS6 round-1 `BROKEN`: 13; clean `ws6b` re-queue: ~17 valid `[h]`.

---

## 3. Live wave (in-flight / just-drained, NOT yet consolidated)

Runner alive (`campaign_runner --poll 3 --max-per-gpu 2`); 8 running, **23 pending**,
252 done, 51 failed. State at re-synthesis:

1. **DRAINED, needs consolidation:** `pm_*` (5-mode mixer, all seeds in §2);
   `sch_lenet_*` incl. **SVHN** (§1/§2); `ws7esc_*` escalation A/B (18 runs, §1);
   `ws6b_*` breadth re-queue (§1); `cp_d8h_SVHN_*`. **None of these have been folded
   into ledger verdict records yet** — they are the harvested system-of-record gap.
2. **STILL RUNNING/PENDING:** `sch_dmlp_*` — the `deep_mlp` schedule re-sweep under
   the scheduler (FMNIST d4/d6/d8, KMNIST d4/d6/d8, SVHN d4 — cascaded/sync × 3
   seeds). This re-runs the WS3 trainable band under the *scheduler's* recipe; given
   the §1 recipe-dependence finding, **these are now the key cells to compare against
   the WS3-recipe d8 collapse** — do not treat as redundant, treat as the recipe A/B.

**Action:** run the analyzers to append consolidated `cnn_schedule_compare`
(incl. SVHN), `mode_compare` (pm), `breadth_ci` (ws6b), and `keystone_escalation`
(the 18-run no-escalation result) verdict records, then fold into the WS3/WS6/WS7
findings docs. Do not let the harvested `generated/` numbers (most of §1–§2) stand
as the system of record.

---

## 4. Open questions (ranked by leverage, re-ordered by the new wave)

1. **Which recipe knob produces the WS3 d8 cascaded 9 pp swing (0.872 → 0.963)?**
   *New #1.* The WS3 headline rung is recipe-confounded: the same model collapses
   under the WS3 recipe and holds under the conversion-policy fast-ladder recipe.
   Until this is localized (LR-find vs fixed spanning-cosine, ramp schedule, θ
   defaults), the "cascaded death-cascade with depth" claim is **not safe to ship**.
   Cheap: it is a recipe A/B on an existing trainable cell. **Highest leverage.**
2. **Does the depth × firing-gain collapse continue past d8 — or is d8 a backbone
   artifact?** Still *the* headline question, still hard-blocked on a trainable deep
   backbone (WS2/`deep_cnn`). Now compounded by #1 (must fix the recipe first or
   d≥12 inherits the confound).
3. **Is the cascaded penalty architecture-determined or difficulty-determined?**
   The SVHN LeNet 19 pp blowout says **difficulty/colour, not architecture** —
   convolution did *not* save cascaded on RGB. Need a *deep, trainable* CNN to
   separate "conv inductive bias" from "depth," and a non-MNIST mixer point.
4. **Can the keystone be made to ESCALATE at all on a real cell?** Now shown to fire
   the fast ladder on 18/18 runs. The re-test must use a cell that *actually
   death-cascades* (the WS3-recipe d8 collapse) routed through `conversion_policy`
   with C7 instrumentation. If it still won't escalate, the probe/threshold design —
   not the schedule — is the bug. Cheap, single run.
5. **Do the keystone thresholds generalize, and should `conversion_policy` be kept
   as a recipe-default applier?** The A/B shows it gives a small consistent lift
   (+0.3 to +1.8 pp) with no escalation. If escalation can't be made to fire, the
   honest framing is "conversion_policy = a better default recipe," and the
   thresholds become moot — decide which.
6. **Is KMNIST's ~6 pp synchronized breadth gap a data property or a recipe one?**
   Multi-seed confirmed and depth-worsening (5.6 → 6.7 pp d4→d8). The first
   non-trivial synchronized loss — worth a focused look (is it the same recipe knob
   as #1?).
7. **What is the accuracy↔latency(S)↔energy Pareto?** No allocator exists; every S
   is a hand-set constant. WS5 entirely unaddressed.

---

## 5. Capabilities still needed (build backlog)

Verified by source inspection at HEAD `18eee5d`
(`grep` for `deep_cnn|residual_block`, `qcfs|rmp|snn_calib`, the gate-fix flags in
`config_schema/defaults.py`, and the `builders/` directory listing).

| # | Capability | Status | Unblocks | Notes |
|---|-----------|--------|----------|-------|
| C1 | **`deep_cnn` builder** (variable-depth residual conv backbone that trains deep) | **NOT BUILT** — only `lenet5_builder.py` (fixed depth) + `torch_sequential_conv_builder.py` exist | Open-Q2, Q3 (depth-on-CNN; the headline deep test; separate conv-bias from depth) | The single highest-leverage build. LeNet is fixed-depth → cannot sweep depth on a CNN; SVHN needs it (plain MLP can't train SVHN). |
| C2 | **WS2 modern-op single-op probes**: residual / LayerNorm / GELU / attention blocks + a probe harness | **NOT BUILT** (`grep residual_block|op_probe` = empty) | Open-Q2 (a backbone that trains past d≈8); WS2 entirely; prerequisite for C1's "trains deep" | Residual+norm is what lets *any* model train deep. Attention is the largest lift. |
| C3 | **Baseline conversion methods** (RMP / QCFS / SNN-Calibration / TTFS-conversion) | **NOT BUILT** (`grep qcfs|rmp|snn_calib` = empty) | *Every* publishable claim — no external comparison point exists | Pure absence. Nothing ships without a head-to-head per the charter. |
| C4 | **Per-layer-S Pareto allocator** (budget → best-accuracy S allocation + frontier) | **NOT BUILT** (per-layer-S config plumbing exists; no allocator) | Open-Q7 (WS5); the accuracy/latency/energy story | S is hand-set per run today. |
| C5 | **WS3 recipe A/B + θ-cotrain validation on a real cell** | flags `ttfs_theta_cotrain`/`ttfs_staircase_ste`/`ttfs_staircase_ste_fast` **merged, default-off** (confirmed at `defaults.py:192/199/209`); never run on a real cascaded rung; **recipe-knob localization (Open-Q1) is the new prerequisite** | Open-Q1 (recipe swing), Open-Q4 (escalation), Open-Q3 (rescue) | The mechanism exists; the experiment doesn't. C5 now *starts* with the recipe A/B (cheap), then θ-cotrain. |
| C6 | **WS6/run-loop provenance hardening**: artifact-freshness (run-id/mtime) stamp + pre-flight conflict-marker / `git diff --check` gate | **NOT BUILT** (re-queue was manual; the trap can recur) | Trustworthy re-runs (13 cells were silently lost once) | Cheap, high-trust-value. |
| C7 | **Keystone escalation instrumentation**: log `decision.escalation_reason` + `characterization.probes` per run | **NOT BUILT** (decision is *inferred* from the driver trace — that inference is exactly how we found the 18-run no-escalation result) | Open-Q4/Q5 (read the escalation decision directly instead of inferring it) | One-line provenance log; converts inference → measurement. Now a hard prerequisite for the escalation re-test. |

---

## 6. Recommended next director phase

Sequenced cheap→expensive, kill-gated, one-axis-at-a-time. The new wave **re-orders
the cheap track**: the highest-leverage open item is no longer the escalation re-test
but the **WS3 recipe-confound localization** (Open-Q1), because it gates the entire
headline.

### Phase N (immediate — analysis + cheap probes, ~0 new GPU-days)
- **N0 (NEW, top priority). Localize the WS3 recipe confound.** Run the d8 cascaded
  recipe A/B explicitly: WS3-cluster recipe vs conversion-policy fast-ladder recipe,
  same model/seeds, logging which knob (LR source, ramp, θ) moves the 9 pp. Until
  this lands, **do not ship the WS3 "cascaded death-cascades with depth" headline** —
  it is recipe-confounded at the very rung that defines it.
- **N1. Consolidate the drained wave into ledger verdicts.** Append
  `cnn_schedule_compare` (LeNet ×4 incl. SVHN — *the corrected
  architecture-dependence headline*), `mode_compare` (pm 5-mode), `breadth_ci`
  (ws6b synchronized ×4 datasets with the §1 CIs), and `keystone_escalation` (the
  18-run no-escalation A/B + the conversion_policy lift). Fold into the WS3/WS6/WS7
  findings docs. **This is where most of §1–§2 graduates from harvested to recorded.**
- **N2. Land C7, then re-run the escalation re-test on a model that actually
  death-cascades** (the WS3-recipe d8 collapse, *not* the fast-ladder recipe that
  doesn't collapse), `conversion_policy:true`, instrumented. Either it escalates
  (validates the keystone) or it doesn't (the probe/threshold design is the bug) —
  both publishable, and only now testable because we know the fast-ladder cells
  don't collapse.
- **N3. Land C6 (provenance gate)** before any further re-queue, so the WS6 breadth
  CIs (already harvested) can be re-confirmed and trusted.

### Phase N+1 (the headline unlock — requires builds)
- **Build C2 (residual/norm op-blocks) → then C1 (`deep_cnn`).** Coupled:
  residual+norm is what lets any model train past d≈8 *and* is what SVHN needs.
  Kill-gate: the new backbone's **ANN must train ≫ chance at d≥12** (the exact thing
  plain MLP + w128 + SVHN-MLP all failed). Only then re-run the d12/d16/d24
  cascaded-vs-sync sweep — answering Open-Q2/Q3 (does the collapse deepen; is it
  conv-bias or depth) **on a recipe already de-confounded by N0**.
- **Run C5's θ-cotrain trial** in parallel (no backbone needed), but only *after* N0
  tells us whether there is still a deficit to rescue.

### Phase N+2 (publishability + frontier)
- **Build C3 (baselines).** No claim ships without a head-to-head — the gate between
  "we have an engine" and "we have a paper." RMP/QCFS first (cheapest, most-cited),
  then SNN-Calibration.
- **Build C4 (per-layer-S Pareto allocator).** Opens WS5 — the second paper axis.

### Capability build order (dependency-sorted)
`C7, C6` (cheap, this wave) → **N0 recipe A/B** (cheap, gates the headline) →
`C2 → C1` (deep-backbone long pole) → `C5` (parallel) → `C3` (publishability gate) →
`C4` (frontier).

---

## 7. Honest limits (one place)

- **Most CNN, pm, ws6b, and ws7esc numbers in §1–§2 are harvested from `generated/`
  artifacts, not consolidated ledger verdicts** (`[h]`). They are real run outputs
  but have not passed the analyzer that writes per-cell verdict records with CIs.
  Strong evidence, not the system of record, until N1 lands.
- **The WS3 d8 cascaded "collapse" is recipe-confounded** — the same model deploys
  0.872 (WS3 recipe) or 0.963 (conversion-policy recipe). The depth-collapse headline
  is **not safe to ship until N0 localizes the knob.**
- **No firing-gain claim holds past d8** — the deep band is a training-floor confound.
- **"CNN makes cascaded free" is FALSE in general** — true only on easy grayscale;
  SVHN LeNet cascaded loses 19 pp to sync. The robust claim is "synchronized is the
  universal near-lossless default."
- **The keystone escalation path has been shown NOT to fire on 18/18 real runs.**
  Only MATCH/keep-fast is demonstrated; `conversion_policy` currently acts as a
  recipe-default applier (+0.3 to +1.8 pp), not an adaptive selector.
- **KMNIST is a genuine synchronized breadth limit** (~6 pp, multi-seed,
  depth-worsening) — synchronized is *not* universally lossless across datasets.
- **SVHN `deep_mlp` is a non-result** (ANN ~0.61–0.69; backbone limit) — SVHN
  requires the CNN backbone.
- **No baseline exists** — every accuracy here is absolute, not relative to a
  published conversion method. Nothing is publishable until C3.
- **C6 (provenance gate) is still unbuilt** — the merge-conflict-marker trap that
  silently lost 13 WS6 cells can recur.

### Key references
- Charter: `docs/research/RESEARCH_PROGRAM.md` · North-star:
  `docs/mimarsinan_closeout_analysis_v2.md` · GPU substrate:
  `docs/PUBLICATION_CAMPAIGN.md`
- Findings: `docs/research/findings/{WS3_depth_firing_gain,WS1_WS6_breadth_rigor,WS7_keystone_automatic}.md`
- Ledger: `scripts/campaign/research_loop.py ledger-read` → `runs/campaign/ledger.jsonl`
- Keystone code: `src/mimarsinan/tuning/orchestration/{characterization,conversion_policy}.py`,
  `tuning/tuners/ttfs_cycle_adaptation_tuner.py`, `tuning/orchestration/ttfs_adaptation_plan.py`
- Gate-fix flags (merged, default-off): `src/mimarsinan/config_schema/defaults.py:192/199/209`
  (`ttfs_theta_cotrain` / `ttfs_staircase_ste` / `ttfs_staircase_ste_fast`)
- Builders (capability ground truth): `src/mimarsinan/models/builders/`
  (`lenet5_builder.py`, `torch_sequential_conv_builder.py` exist; **no `deep_cnn`**)
- Harvested artifacts (this synthesis): `generated/{ws7esc_*,sch_lenet_SVHN_*,pm_*,ws6b_*,cp_d8h_SVHN_*}_phased_deployment_run/__target_metric.json`
- Live queue: `runs/campaign/q/{pending,running,done,failed}/`, logs in `runs/campaign/logs/`
