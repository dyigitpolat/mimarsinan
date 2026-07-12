# Lossless refinement ledger — LIF and TTFS-sync MNIST deployments (wave 2026-07-12)

**Mandate.** For every LIF and TTFS-sync MNIST deployment, deployed accuracy must
MATCH pretraining accuracy. Definitions used throughout:

- **STRICT lossless**: `deployed >= pretrain`.
- **TOLERANCED lossless**: `deployed >= pretrain − 2·SE`, `SE = sqrt(p(1−p)/10000)`
  at `p = pretrain` (~0.13–0.20 pp on this wave; ≈0.17 pp at p = 0.97).
- **Sanctioned levers**: analytical (arithmetic-consistency), statistical
  (calibration-only), and spec-hygiene corrections ONLY. "Train longer / more
  budget" is never a lever. Where a QAT/adaptation stage is itself the loser,
  the ledger names the arithmetic term the training cannot express.

**Sources.** 21 runs of 2026-07-12 (logs at
`~/.slurmech/workspaces/mimarsinan-xlog1/runs/<run_id>/artifacts/stdout.log`;
run_ids in §1). Theory base: `lif_deployment_exactness.md` (Theorems 0–3,
corrections C1–C6, violations V1–V9), `sync_deployment_exactness.md`
(Theorem 1, preconditions P1–P6, corrections E1–E5/S1–S4),
`mixer_column_scale_pathology.md` (M4 migration, §3.3 axis-flip limit),
`wq_cascade_crater_repair.md` (M2 two-scale, θ̂ law),
`numerical_boundary_consistency.md` (seeding, metric-grade eval). Recipe
arming: `src/mimarsinan/tuning/orchestration/conversion_policy.py`.

**Conventions.** Stage columns are the `[PROFILE]` full-test (10k) reads; all
deltas quoted below are signed pp, negative = loss, later-minus-earlier
(the sync extraction's `loss = earlier − later` convention is normalized away
here). `wq entry` numbers quoted in §2 are `[MBH-ENDPOINT]` eval-subset reads
(different domain — never differenced against test reads). "deployed" = the
last captured deployment read (Loihi/Simulation where present, else HCM;
sync-nonmixer logs truncate at SANA-FE with parity 1.0 confirmed via
live_metrics for t0_22/t0_24/t0_25, unverifiable for t0_23). Replication
clones reproduced their anchors bit-identically → run seeding works; draw
variance on this wave is 0 and the SE tolerance is purely test-sampling.

---

## 1. Per-cell ledger

Toleranced bar = pretrain − 2·SE. Net = deployed − pretrain (pp).

### LIF mixer (pretrain 0.9820, bar 0.9793)

| cell (run_id) | S/wb | AQ | adapt | WQ | deployed | net | strict | toler. | dominant loss (pp) |
|---|---|---|---|---|---|---|---|---|---|
| t0_01 (…-9f94f95d) | 4/5 | 0.9719 | 0.9249 | 0.9326 | 0.9326 | −4.94 | NO | NO | LIF adaptation −4.70 |
| t01_01 (…-de58f020) | 8/5 | 0.9754 | 0.9585 | 0.9459 | 0.9459 | −3.61 | NO | NO | LIF −1.69, fold-leg −1.67 |
| t01_02 (…-c6d57142) | 16/5 | 0.9779 | 0.9688 | 0.9665 | 0.9665 | −1.55 | NO | NO | LIF −0.91, shift −0.81 |
| t01_08 (…-c88da3c7) replica of t0_01 | 4/5 | 0.9719 | 0.9249 | 0.9326 | 0.9326 | −4.94 | NO | NO | (bit-identical to anchor) |
| t01_21 (…-67a8e131) | 4/8 | 0.9719 | 0.9249 | 0.9451 | 0.9451 | −3.69 | NO | NO | LIF −4.70 (WQ@wb8 +3.56) |

### LIF non-mixer

| cell (run_id) | vehicle, S | pretrain | AQ | adapt | WQ | deployed | net | strict | toler. (bar) | dominant loss |
|---|---|---|---|---|---|---|---|---|---|---|
| t0_02 (…-0ed3d092) | lenet5 fp s8 novena offload pruned | 0.9928 | 0.9844 | 0.9788 | — | 0.9788 | −1.40 | NO | NO (.9911) | pruning −0.80, LIF/Novena −0.56 |
| t0_03 (…-4fef1565) | deepcnn d8 s16 | 0.9944 | 0.9946 | 0.9949 | 0.9925 | 0.9925 | −0.19 | NO | NO by 0.04 (.9929) | WQ −0.24 |
| t0_04 (…-af02cb87) | deepmlp d8 s32 | 0.9563 | 0.9623 | 0.9608 | 0.9613 | 0.9613 | **+0.50** | **YES** | YES (.9522) | — |
| t0_05 (…-163ace0e) | simplemlp s4 | 0.9789 | 0.9809 | 0.9766 | 0.9712 | 0.9737 | −0.52 | NO | NO by 0.23 (.9760) | WQ −0.52, LIF −0.43 (SCM +0.25) |
| t01_16 (…-cbe1a769) | deepcnn d8 s8 | 0.9944 | 0.9952 | 0.9945 | 0.9938 | 0.9938 | −0.06 | NO | **YES** (.9929) | WQ −0.07 |
| t01_19 (…-cb10d136) | deepcnn d6 s16 | 0.9919 | 0.9940 | 0.9935 | 0.9878 | 0.9878 | −0.41 | NO | NO by 0.23 (.9901) | WQ −0.57 |

### Sync mixer (pretrain 0.9820, bar 0.9793; ALL pruned10 — prune −1.05 baked into every net)

| cell (run_id) | S/wb | AQ=adapt | WQ | deployed | net | strict | toler. | dominant loss |
|---|---|---|---|---|---|---|---|---|
| t0_21 (…-c06d3bf2) | 8/5 | 0.9494 | 0.9603 | 0.9603 | −2.17 | NO | NO | AQ −2.37, prune −1.05 |
| t01_04 (…-ad1e3619) | 16/5 | 0.9703 | 0.9648 | 0.9648 | −1.72 | NO | NO | prune −1.05, WQ −0.55 |
| t01_05 (…-21fa617b) | 4/5 | 0.9410 | 0.9471 | 0.9471 | −3.49 | NO | NO | AQ −2.94, clamp −0.49 |
| t01_09 (…-6f04a4cf) replica of t0_21 (config defect) | 8/5 | 0.9494 | 0.9603 | 0.9603 | −2.17 | NO | NO | (bit-identical to anchor) |
| t01_20 (…-11cfc536) | 8/8 | 0.9494 | 0.9669 | 0.9669 | −1.51 | NO | NO | AQ −2.37 (WQ@wb8 +1.75) |
| t01_24 (…-d79b8fc4) replica of t0_21 (config defect) | 8/5 | 0.9494 | 0.9603 | 0.9603 | −2.17 | NO | NO | (bit-identical to anchor) |

### Sync non-mixer

| cell (run_id) | vehicle, S | pretrain | AQ=adapt | WQ | deployed | net | strict | toler. (bar) | dominant loss |
|---|---|---|---|---|---|---|---|---|---|
| t0_22 (…-9aadb7c3) | lenet5 s4 sched | 0.9928 | 0.9844 | 0.9908 | 0.9908 | −0.20 | NO | NO by 0.03 (.9911) | AQ −0.60, clamp −0.28 |
| t0_23 (…-416426cc) | deepcnn d4 s16 | 0.9906 | 0.9941 | 0.9934 | 0.9934 | **+0.28** | **YES** | YES (.9887) | — |
| t0_24 (…-1f831836) | deepmlp d4 s8 | 0.9761 | 0.9748 | 0.9761 | 0.9761 | **+0.00** | **YES (=)** | YES (.9730) | — (AQ −0.40 recovered by WQ) |
| t0_25 (…-e2ca9ff4) | simplemlp s32 | 0.9770 | 0.9826 | 0.9822 | 0.9822 | **+0.52** | **YES** | YES (.9740) | — |

**Scoreboard.** Strict 4/21 (t0_04, t0_23, t0_24, t0_25 — all non-mixer, all
S ≥ 8). Toleranced-only 1/21 (t01_16). Near-misses ≤ 0.25 pp below the bar:
t0_03 (0.04), t0_22 (0.03), t0_05 (0.23), t01_19 (0.23). Every mixer cell
fails by 1.5–4.9 pp. The deployment tail is exactly lossless in 21/21 cells
(WQ read == NF == SCM == HCM == Sim == Loihi; sole exception t0_05's +0.25 pp
SCM GAIN — see §2F): the sync Theorem 1 and the LIF commutation theorem hold
at machine precision in deployment; **all loss is upstream of mapping**.

---

## 2. Loss attribution — theory term per losing stage

### A. Scalar-θ staircase starvation at low S (the AQ leg)

**Cells/evidence:** sync mixer AQ −2.94 (S=4, A6 7/9 hops starved + quantile
deflate on 5 hops), −2.37 (S=8, 6/9), −0.28 (S=16, 1/9); AQ endpoint entries
0.50–0.78; lif mixer shift+AQ net −1.01/−0.66/−0.41 at S=4/8/16 (A6 5/1/0
starved hops); t0_22 AQ −0.60 (S=4, 2/4 hops, masses 0.86/0.75, only
starvation FAIL of its group and the only unreached AQ endpoint).

**Theory term:** composed floor/ceil-staircase drift-then-saturate under a
scalar θ per hop (`sync_deployment_exactness.md` §3, Identity 1; mixer memo
§2). The loss is monotone in `starved_hops` and ∝ 1/S exactly as the gauges
predict.

**Landed levers that engaged (and why loss remains):**
- `sync_entry_half_step` + the E1 encoder fold (landed, `[MBH-B1] … folded on
  9 hops` includes the subsumed encoder; commit f4763e67) — the first-moment
  term is cancelled; the residual is second-moment + within-channel tails.
- M4 scale migration **fired in all 10 mixer cells** at its full theoretical
  adjacency set (5 pairs = the §3.3 matching-axis set, clip 4.0, read
  unchanged as designed) and on deepmlp (8) / simplemlp (3) / lenet5-sync (2);
  deepcnn migrated 0 (homogeneous — expected no-op). **"Migration not firing"
  is refuted.** The residual starvation lives exactly where the memo says
  migration cannot reach: axis-flipped hops (patch_embed, token-fc2 outputs —
  the starved A6 hops in every mixer gauge) and within-channel heavy tails
  (starved_mass ~0.5–0.75 persists post-migration; mixer memo limit 3).
- `starvation_aware_scale_quantile` (binary 1.0→0.99 deflate) fired at sync
  S=4 — it is the first rung of S1; the measured optimum is an (S,
  depth)-dependent quantile, not landed (sync §3.3).

**Not landed, named by the theory:** S1 quantile-descent θ, S2 per-channel θ
(exact on matching-axis edges TODAY; blocked elsewhere by the three
scalar-collapse seams `activation_scales.py:49-63`,
`adaptation_manager.py:104-113` (+ `scale_aware_boundaries.py:31-38`),
`scale_propagation.py:76-94`), S3 sequential first-moment fold.

### B. LIF rate-grid conversion gap (the LIF-adaptation leg)

**Cells/evidence:** lif mixer −4.70/−1.69/−0.91 at S=4/8/16 (clean 1/S);
endpoint entries 0.875/0.894/0.968 recovered in-loop; temporal A6 FAIL at
S=4/8 (total first-fire delay 21.1/24.6 vs windows 4/8, per-depth 1.6–6.1);
t0_02 −0.56 (Novena, S=8, 2/4 value-starved hops); t0_05 −0.43 (S=4 shallow).

**Theory terms:** by Theorem 2 the deployed LIF chain ≡ the composed integer
staircase, so this leg decomposes into (i) the shared starved-θ grid distance
(term A above — the nearest-staircase twin itself sits ~3 pp below float at
S=4 on the 9-hop shape), (ii) V3 back-loading (the temporal-FAIL signature;
killed value-exactly by C3 re-timing, +1.9 pp at S=4 — the mixer trunk is ONE
neural segment, so it never re-times between its 8 hops), (iii) V5 dead-zone
bias (C4's target), and (iv) V7 Novena discard on t0_02 (Theorem 0 broken;
only expectation-level repair exists, the C4 Novena arm gated S ≥ 8 — which
never ran, see D).

**Endpoint verdict (mandate rule):** at S=16 the tuner's recovery added
exactly zero (entry == exit == 0.967618, divergence_rescued) — the training
sits in a piecewise-constant basin; the unexpressed arithmetic terms are
(ii)+(iii) plus term A's grid capacity, not budget.

### C. WQ grid — the shared 5-bit max(|w̃|,|b̃|) lattice

**Cells/evidence:** t01_19 −0.57 (WQ endpoint frozen: entry == exit ==
0.985028, divergence_rescued), t0_03 −0.24 (same freeze pattern), t01_16
−0.07, t0_05 −0.52; lif-mixer WQ eval-entry craters 0.7998 (wb5) vs 0.9091
(wb8) with the wb8 control worth **+1.25 pp deployed** at S=4 purely from
grid resolution (and halved WQ wall); sync S=16 WQ −0.55 with a frozen,
rescued endpoint; sync wb8 control +0.66 pp.

**Theory terms:** M2's bias-set shared grid (`wq_cascade_crater_repair.md`
§2–3) — and a LIF-specific aggravation: the armed `lif_half_step_bias` folds
+θ/(2T) into the effective bias BEFORE the WQ QAT, i.e. the exactness fix
itself adds bias mass to the very row that sets the shared grid, worst at low
S. For sync S=16 the residual is the P4 half-step-on-bias-lattice erosion
(`g_b/(1/(2S))` grows with S; blanket refold measured −1.6 pp — the exact
repair is E3, comparator-side half-step).

**Arming gap:** `wq_two_scale_projection` is landed and armed for sync and
cascaded (`conversion_policy.py:107,131`) but **absent from
`_LIF_RECIPE_KNOBS` (conversion_policy.py:46-77)**. The resolver gate
(`weight_quantization_step.py:27-35`) requires on-chip bias — tier-0
platforms are `has_bias: true`, so the gate passes; nothing but the recipe
row blocks it.

**Endpoint verdict (mandate rule):** four frozen WQ endpoints
(t0_03, t01_19 ×2 tuners, t01_04) prove the QAT cannot express the repair:
the loss surface is piecewise-constant at grid scale; the unexpressed terms
are the bias-row grid coupling (two-scale) and the half-step's lattice
placement (E3) — never recovery steps.

### D. Affine-fold premise skip + pre-applied entry fold (the fold leg)

**Cells/evidence:** `lif_affine_fold: {'folded': 0, 'skipped_premise': True}`
in **11/11 LIF cells where the step ran**; the step read still moved
−1.54/−1.67/−0.51 pp (mixer S=4/8/16) because the half-step entry fold is
applied at `lif_affine_fold_step.py:57-63` BEFORE the premise check and stays
applied on skip; WQ then reconciles only partially (s8 fold+WQ net −1.26 pp).
On t0_02 (fp) the step never scheduled at all: `applies_to` requires
`plan.activation_quantization` (`lif_affine_fold_step.py:30-35`), which the
fp config does not set — so the **Novena C4 repair (the only sanctioned fix
for the Theorem-0 breakage, +8.1 pp class at S=8 on chains) never ran on the
one Novena cell of the wave.**

**Theory term / seam:** `crater_premise_holds` (`lif_affine_fold.py:199-227`)
uses the clamp-envelope forward as teacher and skips when deployed ≥ envelope
— but the LIF-adaptation endpoint TRAINS the deployed composition past the
float twin (the docstring says exactly this), so post-adaptation the premise
is false by construction: **C4 is armed yet structurally unreachable exactly
where the memo predicts +0.3…+1.6 pp (dead-zone absorption) and where the
Novena arm lives.** The premise itself is honest (sequential folds with a
worse teacher measured 0.9574→0.7684); the defect is teacher staleness plus
the fold-before-premise sequencing. This is also a **C2 interaction**: the
armed membrane readout lifts the deployed read, making `deployed ≥ envelope`
(and hence the skip) more likely — two armed levers competing at one seam.

### E. Pruning (structural, not a conversion term)

−1.05 pp in all six sync-mixer cells, −0.80 pp in t0_02, identical everywhere.
No arithmetic-consistency lever can restore a removed function. Under the
mandate, pruned specs can never be strict-lossless against the pre-prune
pretrain read; §4 states the re-referencing decision.

### F. Cross-cutting statistical / verification notes

- **Draw variance is closed:** replication clones bit-identical (seeding fix
  landed per `numerical_boundary_consistency.md` (a)); the SE tolerance now
  measures only test-set sampling.
- **t0_05 SCM +0.25 pp GAIN** — the only post-WQ read that moved in 21 cells.
  A movement in EITHER direction is a parity-contract deviation (mapping must
  be read-exact); audit alongside R8.
- **Config defects:** `t01_09_…e4.json` and `t01_24_…floor.json` are
  byte-identical to the t0_21 baseline except `experiment_name` — the e4/floor
  knob deltas were never generated (two burnt slots; the bit-identical reruns
  are the proof). `t01_08` was a declared clone (variance calibration) — fine.
- **Verification coverage:** all logs truncate at/before SANA-FE; t0_23 has no
  preserved workspace (SANA-FE outcome unverifiable); t01_16 lacks
  Simulation/Loihi reads. t0_24 (32.75%) and t0_25 (25.42%) are VALID_FLAGGED
  below the on-chip-majority constraint — a validity concern independent of
  accuracy.
- **Clamp:** ClampTuner constructive-stalls everywhere (inert by design); the
  one negative clamp read (t01_05, −0.49) folds into term A's S=4 regime.

---

## 3. Ranked refinement plan toward lossless

Ranking = (covered pp × exactness class × readiness). Every action is
analytical or statistical; none spends training budget.

| # | mechanism | exactness class | integration point | cells covered | predicted pp | prerequisite |
|---|---|---|---|---|---|---|
| **R1** | Arm two-scale WQ projection for LIF (`wq_two_scale_projection` into `_LIF_RECIPE_KNOBS`) | exact identity (grid refactoring, same bit budget) | `tuning/orchestration/conversion_policy.py:46-77`; gate already passes at `pipelining/pipeline_steps/quantization/weight_quantization_step.py:27-35` (tier-0 = on-chip bias) | all 9 LIF-wq cells; binds on t01_19 (−0.57), t0_05 (−0.52), t0_03 (−0.24), t01_16 (−0.07), lif-mixer WQ entries | +0.2…+1.3 on grid-bound cells (wb8 control +1.25 deployed at S=4; M2 exact-recovery table); residual weight rounding ≤ 0.6 (lif §5) | none (transform landed 84743bdc; verify `chip_quantize.py` second-scale export per M2 §5) |
| **R2** | Repair C4 affine fold reachability: (a) teacher = pre-adaptation post-AQ staircase snapshot (not the stale clamp envelope), premise on per-channel first-moment mismatch; (b) admit fp-LIF cells in `applies_to`; (c) move the entry half-step fold behind the premise or reconcile in-step | statistical (C4, full affine only) + arithmetic sequencing | `pipelining/pipeline_steps/adaptation/lif_affine_fold_step.py:30-35,57-63`; `tuning/lif_affine_fold.py:199-227` | all 11 LIF cells (11/11 premise-skips); t0_02's Novena arm | +0.3…+1.6 (C4 bound, dead-zone absorption); Novena-class up to +8 on t0_02's LIF leg; kills the −0.5…−1.7 fold-leg read leak | keep the S≥8 Novena gate; full affine only (bias-only refuted −4.2) |
| **R3** | Per-channel θ (S2) on matching-axis edges now; E4 plumbing for the 3 scalar-collapse seams next | exact scale-space identity (`Q_S(z/θ)` per channel; decode folded into consumer `per_input_scales`) | container `spiking/theta_cotrain.py:19-40`; `mapping/mappers/scale_propagation.py:62-74`; seams `mapping/support/activation_scales.py:49-63`, `tuning/orchestration/adaptation_manager.py:104-113`, `scale_propagation.py:76-94` | all 10 mixer cells (lif+sync); deep_mlp family prophylactically | +1.0–1.1 at S=8/16, up to +4.4 at S=4 (sync A5/B4 vs B1); the ONLY lever that beats the 1/S law (lif §5) | threshold-group packing dry-run (`capacity/dryrun.py` oracle) — per-neuron thresholds fragment packing groups; export via `packing/canonical.py:96-101` |
| **R4** | S1 quantile-descent θ: per-hop candidate scan {0.90,0.95,0.99,0.995,1.0} scored on the deployed calib forward; make LIF's fixed q=0.99 S-aware; keep the deflate as floor candidate | statistical (calibration-only) | generalize `pipelining/pipeline_steps/adaptation/activation_analysis_step.py:218-245`; stats already in `install_resolution/capture.py` | every AQ-losing cell: sync mixer S=4/8, t0_22, lif mixer | up to +2.4 beyond the binary deflate (sync B1 vs A2); +0.1–0.2 lenet-class → flips t0_22 toleranced | score on full calib cache (4k-sample greedy has ~0.7 pp selection noise) |
| **R5** | C3 per-hop re-timing (boundary decode + count-preserving re-encode) for single-segment LIF chains at S ≤ 8 | value-exact (transcode identity `round((c/T)·T) = c`) | mapping-level segmentation option; machinery exists at `models/spiking/hybrid/rate_forward.py:110-113`, `spiking/segment_boundary.py:181+` (lif memo §7.3) | lif mixer S=4/8 (the temporal-A6 FAIL cells: delay 21.1/24.6 vs windows 4/8) | +1.9 at S=4, +0.5 at S=8 (chain9); nil at S ≥ 16 | gate on temporal-A6 FAIL + single-segment; latency/energy cost is mapping-visible |
| **R6** | S3 sequential first-moment fold for sync (DFQ family, own-offset-EXCLUDED) | statistical (closed-form calibration) | `spiking/dfq_bias_correction.py` seam at AQ install, before endpoint recovery; gate `is_synchronized_ttfs` | sync mixer (largest at S=4); post-E1 residual entry systematics | +0.1…+0.9 (sync B2) | the §3.2 sign trap: folding the raw mean gap cancels the half-step (0.93→0.59 measured) — exclusion is load-bearing |
| **R7** | E3 comparator-side half-step at WQ: carry +θ/(2S) in the per-core threshold when the bias lattice is too coarse | exact identity (threshold shift ≡ bias fold; zero bit cost) | measure `g_b/(1/(2S))` at projection (`transformations/normalization_aware_perceptron_quantization.py:67-83`); arm when ≥ ~0.5 | sync S=16 (t01_04 WQ −0.55, frozen endpoint); any higher-S / fewer-bias-bit sync; LIF analogue after R1 | +0.3…+0.6 at sync S=16; prevents the −1.6 blanket-refold trap | SCM threshold is float64 (not integer-locked); nevresim n/a for sync |
| **R8** | C2 membrane-readout engagement audit + decode-domain parity: verify the armed `lif_membrane_readout` actually reaches the deployed read; explain t0_05 (theory: mlp-S4 +2.3 ABOVE staircase with C2; measured −0.43 below) and the t0_05 +0.25 SCM jump; document the C2↔C4-premise interaction (§2D) | verification of an exact identity (Theorem 0: `Q = θ·c_T + m_T`) | `models/spiking/hybrid/lif_step.py:199-217`; decode `rate_forward.py:113,199`; parity locks per lif memo §7.5 | all LIF; decisive for t0_05, contributory for the mixer S=4 family | 0 direct; unlocks up to +2 where C2 is armed-but-inert; zero risk | none |
| **R9** | Spec hygiene: (a) decide the pruning reference — drop `pruned10` from lossless-mandate cells or bind the mandate to the post-prune envelope; (b) regenerate `t01_09_e4` / `t01_24_floor` with their real knob deltas (`generate.py` defect, byte-identical configs) | spec / statistics | `test_configs/generate.py`; campaign DoD text | sync-mixer family + t0_02; two burnt matrix slots | removes a structural −1.05/−0.80 that NO sanctioned lever can touch | user decision on the reference |
| **R10** | S=4 mixer ceiling adjudication: treat the AQ'd staircase ANN as the object that must match pretrain; conversion below it is closed (deployment tail is exact; B-terms close with R2/R5/R8). At S=4 × depth-9 × scalar-θ the staircase itself cannot reach pretrain — refinement target is AQ capacity (R3 + QAT level-assignment), not conversion | theorem-bounded feasibility statement | — | mixer S=4 cells (t0_01/t01_08/t01_21, t01_05) | re-adjudicate after R3; do NOT spend levers chasing conversion here | R3 measured at S=4 |

**Refuted branches — do not re-derive** (all measured): threshold-guard ramps
for multi-spike LIF (Theorem 3; 0.7825 vs 0.9180), V0 half-step placement,
encode phase stagger, stochastic encoding (√S vs 1/S law), bias-only affine
folds (−4.2), blanket post-WQ half-step refold (−1.6 at S=16), sum-ratio θ̂
rescale (0.95→0.50), per-channel MSE offsets beyond the half-step (S4 ≡ B1,
closed negative result).

---

## 4. Feasibility per cell family (honest)

- **Sync non-mixer** — 3/4 strict TODAY (t0_23, t0_24, t0_25). t0_22 (lenet5
  S=4): toleranced with R4 (needs +0.03); strict needs +0.20 and the sync
  memo's full analytic stack tops at −0.14 pp to float at lenet-S=4 (B1
  0.9902 vs 0.9916) — **strict at S=4 is inside test-sampling noise, not
  provable; toleranced is the honest commitment.**
- **LIF non-mixer, unpruned** — t0_04 strict today. t01_16 (needs +0.06) and
  t0_03 (+0.19): R1 covers the named WQ term; strict PLAUSIBLE, toleranced
  expected. t01_19 (+0.41): R1 is the matching lever (its endpoint froze on
  the grid term); toleranced expected, strict plausible. t0_05 (+0.52): R1 +
  R8 (the C2 audit — theory holds +2.3 pp at mlp-S4 that is armed but not
  observed) → strict plausible once C2 is verified live. Ceiling citation:
  post-two-scale weight rounding at 5 bits is bounded ≤ 0.6 pp (lif §5) and
  is not sign-guaranteed — cells whose gap is smaller than that bound can
  only be committed at TOLERANCED.
- **LIF non-mixer, pruned/fp (t0_02)** — strict vs pre-prune pretrain is
  IMPOSSIBLE by construction (−0.80 structural). Vs the post-prune envelope
  (0.9848): needs +0.60, and R2(b) unlocks the Novena C4 repair whose class
  value (+8.1 pp on chains at S=8) dwarfs the gap → toleranced-vs-post-prune
  expected, strict-vs-post-prune plausible. Requires the R9 re-reference
  decision first.
- **LIF mixer** — S=16 (−1.55): R2+R3+R4 stack predicts +1.5…+2.5 →
  toleranced PLAUSIBLE, strict open. S=8 (−3.61): stack (R1 0.5–1.3, R2
  0.3–1.6, R3 ~1.0, R5 0.5, R4 small) sums to the gap only at the optimistic
  edge → toleranced BORDERLINE, must be measured, strict not this round.
  S=4 (−4.94): **ceiling** — the nearest-staircase twin is ~3 pp below float
  at this shape (lif §6), the sync capacity wall reads −9.3 pp for the
  analytic stack (sync §4.1), and nothing in the correction space beats 1/S
  except more levels per channel (lif §5). Neither strict nor toleranced is
  claimable at scalar θ; re-adjudicate after R3 per R10.
- **Sync mixer** — vs pre-prune pretrain: INFEASIBLE while `pruned10` stands
  (§2E). Vs the post-prune envelope (0.9715): S=16 needs +0.67 → R7+R4+R6
  plausible toleranced; S=8 needs +1.12 → R3+R4+R6 plausible toleranced
  (wb8 already banks +0.66 of it); S=4 needs +2.44 against the §4.1 capacity
  wall → not claimable at scalar θ (R10). Strict even vs post-prune remains
  open until R3 numbers exist.

**True ceilings (theorem citations):** (1) S=4 × depth≥8 scalar-θ capacity
wall — sync §4.1, mixer memo §6 limit 3, lif §5 1/S law; (2) pruning — outside
the conversion algebra entirely; (3) 5-bit weight rounding residual ≤ 0.6 pp
post-two-scale, sign-unguaranteed — lif §5, sync headline 7; (4) strict-at-
equality is a coin flip when the true gap ≈ 0 on a single 10k draw — the
toleranced definition exists precisely for this.

---

## 5. Tentative generic advisory rules (for the programmatic failure-mode framework)

Pattern-based only — no vehicle names. Each rule: trigger pattern → predicted
loss (mechanism).

- **G1 starved-composition:** value-gauge `starved_hops/hops ≥ 1/2` at grid S
  on a staircase chain of depth ≥ 6 → AQ-leg loss ≥ ~1.5 pp (S=8) / ≥ ~3 pp
  (S=4), concave in depth (drift-then-saturate; entry hops dominate — fix
  entry first). [sync §3.2; this wave: −2.94/−2.37 mixer, −0.60 lenet-S4]
- **G2 temporal back-loading:** Σ_depth(first-fire delay) > window on a
  single-segment chain → emission-cap deficit ~1–2 pp at S ≤ 8, repaired
  value-exactly by boundary re-timing; nil at S ≥ 16. [lif V3/C3; the two
  temporal-A6 FAIL cells]
- **G3 axis-flip spread:** per-channel q99 spread > ~10× on a weight-shared
  (axis-flipped) edge → weight-space migration void there; foregone gain =
  per-channel-θ delta (~+1 pp at S ≥ 8, up to +4 pp at S=4). [mixer §3.3,
  sync §4.2–4.3]
- **G4 bias-dominated grid:** `max|b̃|/max|w̃| ≥ ~2` on any perceptron under a
  shared WQ grid → weight-level annihilation; crater scales with S·g_b in
  timing modes, level starvation in value modes; two-scale repairs. Corollary:
  any half-step fold ADDS bias mass (θ/(2S) or θ/(2T)) — exactness folds and
  shared grids interact. [M2 §2–3; lif-mixer wb5 entry 0.80 vs wb8 0.91]
- **G5 frozen endpoint:** endpoint `entry == exit` + divergence-rescued from a
  NON-cratered entry → the residual is grid arithmetic (piecewise-constant
  loss at grid scale); training cannot express it; route to the matching
  arithmetic lever, never to budget. [×5 this wave: t0_03 (2 tuners), t01_19
  (2), t01_04-WQ]
- **G6 half-step/lattice ratio:** `g_b/(θ/(2S)) ∈ [~0.5, ~1.5]` at projection
  → the folded half-step erodes on the bias lattice; carry it in the
  comparator/threshold (exact, zero bit cost); never blanket-refold. [sync §5;
  t01_04 S=16 −0.55]
- **G7 armed-lever witness:** every armed exactness lever must emit a per-cell
  scheduling/engagement witness; premise-gated or predicate-excluded arms are
  silent losses (11/11 C4 skips; the fp cell's absent fold step; C2
  unverified at the one cell theory says it must dominate).
- **G8 structural re-reference:** any function-changing spec item (pruning
  here) shifts the attainable envelope by its own measured delta; lossless
  mandates must bind to the post-structural envelope or exclude the item.
- **G9 replication integrity:** with seeding landed, a "variant" whose stage
  trajectory matches its anchor to 6 decimals is a generator defect, not a
  null result (two slots burnt this wave).
- **G10 encoder-hop parity:** any hop executing a floor/ceil kernel without a
  mid-tread offset contributes a −θ_hop/(2S) same-sign entry drift; count such
  hops (subsumed encoders included) — each is a whole-population floor risk at
  low S. [sync E1, landed; keep as an invariant check]
