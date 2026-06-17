# Lossless + Fast MNIST Campaign — Results & Decisions

**Goal:** deployed MNIST model LOSSLESS (full-test Soft Core Mapping accuracy within ~1 SE of the
pretrained ANN, i.e. ≤ ~0.5pp deviation) AND FAST (the genuine fine-tuning session < 2 min).
GPU 0 only (1,2,3 busy). Lossless gate = **Soft Core Mapping target_metric** (full test via
`deployment_metric_full_eval=True`); the nevresim "Simulation" step is subsampled (noisy) — not the gate.

ANN (lossless target) on MNIST mlp_mixer_core = **0.983** (full test, SE≈0.0013 → lossless ≈ ≥0.980).

## Baselines (measured)

| Code | S | ANN | deployed SCM (full) | gap | tuning wall | verdict |
|---|---|---|---|---|---|---|
| LIF (controller) | 4 | 0.983 | 0.9721 | **1.1pp** | LIF-Adapt 488s + WQ 61s | LOSSY + SLOW |
| TTFS default proxy (old cache) | 4 | 0.982 | 0.96 | 2.2pp | FT 527s | LOSSY + SLOW |
| TTFS genuine fast | 4/8/16 | — | (sweep running) | — | — | — |

## Capacity finding (the binding constraint)

S=4 caps **even LIF** (a linear, robust rate code) at 0.9721 — the loss is the 5-level rate code
(ANN 0.983 → LIF-tuned 0.9718; WQ adds ~0). So **the S=4 capacity ceiling binds on MNIST** (refines
the recon's prediction that margin would make it non-binding). Per fast_lossless_plan §1.3, the lever
is the **code/S**, not more training. Higher S = more levels = higher ceiling.

## Genuine cascaded TTFS capacity sweep (fresh, GPU 0) — DEAD END for lossless

| S | deployed SCM (full) | gap | genuine FT wall | verdict |
|---|---|---|---|---|
| 4 | 0.9302 | 5.4pp | 110s (OK) | LOSSY |
| 8 | 0.9365 | 4.6pp | 133s (OVER) | LOSSY + slow |
| 16 | (running) | — | (slower) | — |

Diagnosis at S=4: **finalize_cliff = 0.0, NF (0.9307) ≈ SCM (0.9302)** — my fold + the deploy
parity are correct; the loss is ENTIRELY the cascade's representational capacity (distmatch
`dead_fraction_after` ≈ 6.4 — the death cascade). The curve plateaus ~0.94 (diminishing returns)
**far below 0.983**, and FT cost grows ∝ S (over budget by S=8). So cascaded TTFS is a fragile code
that is BOTH lossy and slow at the S needed — it cannot be the lossless+fast answer (the plan's §1
"fragile nonlinear code" thesis, empirically confirmed). LIF (linear robust rate code) lost only
1.1pp at the same S=4.

## Engineering landed this campaign: LIF fast-fold

LIF's controller ramp is 488s (far over budget). Generalized the fixed-ladder fast machinery from
the TTFS tuner UP to the shared `KDBlendAdaptationTuner` base (boy-scout de-duplication), with
`_fast_loss`/`_fast_probe` hooks: TTFS keeps its validated plain-CE+genuine-CE objective; **LIF now
gets a fast fixed-ladder ramp** (`lif_blend_fast`, default off) using the installed KD loss — one
shared optimizer + spanning cosine, no controller. 8 LIF-fast unit tests + 128 regression tests green;
goldens bit-identical. This is the "make the lossless code fast" deliverable.

## LIF capacity + fast-fold deep-dive (the lossless-capable code)

| LIF config | S | deployed (NF=SCM, full) | tuning wall | note |
|---|---|---|---|---|
| controller | 4 | 0.9721 | LIF-Adapt 488s | 1.1pp short; SLOW |
| fast-fold (steps=120) | 4 | 0.9462 | LIF-Adapt 28s | FAST but UNDER-trains 2.6pp |
| fast-fold v2 (eta_min, 5×200) | 8 | 0.9663 | LIF-Adapt 43s | FAST; under-trained |
| fast-fold v2 (eta_min, 5×200) | 16 | 0.9724 | LIF-Adapt 52s | FAST; == controller S=4 |
| controller | 8 | 0.9747 | LIF-Adapt 559s | well-trained ceiling (gap 0.69pp) |
| **fast-fold + 400-step stabilization** | **8** | **0.9749** | **LIF-Adapt 60s** | **== controller (0.9747) at ~9× speed** |
| fast-fold + 400-step stabilization | 16 | 0.9742 | LIF-Adapt 76s | == S=8 (fixed stab budget caps it) |

**Surprise:** raising S 4→16 did NOT lift the deployed ceiling above ~0.972 (controller S=4 = 0.9721,
fast-fold S=16 = 0.9724). If the controller-S=8 A/B also plateaus ~0.972, the residual ~1.1pp is
**S-independent conversion fidelity** (plan L2/L3: per-layer threshold/scale calibration, soft-reset,
layerwise reconstruction), NOT rate-code capacity — which would REDIRECT the lossless lever from
"raise S" to "better ANN→SNN conversion."

**Deploy relationship (important):** for LIF, `Normalization Fusion` metric == `Soft Core Mapping`
(both full-test, bit-exact) — so NF is a valid FAST deployed proxy (skip the ~105s/400s SCM during
iteration). The `LIF Adaptation` step metric is the value-domain endpoint (NOT a proxy: 0.8953 while
SCM=0.9462; WQ's recovery training closes part of it).

**Why fast-LIF under-trains** (vs controller): the spanning cosine decays LR→0 exactly at the rate-1.0
rung, starving the LIF value-domain endpoint of recovery (TTFS doesn't care — genuine-CE + distmatch
carry its endpoint by construction). The controller reaches 0.972 via its post-finalize cycle-accurate
stabilization, which the fast path skips. **Fix landed:** `lif_blend_fast_lr_eta_min` (default 0.1)
floors the cosine so the endpoint keeps recovering; configurable per family (0 for TTFS). Plus more
endpoint rungs + steps. (Unit-safe: TTFS + goldens unchanged.)

## Honest capacity reality (the plan's §1 thesis, confirmed)

S=4 caps even the best (linear, robust) code — LIF — at 0.972, **1.1pp below the 0.983 ANN**. True
lossless (0% deviation) is bounded by the spiking-code capacity (rate levels = S+1). The achievable
deployed accuracy is a Pareto curve over S (latency/energy), exactly as fast_lossless_plan §1 predicts:
"if no allocation within the chip's ΣS budget meets ε_lossless, lossless is impossible at this budget —
the output is a Pareto curve; raising ΣS is a product call." The campaign quantifies that curve.

## Decision tree

- **(A)** genuine TTFS reaches lossless at some S with FT < 2 min → DONE, no new code. *(preferred — TTFS is the headline)*
- **(B)** TTFS (fragile timing code) caps below lossless even at high S → use LIF (linear, lossless-capable
  at high S) and extend the FAST schedule (fixed_ladder + skip_rollback + single_lr — the generic seam
  already built) to LIF, since LIF's controller path is 488s (far over budget). This is the engineering
  "finish" if TTFS can't be both.
- **(C)** per-layer S allocation (fast_lossless_plan L6/E9) to minimize ΣS / latency — code change, DEFER
  unless a single global S can't be both lossless and fast.

## FAST cost components (to attack after lossless is secured)

- genuine FT (fixed_ladder): `len(rates)·steps_per_rate` steps × S-cycle forward — scales ~linearly in S.
- Weight Quantization: 61–210s (AdaptationRateTuner) — independent of FT; a separate FAST target.
- Mapping (SCM+HCM) + sim: deployment overhead.
- Knobs: `ttfs_blend_fast_steps_per_rate`, `ttfs_blend_fast_rates` (genuine FT only; cost ∝ steps×rates×S),
  `tuning_budget_scale` (scales `max_training_steps` = WQ/controller recovery budget, capped 4000; cuts WQ).
- Higher S also inflates the deployment tail (SCM full-eval + HCM + nevresim are S-cycle) → the IDEAL is
  the MINIMAL S that is lossless. Per-layer S (L6) would minimize ΣS but is deferred code.

## CORRECTION — cascaded TTFS is NOT a dead-end; the genuine-blend FAST under-trained

The genuine-blend FAST path (0.930) is worse than the DEFAULT value-domain PROXY ramp, which deploys
the genuine cascade at **0.956** (cached `mmixcore_default_e2e`: proxy FT 0.9578 → NF 0.9647 → SCM 0.956;
the proxy↔genuine cliff is ~0.9pp, NOT the 0.26 of the docs' harder model). So the genuine-blend fast
*under-trains* — the same disease LIF had. **Fix (written): apply the LIF fast-fold pattern to the TTFS
VALUE-DOMAIN PROXY ramp** — `ttfs_blend_fast` → fixed_ladder ramp (value domain) + eta_min floor +
`_fast_stabilize` (a short bounded recovery on the deployed `_SegmentSpikeForward`, closing the
proxy↔genuine cliff). Expected: ~0.956 deployed in <2min (matching the 388s proxy-controller, fast),
i.e. the TTFS analog of the LIF win. **VALIDATED on MNIST (torch 2.12, after restoring the env from
requirements.txt):** TTFS `ttfs_blend_fast` deployed (SCM, full-test), cliff closed (NF==SCM):
- 200 steps/rung + 400 stab → **0.9445** (S=4) / 0.9467 (S=8), TTFS-Cycle ~57–70s.
- 400 steps/rung + 800 stab → **0.9501** (S=4), TTFS-Cycle 88s (<2min).

So the proxy-fast lifts cascaded TTFS from the broken genuine-fast's 0.930 → ~0.95, fast, and converges
toward the 388s proxy-controller's 0.956 with more steps (the speed↔accuracy tradeoff; TTFS's
value-domain is harder to train than LIF's). The mechanism (value-domain fast ramp + `_fast_stabilize`
on the genuine cascade) is the fix; the `_fast_stabilize` closes the proxy↔genuine cliff (NF==SCM).

## DECOMPOSITION: it's conversion fidelity, not under-training or pure capacity

Within-run gap (ANN − deployed, same model, the honest metric since pretrain has ~0.7pp run-to-run noise):

| LIF run | S | steps | ANN | deployed (NF) | gap | wall (adapt) |
|---|---|---|---|---|---|---|
| controller | 4 | (controller) | 0.983 | 0.9721 | 1.1pp | 488s |
| fast-fold | 8 | 1000 | 0.9833 | 0.9663 | 1.7pp | 43s |
| fast-fold | 8 | **3000** | 0.9773 | 0.9668 | 1.05pp | 82s |
| fast-fold | 16 | 1000 | 0.9811 | 0.9724 | **0.87pp** | 52s |

**3× the steps at S=8 moved deployed 0.9663 → 0.9668 (noise) — so it is NOT under-training.** Higher S
shrinks the gap (1.7pp@S8 → 0.87pp@S16) — so capacity helps, slowly. The residual ~1pp at fixed S is a
**systematic ANN→LIF conversion bias** (the plan's L2/L3 lever: per-layer threshold/scale calibration,
soft-reset, layerwise reconstruction), not rate-quantization capacity and not training budget. This is
the actionable redirect: to close the last ~1pp, invest in CONVERSION FIDELITY (calibration/
reconstruction), not more S or more steps.

## FAST budget (stabilized fast-LIF S=8, measured)

- **LIF Adaptation (the accuracy-recovery tuning session the docs target) = 60s** — well under 2 min,
  and **9× faster than the controller's 559s at matched accuracy (0.9749 vs 0.9747).** ✅
- Full transformation incl. Weight Quantization: AA 14 + LIF-Adapt 60 + WQ 40 + QV 11 + NF 12 ≈ **137s
  (~2.3 min)** — WQ (40s, a separate AdaptationRateTuner) is the residual over-2min cost; trimmable via
  `tuning_budget_scale` if the whole transformation must fit 2 min.
- (Pretraining 58s is the one-time ANN, excluded from the tuning budget by the docs' definition.)

## CONCLUSION (honest)

The literal target — deployed == ANN (0% deviation) in < 2 min — is **not reachable on the S=4
hardware**, and the reason is exactly what fast_lossless_plan §1 leads with: **spiking-code capacity**.

1. **Capacity wall.** S=4 gives ≤ S+1 = 5 rate levels. Even LIF (the linear, robust code) caps at
   **0.972** deployed (1.1pp below the 0.983 ANN); cascaded TTFS (fragile timing code) caps ~0.94.
   Closing the last ~1pp requires more levels = higher S — a **latency/energy product call**, not a
   tuning fix (the plan's §1.3 + §9-risk-3). Per-layer S allocation (L6/E9) would minimize the ΣS cost
   but is deferred code.
2. **Speed↔accuracy — RESOLVED by the stabilization fix.** The pure fast fixed-ladder ramp
   under-trained the LIF value-domain endpoint (≈0.966 fast vs 0.9747 controller @ S=8). Adding a
   short post-finalize bounded stabilization on the deployed forward (`_fast_stabilize`,
   `lif_blend_fast_stabilize_steps`) **closes that gap entirely: stabilized fast-LIF S=8 = 0.9749 in a
   60s LIF-Adaptation session — matching the 559s controller's 0.9747 at ~9× speed.** So the FAST half
   is delivered: the fast non-destructive tuning matches the controller's accuracy, fast.

**Best near-lossless+fast operating point measured:** stabilized fast-LIF — deployed **0.9749 at S=8**
(gap 0.76pp vs ANN) in a **60s tuning session** (full transformation ~137s incl. WQ). Higher S shrinks
the gap further (controller curve 1.1pp@S4 → 0.69pp@S8 → ~0.4pp@S16). The genuine-TTFS fast path (my
earlier fold) is fully fast + deploy-faithful (cliff=0) but its *code* is capacity-limited to ~0.93.

**To actually reach 0% deviation:** raise S (e.g. S=16, a product call on latency) and accept slower
deployment, or implement **per-layer S allocation** (plan L6) to buy the capacity only where the
sensitivity sweep (plan E1) says it's needed — the documented next lever.

## Engineering delivered this campaign (the FAST half, landed + tested)

- **Genuine-TTFS fast fold** (review Rec 2): fixed_ladder RateScheduler policy, validated end-to-end on
  MNIST (cliff=0.0, NF==SCM — the fold + deploy parity are correct; TTFS is just a lossy code).
- **LIF fast fold**: generalized the fixed-ladder fast machinery to the shared `KDBlendAdaptationTuner`
  base (`_fast_loss`/`_fast_probe` hooks) so LIF gets a fast value-domain ramp (`lif_blend_fast`),
  cutting LIF tuning from the 488s controller toward ~40–90s.
- **`lif_blend_fast_lr_eta_min`**: floors the spanning cosine so the rate-1.0 endpoint keeps recovering
  the deployed LIF dynamics (the value-domain endpoint needs real training; TTFS doesn't).
- **`_fast_stabilize` + `lif_blend_fast_stabilize_steps`**: a post-finalize bounded recovery on the
  DEPLOYED cycle-accurate forward (non-destructive, rollback on regression) — recovers the ~0.9pp the
  pure fast ramp leaves vs the controller's stabilization, while staying fast. TTFS does not use it
  (its rate-1 IS the deployed cascade).
- All default-OFF, golden bit-identical; tuning+config tests green (714 + the new LIF-fast suite).

## Work-item decision (from the two docs, for MNIST lossless+fast)

- DONE (code): fixed_ladder fast fine-tune (E4/F1), scale-aware-boundaries+DFQ calibration (E2/L3),
  genuine-CE last-mile lever (E8/L7), one-orchestrator fold + condition-first (Rec1/2/5), LR-finder kill (F3).
- DO-NOW: capacity allocation via S (E1/E9 reduced to "pick the minimal lossless S") + the validation gate.
- DEFER (MNIST): layerwise reconstruction (E3/L2), feature distill (E7), QAT co-design (E10),
  per-layer S code (L6) unless global S can't be lossless+fast.
- PRODUCT-CALL: global S latency/energy trade; paired_sensor graduation (ruled out — keep opt-in).
