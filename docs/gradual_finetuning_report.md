# Gradual Fine-Tuning of Genuine Cascaded-TTFS Models — Engineering & Research Report

**Status:** Mechanism solved and landed (opt-in, default-off, golden-safe). Performance tuning in progress.
**Scope:** Why genuine single-spike cascaded `ttfs_cycle_based` deployment could not be *gradually* fine-tuned,
the root cause, the recipe that fixed it, and the flow-performance work that followed.
**Validation model:** MNIST `mlp_mixer_core`, `simulation_steps = T = 4` (a deliberately coarse 2-bit-time grid),
teacher (continuous ANN) accuracy ≈ 0.967.

---

## 1. Executive summary

The genuine cascaded-TTFS forward (single-spike, timing-weighted ramp integration) could **not** be gradually
fine-tuned: the genuine-throughout annealed ramp never committed a single rate (0/8 cycles, capped ~0.75 only via
post-finalize stabilization). We traced this to a **"death cascade"** on cold (ANN) weights and showed the fix is a
**distribution-matching calibration** plus a **teacher→genuine blend ramp**. The genuine cascade now fine-tunes
gradually to **~0.94 deployed**, cliff-free, committing rate after rate — the thing that was impossible before.

One-line recipe:

> **scale-aware [0,1] boundaries  +  DFQ activation-distribution matching  +  teacher→genuine blend ramp  +  fine-tune along it.**

A separate finding: the per-step cost is *not* the bottleneck (a genuine cascade train step is **26 ms**; 600 steps =
16 s). The slowness is the general-purpose adaptation controller spending ~80× more steps than needed. A fast
fixed-increment path reaches ~0.94 in ~30–60 s; the general controller is being tuned to close the gap.

---

## 2. The problem

`ttfs_cycle_based` (cascaded) deploys each neural segment as a **single-spike, ramp-integrate, fire-once** cascade.
The prior fine-tuning approaches:

- **Value-domain proxy ramp** (default): trains a pointwise staircase proxy fast, installs the genuine cascade only at
  *finalize* → a large **finalize cliff** (~0.51 here) recovered by a long post-finalize stabilization tail.
- **Genuine-throughout annealed ramp** (`ttfs_genuine_annealed_ramp`, added this session): installs the genuine cascade
  for the whole ramp and anneals the spike-surrogate sharpness. It is cliff-free **by construction** but **never
  committed** — every cycle's recovery plateaued below the pipeline floor (0.803) and rolled back.

The genuine cascade's **cold-start** accuracy (untrained, ReLU weights) is ~0.10–0.20 — far below the teacher's 0.967 —
so the gradual ramp had no foothold to climb from.

---

## 3. Root cause: the death cascade (and why LIF never had it)

### 3.1 LIF is a linear rate code; TTFS is a nonlinear timing code
- **LIF** (`LIFActivation`, `LifSegmentPolicy`): `out ≈ clamp(x/θ, 0, 1)·θ = clamp(x, 0, θ)`. The *only* distortion is
  the [0,1] clamp, so a single scale θ = activation range is sufficient — **no activation-scale drift**. This is why
  LIF only ever needed scale-aware [0,1] boundaries.
- **TTFS** (`TTFSActivation` cycle-accurate + `TtfsSegmentPolicy`): a single spike's *timing* encodes the value, and the
  cross-layer membrane integration is **timing-weighted** (early spikes integrate longer). This is genuinely
  **nonlinear**, so a per-layer scalar cannot match it.

### 3.2 Per-layer distribution decomposition (the smoking gun)
Measuring `mean|activation|` per layer, continuous-ANN vs analytical-staircase vs genuine-cascade (cold weights):

| metric | ANN→staircase | staircase→cascade |
|---|---|---|
| ratio | **0.08–0.28** (4–12× attenuation) | ~0.78–1.64 (≈1×) |

The dominant drift is the **non-ANN-aware [0,1] clamp** (the ttfs_cycle path runs at `activation_scale=1.0`, never
applying the Activation-Analysis scales). On top of that, the cascade **progressively kills neurons** with depth:

```
boundary q99, cold weights:  %dead/layer = [83, 84, 92, 95, 98, 98, 99, 99, 100]
```

By layer 6 **100% of neurons are dead** (no fire → no gradient). Cause: θ_out grounded in the ANN's 99th percentile is
far too high for the cascade's firing threshold (`fires iff w·in_scale ≥ θ/T`), so only the top ~16% fire; the median
dies, and dead inputs starve the next layer → death compounds. `R²(cascade ≈ a·ann + b)` is low/negative → the gap is
**not** a fixable scalar scale.

### 3.3 Grid coarseness is real but secondary
Sweeping S (grid): the *staircase* recovers to ~0.88 at S=32, but the *cascade* stays ~0.30 even at S=32 — the
timing-weighted integration gap does **not** close with finer grid or with scalar scaling. It closes only with
**training** (the deployed cascade reaches ≈0.93 on adapted weights; NF↔SCM parity is exactly 1.0 there).

---

## 4. The solution recipe (what worked)

### 4.1 Scale-aware [0,1] boundaries
A spike train of length T encodes values in [0,1]. Every perceptron block must therefore have **scale-aware
boundaries**:
- `θ_out[l]` (`activation_scale`) — activation-distribution-grounded scale normalizing the **output** to [0,1].
- `input_scale[l] = θ_out[upstream(l)]` (`input_activation_scale`) — un-normalizes the [0,1] spike **input** back to the
  value domain. The cascade is the *only* forward that consumes `input_scale`; it was stuck at 1.0.

Empirically necessary: with proper θ_out the staircase hits 0.96, but **without** the input-side boundary the cascade
**collapses** (0.10); restoring `input_scale = θ_out[upstream]` recovers it (3.2×). **Landed:**
`src/mimarsinan/spiking/scale_aware_boundaries.py` (flag `ttfs_scale_aware_boundaries`).

### 4.2 DFQ activation-distribution matching (the key piece)
Boundaries fix the *scale* but not the *death cascade*. **Per-neuron bias correction** (Data-Free-Quantization style)
adjusts each perceptron bias so the cascade's per-neuron mean matches the ANN's — a deployable transform that **revives
starved neurons** (raising the membrane baseline so they fire) and matches the first moment:

```
                              cold cascade acc   mean|gap|   %dead
boundary q99 (pre-correction)      0.106          0.231     [83..100]   ← death cascade
+ DFQ bias correction              0.412          0.045     [55..71]    ← revived + matched
```

It also **stabilizes** the fine-tuning (controlled-LR: bias-corrected recovery holds 0.65–0.75 where boundary-only
collapses to chance). **Landed:** `src/mimarsinan/spiking/distribution_matching.py`
(`match_activation_distributions` = scale-aware boundary + DFQ bias correction). This generalizes the LIF
scale-aware-boundary idea to TTFS via weight/bias/scale stats transforms — generic, deployable, no model-specific hacks.

### 4.3 Teacher→genuine blend ramp (the correct ramp shape)
The genuine-*throughout* ramp starts at the r=1 endpoint → no foothold. The correct shape is the **output blend**
`(1−r)·teacher + r·genuine`, which reads ≈ teacher (0.967) at r→0 and degrades only at r=1. The previously-*rejected*
`genuine_gradual_cascade_ramp` had this shape but was rejected because — **without** distribution matching — its r=1 end
was a catastrophic cliff. With distribution matching, the curve is **smooth and recoverable**:

```
TEACHER→GENUINE BLEND CURVE (teacher 0.967):
mode         r=0.01  r=0.5  r=0.75  r=0.9  r=1.0
scale=1.0    0.967   0.973  0.969   0.702  0.196   ← cliff
boundary     0.967   0.967  0.968   0.968  0.106   ← sharp cliff
bias (DFQ)   0.967   0.967  0.688   0.505  0.412   ← SMOOTH, rampable
```

**Landed:** `src/mimarsinan/models/spiking/training/blended_genuine_forward.py` (`BlendedGenuineForward`; rate=0 ≡
teacher exactly, rate=1 ≡ pure genuine cascade exactly). Wired into `TTFSCycleAdaptationTuner` behind
`ttfs_genuine_blend_ramp`.

### 4.4 Fine-tune along the ramp → the mechanism works
Training at an intermediate blend rate **pulls the full-transform (r=1) genuine accuracy up**:

```
fine-tune at r=0.75:  full-transform (r=1 genuine)  0.414 → 0.898   (Δ +0.48)
full ramp [0.5,0.75,0.9,0.97,1.0]: 0.414 → 0.8613 → 0.9062 → 0.9189 → 0.9277 → 0.9355
```

And end-to-end through the **real pipeline tuner** (`ttfs_genuine_blend_ramp` on):

```
Cycle summary: COMMITS 9/10 (was 0/8). committed rate 0.125→1.0.
finalize_cliff = 0.0 ; genuine-drop CONVERGING 0.187 → 0.011 ; FINAL deployed = 0.9419
```

---

## 5. The de-fooled diagnostic (also landed)
The original `tuning_full_transform_probe` measured the **value-domain** rate-1.0 accuracy, which converges while the
**genuine** cliff stays large — it was *fooled*. We made it measure the **genuine** full transform per-commit (on a
non-destructive clone via the shared `_finalize_forward_for(model)` builder, so probe ≡ deploy), keyed the
CONVERGING/FLAT trend on the **genuine** drop, and surfaced `proxy_gap` (value↔genuine divergence) as the "was fooled"
evidence. **Landed:** `tuning/orchestration/genuine_probe.py` + the probe rework (flag `tuning_full_transform_probe`).
This is what let us *see* convergence end-to-end.

---

## 6. Performance analysis (why it's slow — and it's not the cascade)

Per-op cost profile (T=4, batch 128, GPU):

```
value-domain forward      :   2.16 ms
genuine cascade forward   :   9.75 ms   (4.5× value)
value-domain TRAIN step   :   7.35 ms
genuine cascade TRAIN step:  25.95 ms   ← 600 steps = 16 s
cascade validation (16 bt): 157.18 ms
genuine probe (per commit):  13.68 ms
```

**The cascade step is cheap.** The prototype reached 0.9355 in ~600 steps = 16 s of compute. Yet the pipeline took
**1248 s** (327 gradual + 634 after_run + 287 stabilization) — **~80× more work for +0.01 accuracy.** The
general-purpose `SmoothAdaptation` controller is over-engineered for this now-well-conditioned problem:
`train_steps_until_target` over-trains (~500 steps/cycle vs 120), stabilization is open-ended, per-cycle LR finding (8
probes) is heavy, and a mis-tuned rollback ratchet stalled the ramp (a 634 s force-climb).

---

## 7. Flow-performance work (landed, all opt-in/default-off)

| improvement | flag(s) | what it does | result |
|---|---|---|---|
| LR re-find on missed target | `tuning_refind_lr_on_miss` | re-discover LR only when a cycle misses target (was cached at cycle-0's easy low-rate forever) | LR adapts 0.0112→0.00058 as rate climbs; **stops the 0.976→0.879 drift** (held ~0.95) |
| plateau LR reduction | `tuning_recovery_lr_plateau` | drop LR ×0.3 on plateau instead of breaking | escapes within-cycle plateaus |
| non-stalling rollback ratchet | `tuning_rollback_ratchet`, `tuning_rollback_cumulative_bound` | gate on `max(pre−margin, best−cumbound, floor)` — per-step relative (no stall) + cumulative cap (no accumulation), tightens on new highs | **fixes both** drift *and* the earlier best-anchor stall |
| bounded, cosine-scheduled stabilization | `tuning_stabilization_bounded`, `tuning_stabilization_ratio` | ≤ 50% of gradual steps, cosine LR over a known budget, hard cutoff | caps the 287–439 s tail |
| tighter plateau detection | `tuning_tight_plateau`, `tuning_recovery_check_divisor` | validate more often (validation is cheap) → detect plateau sooner | less over-training |
| recipe-honoring recovery | `tuning_recipe_recovery` | route `tuning_recipe` (AdamW/SGD, weight_decay, momentum/betas, warmup+cosine) into the **step** recovery (it hardcoded Adam + constant LR, ignoring the recipe) | proper per-cycle schedule; generic |
| experimental fast fixed-increment ramp | `ttfs_genuine_blend_fast` | the ~30–60 s prototype: fixed rate schedule × fixed steps, one warmup+cosine LR, no recovery-to-target / rollback / stabilization / per-cycle LR find | ~0.94 in ~30–60 s |

**Diagnostic correction:** the `target_adjuster` was *not* the cause of the committed-accuracy drift — it held the
target at 0.976 for 9/10 cycles. The drift was the **recovery under-performing on a stale LR**; `refind_lr_on_miss`
fixed it.

---

## 8. Results summary

| configuration | committed | finalize_cliff | final genuine deployed | wall |
|---|---|---|---|---|
| genuine annealed ramp (no calib) | 0/8 (stalls) | 0.0 | 0.281 | — |
| + scale-aware boundary (q99) | — | — | 0.751 | — |
| **blend ramp + distribution matching** (real pipeline) | **9/10 commit** | **0.0** | **0.9419** | ~10 min |
| + recovery fixes (LR-on-miss, ratchet, …) | held ~0.95, no drift | 0.0 | **0.9535** | ~21 min (ratchet stalled) |
| full-ramp prototype (fixed increments) | — | 0.0 | 0.9355 | **~30–60 s** |
| **`ttfs_genuine_blend_fast`** (real pipeline) | — | 0.0 | **0.9229** | **26 s** (fast_blend phase) |
| value-domain proxy ramp (baseline) | 0→1 | 0.51 | 0.961 | ~10 min (stabilization-manufactured) |
| teacher (continuous ANN) | — | — | 0.967 | — |

The blend ramp's 0.94 is architecturally **cleaner** than the proxy ramp's 0.96 (no cliff, genuine throughout, gradual
commits) — the proxy's accuracy is largely manufactured by stabilizing away a 0.51 cliff.

---

## 9. Open items / next steps

1. **Break 10 min on the general controller** — the flow fixes removed the stall + capped stabilization, but per-cycle
   recovery (~30–45 s) is still the driver. Levers: cheaper LR re-find (3–4 probes), `tuning_recipe_recovery` impact
   benchmark (cosine should cut steps/cycle), coarser ladder / fewer bisection rollbacks.
2. **Tune the ratchet `cumulative_bound`** generically (currently 0.05) — too tight stalls, too loose drifts.
3. **Push 0.94 → 0.96** — more total budget and/or finer high-rate steps; the curve is still climbing at finalize.
4. **Full-pipeline parity** — run SCM/HCM/nevresim with the blend ramp to confirm deployed NF↔SCM parity end-to-end
   (the finalized forward is the pure genuine cascade, so it should hold).
5. **Generalize the calibration target** — DFQ bias correction matches the mean; mean-only over-corrects toward
   saturation at extreme quantiles. Consider matching the 2nd moment / nonzero-fraction (the variance-match step).
6. **Verify the cross-layer scale-equalization step (3)** — not yet implemented; only boundary + bias correction are.

---

## 10. Reference: config flags (all default-off / neutral)

```
# calibration
ttfs_scale_aware_boundaries, ttfs_distmatch_{quantile,bias_iters,bias_eta}
# ramp shapes
ttfs_genuine_annealed_ramp, ttfs_genuine_blend_ramp, ttfs_genuine_blend_fast
ttfs_ramp_alpha_{min,max}, ttfs_blend_fast_{steps_per_rate,rates}
# diagnostic
tuning_full_transform_probe
# recovery quality
tuning_refind_lr_on_miss, tuning_recovery_lr_plateau,
tuning_rollback_ratchet (+ tuning_rollback_cumulative_bound),
tuning_stabilization_bounded (+ tuning_stabilization_ratio),
tuning_tight_plateau (+ tuning_recovery_check_divisor),
tuning_recipe_recovery
```

### Key source files
- `src/mimarsinan/spiking/scale_aware_boundaries.py`
- `src/mimarsinan/spiking/distribution_matching.py`
- `src/mimarsinan/models/spiking/training/blended_genuine_forward.py`
- `src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py`
- `src/mimarsinan/tuning/orchestration/genuine_probe.py`, `smooth_adaptation_cycle.py`, `smooth_adaptation_run.py`
- `src/mimarsinan/model_training/basic_trainer.py`, `basic_trainer_steps.py`

### Validated prototypes (under `generated/_genuine_ab/`)
`drift_decompose.py`, `dist_analysis.py`, `bias_correct.py`, `blend_curve.py`, `verify_finetune_075.py`,
`full_ramp.py`, `profile_finetune.py` — the numerical experiments behind every figure above.
