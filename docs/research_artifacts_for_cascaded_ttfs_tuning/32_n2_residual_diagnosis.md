# Phase 3 — N2 (bias-only FT) vs full FT, and a decomposition of the genuine → ideal-staircase residual

Prototype: `experiments/n2_deep.py` (imports `cascade_lab`/`novel`/`closed_form`
infra; edits nothing). All numbers float64, CPU, deterministic, seeds {0,1,2},
same continuous base + calibration per cell so the **only** difference is the FT
constraint. The ideal staircase (`set_cycle_accurate(False)`, linear `(T−τ)/T`
decode, no ramp) is the recoverable ceiling throughout.

---

## TL;DR

1. **Q1: N2 (bias-only FT) does NOT beat full genuine FT on the toy — it loses, often
   badly.** In a controlled apples-to-apples comparison (same base/calib/epochs/lr),
   full-FT beats N2 in **every** cell (N2−full = −0.10 to **−0.64**). N2 reaches its
   *frozen-weight conversion ceiling*; full-FT *retrains the task through the genuine
   cascade* and **exceeds even the ideal staircase and the continuous teacher** (125–
   217% of staircase). The artifact-16 "N2 → 99% of continuous" was N2 hitting *its
   own underfit* continuous ceiling, not out-performing full FT.
2. **WHY:** N2 genuinely learns the **phase-advance** (positive `bias_norm` deltas
   `[0.35,0.17,0.09,0.06]`, decaying with depth — pulling deep neurons into the
   faithful early window, mechanism confirmed). But freezing weight *directions* caps
   it at the conversion ceiling. full-FT does NOT phase-advance (tiny bias deltas
   ~0.02); it rewrites the weights wholesale (**2.5–3.9× relative weight-norm drift**)
   into the cascade's own basin. The bias-only constraint is a *cleaner, more
   interpretable, more deployable* correction — but a strictly *weaker* one.
3. **Q2 residual decomposition** (genuine-FT → ideal-staircase):
   - **(a) surrogate-gradient noise: NOT the bottleneck.** N2 with the default
     ATan `alpha=2.0` already meets/exceeds its staircase (resid ≈ 0); wider/annealed
     surrogate or 3× steps move it by ≤ +0.013, and *too* wide (`alpha=0.5`) **hurts
     −0.28**. alpha is already well-tuned.
   - **(b) the per-sample (T−τ)² gain: this IS the residual.** Per-neuron value error
     decomposes as **~78% per-depth gain, ~83% per-neuron gain, ~17% per-SAMPLE
     residual** — the per-sample part is exactly the nonlinearity *no static encode
     can invert*. An oracle per-sample correction of the last hidden feature buys
     **+0.08 (depth-3) to +0.39 (depth-4)** accuracy; per-depth/per-neuron static
     gains buy **~0.00** (absorbed by readout scale-invariance).
   - **(c) generalization: not the problem.** full-FT's gen-gap is only +0.045–0.050;
     its test win over N2 (0.954 vs 0.801, 0.944 vs 0.310) is real, not memorization.
4. **Verdict: N2 is NOT worth a real-pipeline phase.** The mmixcore already runs the
   teacher-blend genuine ramp = **full genuine FT**, which the toy shows is *strictly
   stronger* than bias-only. N2's only unique regime (deep, past `d_max`, where full
   FT diverges *cold*) does not arise on the mmixcore because the teacher-blend
   curriculum keeps full FT well-conditioned. The residual to LIF is the per-sample
   (T−τ)² ramp gain, which is **encode-fixed** and only an *encode/decode redesign*
   (out of scope) — or simply the synchronized schedule — can fully close.

---

## Q1 — controlled {full-FT, N2, G+full-FT, G+N2} vs ideal staircase

Each row: same continuous base + calibration; four FT variants (40 epochs, lr 2e-2,
ATan alpha 2.0); `G` = the deployed geometric gain correction (`ρ₀·γ^d`).

| d | S | cont | stair | base | **full-FT** | **N2** | G+full | G+N2 | N2−full | %stair(best) |
|--:|--:|-----:|------:|-----:|------------:|-------:|-------:|-----:|--------:|-------------:|
| 3 | 8 | 0.783 | 0.764 | 0.077 | **0.957** | 0.800 | 0.961 | 0.824 | **−0.157** | 125.7% |
| 3 | 16 | 0.783 | 0.785 | 0.260 | **0.965** | 0.861 | 0.967 | 0.881 | −0.104 | 123.2% |
| 4 | 8 | 0.456 | 0.440 | 0.077 | **0.942** | 0.307 | 0.954 | 0.493 | **−0.635** | 216.6% |
| 4 | 16 | 0.456 | 0.455 | 0.077 | **0.952** | 0.563 | 0.958 | 0.623 | −0.390 | 210.5% |

Readings (these **revise** the artifact-16 framing):

- **full-FT strictly dominates N2 in all 4 cells.** The toy's continuous teacher
  *underfits* (depth-4 only 0.456); full-FT exceeds it because it is no longer doing
  a conversion correction — it is **retraining the task in the cascade's basin**, and
  the cascade's quadratic-ramp basin is just a different (perfectly trainable)
  nonlinearity. This is why it blows past the staircase (125–217%): the "staircase"
  bounds *conversion of a fixed teacher*, not *training from scratch through the ramp*.
- **G helps full-FT a hair and N2 a lot** (G+N2 > N2 everywhere: e.g. d4 S8 0.307 →
  0.493) because G revives the cold cascade so the frozen-weight bias FT has a live
  gradient. But **G+N2 still loses to plain full-FT** in every cell.
- The earlier artifact-16 table that showed "N2 0.777 vs full-FT 0.917" and called
  N2 "promising" used a *different* full-FT recipe / init and N2-from-trim-init; the
  controlled re-run here (identical infra, identical budget) reverses the verdict.

### Why N2 loses — the mechanism (depth-4 S=8, 3 seeds)

| mode | test | train | gen-gap | mean `bias_norm` Δ / depth | rel weight drift / depth |
|------|-----:|------:|--------:|----------------------------|--------------------------|
| **N2** (bias) | 0.291 | 0.351 | +0.060 | `[0.349, 0.171, 0.092, 0.062]` | 0 (frozen) |
| **full** | 0.952 | 0.992 | +0.040 | `[0.025, 0.031, 0.012, 0.010]` | `[2.46, 3.16, 3.25, 3.92]` |

- **N2 IS a genuine phase-advance** — every layer learns a *positive* `bias_norm`
  delta, decaying with depth, exactly the per-cycle membrane lift that pulls fire
  times earlier into the faithful window. The mechanism is real and interpretable.
- **N2 is capped by the frozen-weight conversion ceiling** (depth-4 cont = 0.456),
  so test = 0.291 even with a perfect phase-advance.
- **full-FT does NOT phase-advance** (bias deltas ~0.02 ≈ noise); it **rewrites the
  weights** (2.5–3.9× their norm) into the cascade basin and learns the task afresh.
  The "freeze directions to stay well-conditioned" premise of N2 is exactly what
  *prevents* it from reaching full-FT's accuracy — well-conditioned but ceiling-bound.

**Q1 answer:** No. The bias-only constraint finds the *phase-advance* solution (which
full FT does not bother to find, because it has the freedom to retrain weights), but
that solution is a strictly *weaker* one — it reaches the conversion ceiling, while
full FT reaches the (much higher) trained-through-the-cascade accuracy.

---

## Q2 — decomposing the genuine-FT → ideal-staircase residual

### (a) Surrogate-gradient noise — NOT the dominant cause (depth-3 S=8, N2)

Ideal staircase target = 0.764. N2 with various surrogate settings:

| surrogate setting | acc | resid → stair |
|-------------------|----:|--------------:|
| alpha=2.0, 40 ep (default) | 0.796 | **−0.032** (N2 *exceeds* staircase) |
| alpha=1.0 (wider) | 0.782 | −0.017 |
| alpha=0.5 (much wider) | 0.518 | **+0.247** (collapses) |
| alpha=4.0 (sharper) | 0.797 | −0.032 |
| alpha anneal 0.5→4.0 | 0.793 | −0.028 |
| alpha=2.0, 120 ep | 0.802 | −0.038 |
| alpha=2.0, 120 ep, lr 5e-3 | 0.809 | −0.045 |
| anneal 0.5→4.0, 120 ep | 0.803 | −0.039 |

- N2 at the default `alpha=2.0` **already meets/exceeds** the staircase — there is
  essentially **no surrogate-noise residual to close** for the alive shallow cascade.
- Going **wider** (`alpha=0.5`) **hurts badly** (−0.28): an over-soft surrogate
  rewards pushing values up (fire-earlier / outside-window) — the same anti-aligned
  gradient that killed the Phase-0 "analytical timing proxy". `alpha=2.0` is tuned.
- More steps + smaller LR buys a marginal **+0.013** (0.796 → 0.809). Honest, real,
  but third-order.
- **(a) contribution: ≤ +0.013 (≈ 0).** Surrogate-gradient noise is not the bottleneck.

### (b) The per-sample (T−τ)² gain — THIS is the residual

Each input drives every neuron to a *different* fire-time τ, so the ramp applies a
per-sample gain `g_eff(τ) ∝ (T−τ+1)`. A static per-depth / per-neuron θ cannot invert
a per-sample-varying gain. Variance decomposition of the per-neuron genuine-vs-teacher
decoded-value error (non-readout layers):

| cell | per-DEPTH gain | per-NEURON gain | **per-SAMPLE residual** |
|------|---------------:|----------------:|------------------------:|
| d3 S8 (3 seeds) | ~77.8% | ~81.3% | **~18.7%** |
| d4 S8 (3 seeds) | ~78.5% | ~85.1% | **~14.9%** |

- A per-NEURON gain buys only **+3–7 pp of variance** over per-DEPTH — confirming the
  death cascade is *dominated* by a per-depth term (which the deployed G already
  targets), with a small per-neuron refinement.
- The **~15–19% per-sample residual is irreducible to any static encode** — it is the
  (T−τ)² nonlinearity itself.

**Oracle static correction → last-layer accuracy** (apply the best per-depth /
per-neuron / per-sample gain to the last hidden feature, re-run the readout):

| tier | d3 S8 acc | Δ | d4 S8 acc | Δ |
|------|----------:|--:|----------:|--:|
| base (genuine) | 0.706 | — | 0.092 | — |
| per_depth oracle | 0.700 | −0.006 | 0.092 | +0.000 |
| per_neuron oracle | 0.699 | −0.007 | 0.092 | +0.000 |
| **per_sample oracle** (= staircase feature) | **0.790** | **+0.084** | **0.484** | **+0.391** |

- **per-depth / per-neuron static gains buy ~0.00 accuracy** — a scalar rescale of the
  last hidden feature is **absorbed by the readout weights** (argmax is scale-tolerant).
  This is *why* the deployed per-layer θ trim (G) gives only the modest mmixcore +0.5pp:
  static gain mostly cancels at the readout; its real value is reviving *cold/dead*
  layers so a downstream FT has signal, not adding accuracy directly.
- **The per-sample oracle is worth +0.08 (depth-3) to +0.39 (depth-4)** — and it grows
  with depth, exactly tracking the compounding (T−τ)². This is the **whole recoverable
  residual**, and it lives in the per-sample gain. N2's phase-advance captures *part*
  of it dynamically (its test ≈ the per-sample oracle at depth-3: 0.80 vs 0.79).
- **(b) contribution: the entire recoverable residual (+0.08 to +0.39)**, and it is
  **encode-fixed**: only a per-sample-aware encode/decode (out of scope — the decode is
  bit-exact-locked) could realize it. Static per-neuron biases (N2) only approximate it.

### (c) Generalization — not the cause

| d | S | method | train | test | gen-gap |
|--:|--:|--------|------:|-----:|--------:|
| 3 | 8 | full | 0.999 | 0.954 | +0.045 |
| 3 | 8 | N2 | 0.829 | 0.801 | +0.028 |
| 4 | 8 | full | 0.994 | 0.944 | +0.050 |
| 4 | 8 | N2 | 0.380 | 0.310 | +0.070 |

full-FT's gen-gap (+0.045–0.050) is small and comparable to N2's; its test-set win is
**real**, not overfitting. N2's depth-4 collapse (test 0.310) is a *ceiling* effect
(frozen underfit weights), not a generalization failure. **(c) contribution: ~0.**

---

## Which lever closes the residual, and by how much

| lever | what it is | closes (of genuine→staircase) | deployable? |
|-------|-----------|-------------------------------|-------------|
| wider/annealed surrogate, more steps | (a) | **≤ +0.013** (≈ 0) | yes (training knob) — but no headroom |
| static per-depth θ (= deployed G) | (b)-depth | **~0** at the readout (revival only) | yes — but argmax-absorbed |
| static per-neuron θ/bias (≈ N2 cold) | (b)-neuron | **~0** extra at the readout | yes — but argmax-absorbed |
| **N2 trained per-neuron phase-advance** | (b) dynamic | up to the *frozen-weight conversion ceiling* (≈ the per-sample oracle at shallow depth: +0.08); **far below full-FT** | yes (bias is a chip param) |
| **full genuine FT (= the mmixcore teacher-blend)** | retrain in the basin | **+0.20 to +0.85** — exceeds the staircase | yes (already deployed) |
| per-SAMPLE oracle correction | (b) irreducible | +0.08 (d3) … +0.39 (d4) | **NO** — needs an encode/decode redesign |

**The single dominant lever is full genuine FT** (retrain through the cascade), which
the real pipeline already runs as the teacher-blend ramp. The per-sample (T−τ)² gain is
the residual's *root*, but it is encode-fixed; static corrections cancel at the readout
and N2's bias-only approximation is strictly weaker than full FT.

---

## Is N2 worth a real-pipeline phase? — NO

- **The mmixcore teacher-blend genuine ramp is full genuine FT**, which on the toy
  *strictly dominates* bias-only N2 in every cell (by +0.10 to +0.64). Adding a
  bias-only phase can only *under-fit* relative to what the pipeline already does.
- **N2's one unique regime** (depth past `d_max`, where full FT *diverges cold*) does
  **not arise** on the mmixcore: the teacher-blend is a teacher→genuine curriculum
  that keeps full FT well-conditioned even on the 9-deep cascade (artifact 30 confirms
  it already reaches ~0.93). The premise that "freezing weights keeps the optimization
  well-conditioned where full FT diverges" is moot when the curriculum prevents the
  divergence.
- **The residual to LIF (~3.5pp) is the per-sample (T−τ)² ramp gain** (Q2b), which is
  encode-fixed. No bias-only or static lever closes it; only (i) an encode/decode
  redesign — explicitly out of scope, the decode is bit-exact-locked — or (ii) the
  already-shipping **synchronized** schedule (LIF-level ≥ 0.97) closes it.

**Recommendation:** do **not** add an N2 bias-only-FT pipeline phase. Keep the deployed
recipe (teacher-blend full genuine FT + optional `ttfs_gain_correction` G to revive
deep/cold layers so the FT has gradient). N2 stays a *diagnostic* result: it isolates
and confirms the phase-advance mechanism cleanly, but it is a weaker, ceiling-bound
form of the full FT the pipeline already performs.

---

## Honest scope / caveats

- The toy continuous teacher *underfits* (depth-4 cont 0.456); this is what lets
  full-FT exceed the staircase. The transferable claim is the **ordering** (full-FT >
  N2, robustly across seeds/depth/S) and the **residual decomposition** (per-sample
  gain dominant, surrogate/generalization negligible), not the absolute numbers.
- The oracle per-sample correction is an *upper bound* applied only at the last hidden
  layer (the readout decides accuracy); it bounds (b)'s contribution, it is not a
  deployable mechanism (the decode is bit-exact-locked).
- The per-depth/per-neuron "~0 at the readout" result is specific to the *final*
  feature→argmax path; static θ still matters for **reviving dead deep layers** so a
  downstream FT has signal (artifact 21/25/30) — that is G's real, kept role.
