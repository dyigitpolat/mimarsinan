# Phase C — STE-fast integration + empirical findings (cascaded TTFS, mmixcore)

Goal: ≥96%@S=4, monotone→lossless by S≤32, FT ≤5 min, on the cascaded TTFS mmixcore.
All runs: SANA-FE off (dodging the known SIGFPE), nevresim sim, 1000 sim samples,
ANN ≈ 0.982.

## What landed (code)

- **STE-fast path** (`ttfs_staircase_ste_fast`, +`ttfs_ste_steps`/`ttfs_ste_w_lr`/
  `ttfs_ste_theta_lr`/`ttfs_ste_init_frac`): routes the staircase STE through a
  dedicated clean fixed-step loop (`_run_staircase_ste_fast`) instead of the
  rate-search controller — split-LR Adam (weights@w_lr, per-channel θ@theta_lr),
  progressive shallow→deep unfreeze, cosine LR, rate(surrogate)-anneal 0→1.
  Unit-tested (13), flag-off byte-identical. `_unify_model_device` SCM fix verified.

## The numbers (FT wall, deployed)

| config | FT wall | deployed | note |
|---|---|---|---|
| controller, improvements OFF (the 9-run baseline) | ~24 min | 0.96 (500-sample) | slow; the old path |
| **proxy_fast**, improvements off | **49 s** | **0.92** | fast, but proxy↔genuine cliff (stabilize_steps=0) |
| **proxy_fast + stabilize=800** (revive→KD-refine) | **67 s** | **0.9531** | closing the cliff: +3.3pp over no-stab, FT well under 5 min — the two-stage direction, BEST |
| proxy_fast + STE-refine(800) + theta | 83 s | — | FT 0.9369 (< KD-refine), then WQ crash (theta×WQ shape bug) |
| STE-fast (progressive depth, θ) | 37 s | **0.10** | chance |
| STE-fast + gain-correction revival | 37 s | **0.10** | chance |
| STE-fast + GC, init_frac=1.0 (all-depth) | 37 s | **0.10** | chance |
| STE-fast + GC, S=16 | 35 s | **0.10** | chance |
| STE-fast + GC, S=16, rate-anneal 0→1 | 35 s | **0.10** | chance |

## Diagnosis (traced, S=16, `MIMARSINAN_STE_TRACE=1`)

```
step=0    depth=3 rate=0.001 lr=2.0e-3 acc=0.0909   <- cold genuine cascade = chance
step=150  depth=4 rate=0.126 lr=2.0e-3 acc=0.1052
step=300  depth=5 rate=0.251 lr=1.7e-3 acc=0.1052
...
step=1050 depth=8 rate=0.876 lr=5.2e-4 acc=0.1052   <- FLAT at chance throughout
```

**Not divergence — it never leaves chance.** The cold genuine single-spike cascade on
mmixcore is DEAD (0.09), and direct genuine-cascade STE training does not revive it,
even with: gain-correction revival, all-depth (no freeze), and surrogate rate-anneal.

This matches the prior research verdict ([[cascaded_ttfs_correctable_gain]]:
"gradient θ-calib FAILS — dead-neuron no-gradient; fix = data-grounded revive→FT").
A fully-dead cascade gives no useful gradient; the staircase-backward hedge alone does
not bridge a >0.85 gap from a dead start.

**The proxy path works (0.92) precisely because it does NOT start from the dead cascade**
— it trains the alive value domain (ReLU/teacher) and ramps into the cascade. The STE
skips that revival.

## Implication for the recipe

The STE is a *refinement* lever, not a *revival* lever. The deployable cascaded recipe is
two-stage: **(1) value-domain revive** (proxy ramp → ~0.92, alive cascade) **→ (2) STE
fine-tune** to push past the proxy↔genuine cliff toward the staircase/LIF ceiling. The
current STE-fast applies stage 2 from a dead cold start (no stage 1) → chance.

**Control settled it: the STE-fast loop is NOT buggy.** The slow controller-STE
(`ttfs_staircase_ste_fast=False`, S=16) commits **0.1010 at every rate 0.5→1.0** — chance,
identical to the fast loop. So direct STE (fast OR controller) cannot revive mmixcore's
dead cold cascade; the annealed/STE ramp has no value-domain alive phase at any rate. The
prior "0.827" must have been a non-mmixcore / pre-revived setup. **Revival is mandatory;
the STE is refinement-only** — my fast-loop implementation is correct, it just can't do a
job the STE was never able to do here.

Next: the two-stage path. The proxy ramp already revives to 0.92, and the proxy-fast path
has a built-in `_fast_stabilize(ttfs_blend_fast_stabilize_steps)` cascade-refine pass that
the 0.92 run left at 0 steps. Closing that cliff is the cheap first test (KD-refine); an
STE-refine in that same stage is the code-change follow-on.

## Two-stage results + verdict (so far)

- **Best = proxy revive + KD-refine: 0.9531 @ S=4 in 67 s** (no theta). The clean,
  fast, deployable two-stage path. ~0.95 plateau, ~0.5pp below the 0.96 floor.
- **STE-refine did NOT beat KD-refine** (0.9369 FT < 0.9466): on the *revived* cascade the
  staircase-backward hedge is no better than plain KD recovery for this model/S. The STE
  was the deep-high-S plateau fix; at S=4 it adds nothing over KD-refine.
- **Latent bug found**: `ttfs_theta_cotrain` + `ttfs_blend_fast` crashes Weight
  Quantization — `PerceptronTransformer.get_effective_weight` does
  `scale * weight / activation_scale` with a per-channel (out-dim) `activation_scale`
  that doesn't broadcast against the weight there. theta_cotrain on the proxy path is
  unsupported until `get_effective_weight` handles per-channel θ. (KD-refine path is
  unaffected — it doesn't promote θ.)

**Honest status on AC1 (≥96%@S=4, cascaded, 9-deep mmixcore):** best is 0.9531 — a ~0.95
plateau consistent with all prior deep-cascade research.

## S-sweep (the decisive AC2/AC3 measurement) — two-stage KD-refine, deployed (SCM identity, 1000 samples)

| S | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| deployed | 0.9473 | 0.9457 | 0.9549 | 0.9534 |

FT wall 69–80 s at **every** S (AC5 met across the sweep). ANN ≈ 0.983.

**FLAT at ~0.95 — increasing S does NOT close the gap.** No monotonic rise, no approach to
the ANN; the spread (0.946–0.955) is run-noise on 1000 samples. **This is the decisive
result: for cascaded single-spike TTFS the ~3pp deployment gap is S-INDEPENDENT.**

Why: cascaded fire-once neurons fire on the *partial* running sum (greedy), so the loss is
a premature-firing / death-cascade *firing-gain* deficit — NOT a timing/quantization
resolution limit. Higher S gives finer spike *timing* but cannot un-fire a neuron that
fired early on incomplete information. So S buys nothing here (matches
[[cascaded_ttfs_correctable_gain]] "collapse is NOT quantization" and the round-2 verdict
"more high-S budget HURTS the deep cascade").

**Consequence:** AC1 (≥96%@S=4) AND AC2 (lossless by S≤32) are **both unreachable for
cascaded single-spike TTFS** with current mechanisms — the gap is inherent to greedy
fire-once execution (which the user requires kept unchanged). The monotone→lossless
expectation HOLDS for LIF / synchronized TTFS (complete-sum codes — [[lossless_fast_mnist_campaign]]:
LIF stabilize 400→6000 → 0.9784 ≥ ANN) but NOT for the cascaded partial-sum code. Closing
the cascaded gap requires attacking the firing-gain deficit itself (stronger per-neuron
revival / per-sample correction), not more S and not the STE.

## Also confirmed this phase

- The SCM device-mismatch fix holds end-to-end on fresh full runs (cascaded + proxy):
  `NF↔SCM cascaded decision agreement 1.0000`, `torch↔deployed-sim parity 1.0000`.
- proxy_fast left `ttfs_blend_fast_stabilize_steps=0` → the cliff was not closed (0.92);
  raising it is the cheap lever to test for the proxy path itself.
