# Cascaded single-spike TTFS — first-principles problem framing

**Goal:** close the deployed-accuracy gap of the *cascaded* `ttfs_cycle_based`
single-spike SNN (~0.93–0.95 on mmixcore/MNIST) up to the LIF / synchronized
level (~0.97), WITHOUT giving up the cascade's power advantage (one spike per
neuron, pipelined, no per-layer global sync). This is an open research problem;
prior levers (S-annealing D1, boundary-gradient STE D2-crude, more training D4,
higher S) gave diminishing/no returns — see `fine_tuning_research_directions.md`.

This document is the living root artifact: the mechanism, the measured baseline,
the hypotheses, and the research directions. Companion artifacts (per-direction
findings, prototypes) live alongside it; the isolated harness is `cascade_lab.py`.

---

## 1. The deployed dynamics (what the hardware actually does)

Each neuron fires **at most once**. Its activation `v ∈ [0,1]` (post-scale) is
encoded as a **spike time** `τ = round(T·(1 − v))` (TTFS: high value → early
spike). A downstream neuron integrates an arriving spike as a **ramp**: a spike
at time `τ` contributes to membrane over the remaining window `[τ, T)`, and the
decoded value ≈ `(T − τ)/T ≈ v`. Per-core latency adds **1 cycle per hop**, so
layer `d` cannot fire before cycle `≈ d`.

Contrast LIF: multi-spike **rate** code, value = `count/T`, available regardless
of *when* in the window the spikes land — linear, depth-robust.

## 2. First-principles failure mode: the depth death-cascade

Two compounding effects, both intrinsic to the *forward* (so unfixable by
changing the decode — it must stay bit-exact with HCM):

1. **Window shortening with depth.** Deep layers fire late (latency `≈ d`), so
   their ramp window `[τ, T)` is short → they can express less dynamic range →
   their decoded value is **attenuated**. At depth `d` the maximum representable
   value is bounded by `(T − d)/T`.
2. **Compounding.** Each layer's attenuated output is the next layer's input;
   attenuation multiplies through depth → deep layers **starve to ~0** (a death
   cascade), collapsing the network to chance.

`(T − d)/T` makes the prediction explicit: the cascade has a **depth budget**
`d_max ≈ T`. Below it, signal survives; near/over it, deep layers die. So the
fix axes are: raise `T` (S — costly, latency), **shorten effective depth**, or
**pre-compensate the known per-depth attenuation** so the deployed values land
where the teacher's do.

## 3. Measured baseline (`cascade_lab.py`, float64, deterministic, <1s)

**Trained-cascade conversion gap** (continuous-ReLU cascade trained, then
converted cold to the genuine single-spike cascade — NO genuine fine-tuning, so
this is the *raw representation limit*):

| depth | S=4 | S=8 | S=16 | atten-ratio-by-depth (S=8) |
|---|---|---|---|---|
| 3 | gap +0.36 | ~0 | +0.13 | `[0.77, 0.42, 0.47]` |
| 6 | +0.25 (→chance) | +0.25 (→chance) | +0.09 | `[0.78, 0.47, 0.23, 0.002, 0.0, 0.0]` |

The **death cascade is explicit**: at S=4/8, depth-6 deep layers decode to
`0.0` (`atten → 0`) → genuine accuracy collapses to chance. **S=16 prevents it**
(`atten` stays ≈1 through depth 6). The first-layer *encode* attenuation is also
S-dependent: ratio `0.55 @S=4 → 0.94 @S=16` (random-init profile confirms across
seeds). This matches the `d_max ≈ T` prediction: depth 6 needs `T ≳ 16` to
survive cold.

**Why the real pipeline only shows ~3pp (not collapse):** mmixcore is *shallow*
(few segments, mostly subsume) and is *trained through the genuine forward* (the
FT partly recovers the death cascade). The toy exposes the un-trained limit
cleanly — the ideal fast substrate for candidate fixes.

## 4. Hypotheses (falsifiable)

- **H1 (depth-budget law):** deployed deep-layer dynamic range is capped at
  `(T − depth)/T`; accuracy collapses when a needed layer's depth ≳ T. → predicts
  the S=16-fixes-depth-6 result; test the law quantitatively, derive `d_max(S)`.
- **H2 (pre-compensation transfers):** a per-depth analytical scale/threshold
  correction derived from the decode model (NOT greedy-searched — that was tried
  and didn't transfer) flattens `atten → 1` and recovers cold-conversion accuracy.
- **H3 (well-conditioned proxy):** a differentiable proxy modelling sub-window
  spike timing (true D2) gives gradients whose optimum transfers to the genuine
  cascade, unlike the staircase proxy and the crude soft-sigmoid STE.
- **H4 (code relaxation):** annealing a multi-spike (rate-like, depth-robust) code
  → single-spike (k spikes → 1) walks the optimizer into the deployed basin
  better than S-annealing (D1, which failed).
- **H5 (depth reduction):** shortening effective cascade depth (skip/residual
  paths, segment re-grouping, width-for-depth) directly raises the depth budget.
- **H6 (capacity):** quantify single-spike-timing vs rate channel capacity at
  given (S, depth); is LIF-level reachable in principle, and what is the optimal
  code? Bounds the whole effort.

## 5. Research directions (pursue broadly — not limited to prior D1–D6)

A. **Analytical decode model + depth-budget law** (grounds H1) — closed form for
   the ramp-integrate decode vs `τ`, latency, S; predict the attenuation curve.
B. **Depth-aware analytical pre-compensation** (H2) — per-depth θ/scale/bias.
C. **Timing-aware soft-spike proxy with matched gradients** (H3, true D2).
D. **Multi-spike→single-spike relaxation curriculum** (H4).
E. **Effective-depth reduction: skip/residual + segment re-grouping** (H5).
F. **Information-capacity / optimal-code analysis** (H6).
G. **Open / non-obvious:** learnable per-neuron spike-time offsets (phase shift);
   depth-staggered decode windows; complementary dual-spike codes; input-encoding
   redesign; normalization that is invariant to window length; etc.

Each direction is investigated in isolation on `cascade_lab.py` (fidelity =
`atten → 1`; task = continuous-vs-genuine accuracy) for fast iteration; the
winner(s) graduate to the real pipeline.
