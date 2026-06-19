# Phase 1 synthesis — the death cascade is a correctable decode gain-distortion

Seven parallel first-principles investigations (artifacts 10–16) on the isolated
`cascade_lab.py` benchmark (depth-3 digits, S=8: continuous 0.94, genuine 0.07).
**Three independent agents converged on the same root cause and fix.** This is the
central result and reframes the whole problem.

## The headline result (publishable)

The cascaded single-spike TTFS "death cascade" (deep layers decode to ~0 →
accuracy collapses) is **NOT a representation/capacity limit**. It is a
**correctable per-layer decode gain-distortion**:

- **Capacity is identical to a rate code.** A single-spike timing code carries
  `log2(T+1)` bits/neuron — the *same* level count as the multi-spike rate code
  LIF uses (capacity agent). Single-layer `round()`-TTFS is an unbiased uniform
  quantizer (quant cost at S=8 is only **+0.019**).
- **The IDEAL single-spike decode already reaches LIF level.** An optimal
  (staircase) single-spike decode gets **0.764** at depth-3 S=8 vs the genuine
  cascade's **0.077** — the entire gap (decode cost **+0.688**) is *implementation*,
  not capacity. LIF-level cascaded TTFS is reachable **in principle**.
- **The mechanism: the ramp decode applies a quadratic gain.** A spike arriving at
  time `τ` is integrated as a ramp over `[τ,T)`, giving the consumer an effective
  weight `R(τ) ∝ (T−τ)²`, whereas a faithful decode wants the *linear* `L(τ) ∝
  (T−τ)`. So `R/L ∝ (T−τ)` falls 1.0 → ~0.22 across the window: **late-arriving
  (small-value, and crucially deep-layer) spikes are down-weighted 2–5×.** Deep
  layers fire late (latency ~1 cycle/hop) → systematically under-decoded → their
  output shrinks → the next layer fires even later → **geometric death cascade**.

## The depth-budget law (char agent, bit-exact decode model)

A closed-form decoder was derived and proven **bit-exact** to the genuine
`TTFSActivation` node (`max|diff| = 0` for S∈{4,8,16,32}). From it:

- Per-layer retention `ρ(S) = 1 − c/S`, `c = 1.91 ± 0.07` (flat in S, seed).
- Per-layer fire-cycle **drift** `δ ≈ 1/√S` (confirmed: δ√S = 0.93/1.01/0.99 at
  S=8/16/32); death triggers when the mean fire-cycle reaches ~S.
- **Depth-budget law: `d_max(S) ≈ 0.56·√S`** (measured d_max {S8:2, S16:3, S32:3,
  S64:4}). This **refutes** the prior `d_max ≈ T = S` guess and explains why
  "raise S" is a *weak, expensive* lever (4-deep cold needs S≳50): the budget
  grows like **√S**, not S. The earlier confounded-cross-S diminishing-returns
  observation is now explained mechanistically.

## The fix: per-layer analytical gain/threshold correction (cold, deployable)

All three promising agents land on the same lever — a **per-layer multiplicative
correction to `activation_scale` (the decode threshold θ)**, derived from
calibration statistics (NOT metric/oracle search, which didn't transfer before),
that inverts the per-depth gain so each layer's decoded value lands where the
teacher's does. **It is deployable: it only sets θ; the decode stays bit-exact.**

| source | correction | depth-3 S=8 genuine (cold) |
|---|---|---|
| baseline | — | 0.074 |
| char | θ_d *= 0.5^d (law-implied geometric) | **0.809** (seed0) |
| precomp | θ_d = mean(relu(act_d))/target, target≈0.55 | **0.698** (3-seed mean, **88% of gap**, matches the metric-searched bound with zero metric access) |
| capacity | oracle per-layer θ-scale (upper bound) | **0.909** (seed0) |

So a principled per-layer θ correction closes **~75–90% of the cold gap with no
retraining**. Caveats (traps the agents found):
- **Matching the per-channel MEAN (DFQ bias) HURTS** (collapses to 0.24) — it is
  anti-correlated with accuracy. The correction must be a *gain* (multiplicative θ),
  not a first-moment bias match. (This also explains the earlier DFQ-for-LIF and
  DFQ-bias-in-TTFS observations.)
- **A pure rescale cannot manufacture window budget**: past `d_max(S)` the deepest
  layers still die (depth-6 S=8 stays at chance). For `depth > d_max` you must pair
  the correction with **effective-depth reduction** or higher S.

## Refuted / partial (negative results, equally valuable)

- **Analytical timing proxy (true-D2 attempt): REFUTED.** A faithful forward does
  NOT give a usable gradient — smoothing the death-gate rewards pushing values up
  (firing earlier/outside the window), anti-aligned with deploy; it *hurts*. The
  **genuine-forward STE** (the existing `boundary_surrogate_temp`) IS the
  well-conditioned gradient and recovered **3× at S=16** — so that lever has real
  value *once the cascade is alive* (i.e. after the gain correction).
- **Multi-spike→single-spike curriculum: REFUTED.** Training at k=1 always loses;
  the (T−d)/T wall is untouched by code-annealing. (Consistent with the prior D1
  S-annealing failure.) Lesson: never fine-tune *at* the deployed single-spike
  forward on a starved cascade — fix the cascade first.
- **Effective-depth reduction (skip→decode): PARTIAL.** An input→decode concat skip
  trained through the cascade lifts 0.074→0.83 cold (+0.31 over a no-skip control),
  and is **deployable** (in-segment ConcatMapper routing, no extra spike, no host
  op). But at shallow depth-3, plain + full genuine-FT also recovers to ~continuous,
  so skips earn their keep only where the cascade is *deep* (depth > d_max) — the
  regime to test next.

## The Phase-2 program (derived from the above)

1. **Derive the PRINCIPLED closed-form per-layer correction** unifying the three:
   from `R(τ) ∝ (T−τ)²` and the fire-time model (`τ_d` drift ≈ d/√S), compute
   `g_d(S)` = expected `R/L` over each layer's decoded-value distribution; set
   `θ_d ← θ_d · g_d`. Validate it (a) cold across seeds/S/depth ≈ the oracle, and
   (b) that it SURVIVES + accelerates genuine fine-tuning (the real deployment
   trains through the genuine forward; this is the init that previously stalled).
2. **Validate on the REAL mmixcore pipeline** — implement the correction in the
   cascaded-TTFS calibration (a per-cascade-depth `activation_scale` trim gated by
   a flag), measure deployed Soft-Core-Mapping accuracy vs the 0.95 baseline. This
   is the key deliverable; the decode stays bit-exact so NF↔SCM parity must hold.
3. **Stack the levers** where depth > d_max: gain-correction (alive init) → genuine
   STE fine-tuning (now well-conditioned) → input→decode skip (extend d_max).
4. **Re-run the novel-ideas investigation** (failed on API overload in Phase 1).

This turns "cascaded TTFS is a representation-limited dead-end" into "cascaded TTFS
is gain-distorted and correctable" — a publishable reframing with a concrete,
deployable, theoretically-grounded fix.
