# Phase 2 synthesis — the deployable recipe (G → F/N2)

Phase 1 proved the death cascade is a correctable per-layer decode gain-distortion.
Phase 2 turned that into a **principled, closed-form, validated, deployable recipe**
and showed the levers COMPOSE. Artifacts: `21_closed_form_correction.md`,
`22_stacked_levers.md`, `16_novel_ideas.md`; prototypes `experiments/{closed_form,
stack,novel}.py`.

## The three deployable levers (all keep the decode bit-exact → NF↔SCM parity holds)

### G — gain correction (calibration-only, closed form, NO training)
Per-spike effective gain `g_eff(τ) = (T−τ+1)/(T+1)` (derived from the ramp
`R(τ) ∝ (T−τ)²` vs the intended linear; bit-exact vs the capacity model). Combined
with the fire-time drift `E[τ_d] = τ_0 + d·√S`, the per-layer correction is
**geometric**:

    θ_d ← θ_d · g_d,   g_d = ρ_0 · γ^d,   ρ_0 = 1 − c/S (c≈1.9),   γ = 1 − √S/(S+1)

i.e. shrink deep layers' decode threshold so a late spike's quadratic ramp decodes
to the teacher's value. This is the S-derived, depth-explicit unification of char's
hand-set `0.5^d` and precomp's `mean(relu)/target`. A data-grounded **calib_fire**
variant (`g_d = g_eff(τ_d_calib)/g_eff(τ_0_calib)`, capped ≤1, from one calibration
forward) self-limits and is the safer default past `d_max`.
- **Cold (depth-3 S=8, 3 seeds):** 0.077 → 0.645 vs oracle 0.745 → **85% of the
  oracle gap closed with ZERO metric access**. Recovers 80–97% of oracle across the
  death-cascade regime (depth 2–3 × S 4–16).
- Deployable: pure per-layer `activation_scale` trim; applying it reproduces the
  decode bit-exact (verified `max|Δoutput| = 0`).
- Gate to `depth ≥ d_max(S) = 0.56·√S` (geometric over-corrects a few points in the
  already-healthy regime; calib_fire self-gates).

### F — genuine-forward STE fine-tune (the EXISTING boundary_surrogate_temp)
Back-prop through the genuine single-spike cascade (`cascade_forward grad=True`).
**Alone it is dead** (no gradient on a starved cascade — this is exactly why the STE
read as "refuted" in Phase 0). **But G→F is a non-additive synergy:** corrected+FT >
plain+FT in **all 12/12** seed×config cells; depth-3 S=8 0.711→0.857 (seed0 hits
0.941 ≈ continuous 0.944); **depth-4 S=8 plain-FT 0.077 (totally stalls) → G+F
0.485**. The payoff is largest exactly where plain FT stalls (deep/low-S). G+F beats
the non-deployable static-gain oracle in every deep cell.

### N2 — bias-only phase-advance fine-tune (novel; the deepest-reach lever)
Train ONLY `layer.bias` (freeze weight directions) through the genuine cascade — the
network learns a **positive `bias_norm` = a spike-time phase advance** pulling deep
neurons' fire time into the faithful early window. Reaches **99% of the continuous
ceiling** (vs G's 88%), and **uniquely revives past d_max**: depth-6 S=8 where both
the gain trim (0.077) AND full FT (0.110) fail, N2 reaches 0.265. Deployable (bias is
a first-class chip parameter the genuine node already folds; decode untouched).
(N1 value-warp was refuted — warping small values into the 0.5/S dead-zone adds more
quantization noise than it removes; fully subsumed by N2.)

## The recipe (ordering is load-bearing)

    G  (calibration: revive the cold cascade, alive deep layers)
     → F or N2  (genuine-forward fine-tune: now well-conditioned)
     [→ K input→decode skip, only for models deeper than d_max(S)]

This maps directly onto the pipeline's calibration → fine-tune stages. The whole
recipe is deployable and parity-preserving by construction.

## Reframing (the publishable thesis)

"Cascaded single-spike TTFS is a representation-limited dead end" → **"Cascaded
single-spike TTFS is decode-gain-distorted and correctable."** The gap is an
implementation artifact (ramp over-weights early spikes ∝(T−τ)²), not a capacity
limit (timing carries log2(T+1) bits, same as rate; the ideal staircase already hits
LIF level). A closed-form per-depth threshold trim recovers ~85% cold; composed with
a (now well-conditioned) genuine fine-tune it reaches ~continuous on the toy. Next:
validate on the real mmixcore SCM pipeline.

## Caveats / honest scope

- The toy's *absolute* accuracy is task-ceiling-bound (a plain deep MLP underfits
  digits at depth 4–6); the transferable metric is **conversion-gap closure /
  retention**, not the absolute number.
- The real mmixcore is SHALLOW (most perceptrons at depth < d_max), so it runs only
  mildly into the death cascade — G's absolute win there may be modest (a few points
  on the deeper segments). G's dramatic win is for DEEP cascades; the mmixcore
  validation is primarily a "does it transfer / not regress" check.
- Constants `c≈1.9`, `p0≈0.40` were fit on digits width-64 stats; the √S FORM is
  geometry (robust), but absolute constants should be re-fit / calib_fire used on
  real mmixcore activation distributions.
