# Direction A — Analytical decode model + depth-budget law (H1)

**Verdict: PROMISING.** The closed-form decode model is bit-exact with the genuine
node; the per-layer retention law `ρ(S) = 1 − c/S` (c ≈ 1.9) and the
window-shortening mechanism are confirmed quantitatively. The depth-budget law is
**`d_max(S) ≈ 0.56·√S`** — a `√S` (not the previously-hypothesised linear `T`)
scaling, which is the corrected, more pessimistic, and more useful result. The law
is *prescriptive*: the lever it implies (per-depth threshold shrink `θ_d ∝ γ^d`)
lifts genuine accuracy from **0.074 → 0.809** at the primary benchmark (depth=3,
S=8) cold, with no genuine fine-tuning.

Prototype: `experiments/char_decode_law.py`. All numbers float64, deterministic,
CPU, multi-seed.

---

## 1. The closed-form decode model (and its exact validation)

### 1.1 Encode
A post-θ-normalised value `v ∈ [0,1]` fires a **single spike** at the **local
cycle**
```
τ(v) = round(S·(1 − v))            (high value → early spike)
```
The latched (cumulative-max) signal the consumer sees is `1` from cycle `τ`
onward — a ramp of length `S − τ`. Verified bit-exact against
`spike_modes.to_spikes(..., "TTFS")` for all `v`.

### 1.2 Decode (the ramp double-integral)
A consumer neuron does **not** read a value; it *re-integrates* arriving latched
ramps. With per-cycle weighted input `w_i` from input `i` latched from local cycle
`a_i`, the genuine `TTFSActivation` computes (verified by tracing the node):
```
ramp(t)     = Σ_i w_i · [t ≥ a_i]                 # first integral
membrane(t) = Σ_{s=0}^{t} ramp(s)                 # second integral (double)
fire at first t with membrane(t) ≥ 1
decode      = (S − t_fire) / S                    # 0 if it never crosses in [0,S)
```
The membrane is the **double integral** of the input ramp — a *quadratic* in the
time since arrival. A single `w=1` input latched from `a` gives
`membrane(a) = 1`, so it fires the same cycle and decodes `(S−a)/S ≈ v`
(lossless up to rounding). The loss is a *many-weak-inputs* and *late-arrival*
effect, not a single-input effect.

### 1.3 Latency / depth
Layer `d` cannot fire before global cycle `d` (1 cycle/hop) and integrates **only
inside its own window `[d, d+S)`**. The producer fired one hop earlier, so the
relative arrival inside the consumer window is `τ_producer − 1`. The window length
each layer gets is **always `S`**, but its inputs land progressively later inside
that window with depth (§3) — that, not a literally-shorter window, is the
shortening mechanism.

### 1.4 Validation — the model is exact
`closed_form_matches_node`: over 400 random multi-input configs per S,
```
S= 4: max |analytic_decode − TTFSActivation| = 0.00e+00
S= 8: max |analytic_decode − TTFSActivation| = 0.00e+00
S=16: max |analytic_decode − TTFSActivation| = 0.00e+00
S=32: max |analytic_decode − TTFSActivation| = 0.00e+00
```
The `analytic_decode` closed form **reproduces the deployed node bit-exactly**.
The model below is built on this, not on an approximation.

---

## 2. Per-layer retention law `ρ(S) = 1 − c/S`

Depth-0 attenuation ratio (random-init, `attenuation_profile`, depth-0 isolates the
pure per-layer encode+threshold loss with no upstream attenuation):

| S | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ρ₀ (seed 0) | 0.556 | 0.769 | 0.885 | 0.943 | 0.971 | 0.985 |
| ρ₀ (seed 1) | 0.524 | 0.751 | 0.874 | 0.935 | 0.969 | 0.984 |
| ρ₀ (seed 2) | 0.566 | 0.777 | 0.878 | 0.939 | 0.970 | 0.984 |
| c = (1−ρ₀)·S | ≈1.8 | ≈1.85 | ≈1.9 | ≈1.9 | ≈1.9 | ≈2.0 |

**Fitted: `ρ(S) = 1 − c/S` with c = 1.91 ± 0.07** — flat in S and across seeds.
The `1/S` form is confirmed: `(1−ρ)·S` is constant. (`c≈1.9` ≈ 2 is the mean
late-by-one-cycle rounding bias of `round(S(1−v))` plus the quadratic crossing
delay; it does not grow with S.)

---

## 3. Window-shortening mechanism (the death cascade, measured)

`fire_time_by_depth` on the **trained** cascade records each neuron's mean **local**
fire-cycle (within its own `[d, d+S)` window) and its no-fire fraction. Death = the
local fire-cycle drifts to the window end `S`, so the membrane can't cross in time.

| S | depth → | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| **8** | mean local fire | 3.5 | 6.1 | **dead** | **dead** |
| | frac no-fire | 0.15 | 0.30 | 1.00 | 1.00 |
| | ratio | 0.80 | 0.12 | 0.0 | 0.0 |
| **16** | mean local fire | 6.5 | 9.9 | 14.6 | **dead** |
| | frac no-fire | 0.15 | 0.16 | 0.64 | 1.00 |
| | ratio | 0.90 | 0.50 | 0.005 | 0.0 |
| **32** | mean local fire | 12.4 | 15.5 | 21.6 | 29.2 |
| | frac no-fire | 0.15 | 0.15 | 0.24 | 0.65 |
| | ratio | 0.95 | 0.90 | 0.33 | 0.02 |

(mean over seeds 0,1.) The fire-cycle **marches toward S with depth** and the
no-fire fraction explodes the moment the mean fire-cycle reaches ~S. This *is* the
window-shortening: deep layers fire so late that the rest of their window is too
short for the next layer's quadratic ramp to cross threshold → it never fires → 0
→ the layer below starves harder → cascade death.

---

## 4. The depth-budget law `d_max(S) ≈ 0.56·√S`

Express the fire position as a **fraction of the window**, `p_d = fire_d / S`:

| S | p₀ | per-layer fractional drift δ | δ·√S |
|---|---|---|---|
| 8 | 0.44 | 0.329 | 0.930 |
| 16 | 0.41 | 0.252 | 1.007 |
| 32 | 0.39 | 0.175 | 0.992 |

Two clean constants emerge:
- initial fractional fire position **`p₀ ≈ 0.40`** (≈ the encode-loss position, ~const),
- per-layer drift **`δ ≈ b/√S` with `b ≈ 1.0`** (δ·√S = 0.93, 1.01, 0.99 — flat).

A neuron at depth `d` dies when `p_d = p₀ + d·δ ≥ 1`:
```
        (1 − p₀)·√S        0.6·√S
d_max ≈ ───────────── ≈ ───────── ≈ 0.56·√S      (b≈1, p₀≈0.40)
             b
```

**Measured d_max (depth at which mean genuine acc first fails to beat chance+0.10):**

| S | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| measured d_max | 0 | 2 | 3 | 3 | 4 |
| `0.56·√S` (rounded) | 1 | 2 | 2 | 3 | 4 |

Matches within ±1 everywhere for S ≥ 8 (S=4 is below the model's validity — encode
loss alone, ρ(4)≈0.55, kills even depth-2). The genuine-acc grid that defines
d_max (mean over seeds 0,1, digits, width 64):

```
       d2     d3     d4     d6
S= 4  0.09   0.08   0.08   0.08      (collapsed at all depths)
S= 8  0.62   0.08   0.08   0.08
S=16  0.64   0.49   0.08   0.08
S=32  0.54   0.72   0.19   0.08
S=64  0.41   0.60   0.28   0.13
```

### 4.1 This CORRECTS the prior `d_max ≈ T` hypothesis
The root artifact (§2, H1) hypothesised `d_max ≈ T = S`. **That is wrong** — it
overestimates the budget ~3-10×. The true scaling is `√S`: doubling S buys only
`√2 ≈ 1.4×` more survivable depth, not 2×. The reason: the fire-cycle drift δ is
itself `∝ 1/√S`, so the budget grows like `(1−p₀)/δ ∝ √S`, not like the window
length S. This is the single most important quantitative correction this direction
produces — it reframes "raise S" from a viable lever into a *very* expensive one
(a 4-deep cascade needs S ≳ 50 to survive cold; a 6-deep one needs S ≳ 115).

---

## 5. Compounding is super-geometric near death

Per-layer factor `f_d = atten[d]/atten[d−1]` (depth=4, trained):

| S | atten | per-layer factor |
|---|---|---|
| 8 | [0.80, 0.15, 0.0, 0.0] | [0.80, 0.19, 0.0, 0.0] |
| 16 | [0.90, 0.53, 0.011, 0.0] | [0.90, 0.59, 0.02, 0.0] |
| 32 | [0.95, 0.92, 0.30, 0.02] | [0.95, 0.97, 0.33, 0.06] |

If the cascade were *purely* geometric the factor would be constant ≈ ρ(S). It is
not: the factor is ≈ρ at depth 0 then **collapses** (0.80 → 0.19 at S=8). This is
because the drift is in *time*, and once a layer's fire-cycle nears S the
crossing-failure (no-fire) fraction is highly nonlinear in arrival lateness — a
threshold effect. So: roughly geometric (`atten[d] ≈ ρ^d`) while signal is healthy,
then a sharp cliff at `d ≈ d_max`. The law predicts *where* the cliff is (§4); the
geometric region is only the approach to it.

---

## 6. Which fix-levers the law implies (and a positive test of the headline one)

The law localises the failure precisely: **the threshold θ is effectively too high
at depth, so deep neurons fire too late (or never).** That immediately ranks the
levers:

1. **Per-depth θ-shrink / input-scale boost (VIABLE, law-derived, deployable).**
   Lowering the effective threshold at depth `d` pulls `membrane ≥ 1` to an earlier
   cycle — directly counteracting the `+δ` fire-cycle drift. Mechanism-faithful and
   expressible as a *trained per-layer `activation_scale`* (no decode change → stays
   bit-exact with HCM). **Positive test** (`θ_d *= γ^d`, primary benchmark depth=3,
   S=8, cold, no genuine FT):

   | γ | 1.0 (base) | 0.85 | 0.7 | 0.6 | 0.5 |
   |---|---|---|---|---|---|
   | genuine acc (seed 0) | 0.074 | 0.074 | 0.134 | 0.620 | **0.809** |

   Multi-seed at γ=0.5: seed0 0.074→0.809, seed1 0.085→0.588, seed2 0.071→0.481
   (continuous ceilings 0.944 / 0.749 / 0.655 respectively). **A single scalar γ
   recovers the bulk of the death-cascade gap in every seed**, confirming the law is
   not just descriptive but prescriptive. This is the seed of Direction B
   (analytical per-depth pre-compensation) — and it should be a *closed-form*
   `γ_d` derived from `p₀, δ(S)`, not greedy-searched (greedy didn't transfer
   before).

2. **Shorten effective depth (VIABLE).** d_max ∝ √S means depth is the dominant
   cost. Skip/residual paths or segment re-grouping that cut a layer buy more than
   doubling S. (Direction E.)

3. **Raise S (WEAK / expensive).** Only `√S` payoff; `d_max ≈ 0.56√S` ⇒ each extra
   layer of budget costs a ~`(2d/0.56)·d` increase in S. Confirms the empirical
   "more S gave diminishing returns."

4. **Per-neuron spike-time offset / phase advance (VIABLE in principle).** The drift
   is a constant `+δ` in fire-cycle; a learned per-depth phase-advance of the decode
   window would null it exactly. Equivalent to lever 1 if folded into θ; cleaner if
   done as a window offset (but that touches the decode → only if kept bit-exact).

Levers the law **down-ranks**: bias-only corrections (bias is double-integrated and
helps the early crossing little once the ramp is starved); pure S-annealing (it
walks toward the wrong basin — the death is structural, not optimisation).

---

## 7. Honest limitations / what would falsify or extend this

- `p₀ ≈ 0.40` and `b ≈ 1.0` are fitted on digits-trained width-64 cascades; they
  encode the trained activation *statistics* (sparsity, magnitude growth with
  depth) as much as the pure decode. A very different activation distribution would
  shift the constants (not the `√S` *form* — that comes from `δ ∝ 1/√S`, which is
  geometry, not data). Worth re-fitting `b` on the real mmixcore activation stats
  before trusting the absolute `d_max`.
- d_max is read at a chance+0.10 threshold; the cliff is sharp so this is robust,
  but the exact integer can move ±1 with the margin.
- The γ-shrink test is *cold-conversion* only. It shows the law points at a real,
  large lever; it is not yet a tuned, transfer-validated fix (that is Direction B).
- The `√S` law is the strong, surprising claim. It is supported by (a) the measured
  `δ·√S ≈ 1` constancy and (b) the measured d_max grid matching `0.56√S` ±1. Both
  are CPU-fast to reproduce via `char_decode_law.py`.

---

## 8. Transfer outlook to the real mmixcore pipeline

**Plausibly transfers, with a caveat.** The decode model (§1) is the *deployed*
dynamics verbatim (bit-exact with the node, which is bit-exact with HCM/nevresim/
SANA-FE per the memory notes), so the mechanism is real on-chip, not a toy artifact.
The `√S` budget and the θ-too-high-at-depth diagnosis should hold for any cascade.
The reason the real pipeline only shows ~3pp (not collapse) is exactly what the law
predicts: mmixcore is *shallow* (`d < d_max(S)` for its S) and is trained through
the genuine forward (which partially learns around the drift). The actionable
transfer is **lever 1 as a closed-form per-depth `activation_scale` initialisation /
constraint** derived from `p₀, δ(S)` — cheap, mechanism-faithful, deployable, and
exactly the kind of per-layer scale the calibration step already owns. Before
shipping, re-fit `b` on real activation stats (§7) and validate the closed-form
`γ_d = γ_d(p₀, δ, S)` transfers under genuine FT (hand-off to Direction B).
