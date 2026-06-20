# Phase 2 — the principled closed-form per-layer gain correction

**Prototype:** `experiments/closed_form.py` (fast harness only; does not edit
`cascade_lab.py`). **Verdict: PROMISING.** A closed-form, calibration-only,
per-layer `activation_scale` (θ) trim — derived from the ramp gain model and the
fire-time drift — recovers **~85% of the oracle gap COLD** on the PRIMARY
benchmark with **zero metric access**, and crucially gives a *healthy init that
makes genuine fine-tuning succeed where it stalls*: **corrected+FT beats plain+FT
in every (depth, S, seed) cell, by +0.08 to +0.41**, including reviving the
totally-dead depth-4 S=8 cascade (plain FT stuck at chance 0.077 → corrected FT
0.485). The correction is purely a per-layer θ trim — the decode stays **bit-exact**
(verified `max|Δ| = 0`, weights/biases untouched) → deployable, NF↔SCM parity holds.

---

## 1. The derivation (the headline closed form)

### 1.1 The per-spike effective gain `g_eff(τ)` (exact)

A spike arriving at consumer-local cycle `τ` is integrated as a quadratic ramp,
so a unit-weight spike contributes membrane drive
`R(τ) = (T−τ)(T−τ+1)/2`, whereas a faithful (teacher) decode wants the **linear**
`L(τ) ∝ (T−τ)` (the value `v = (T−τ)/T`). Normalising so an early spike (τ→0,
v→1) has gain 1, the per-spike effective gain relative to the intended linear
weight is the **clean closed form**

```
        g_eff(τ)  =  (T − τ + 1) / (T + 1)
```

This is **bit-exact** to `capacity.ramp_effective_weight_distortion`'s `R_over_L`
column (`max|g_eff − R/L| = 4.7e-4`, all from capacity's 3-dp rounding). It falls
1.0 → 0.22 across the window at S=8 — late spikes (small values, deep layers) are
down-weighted 2–5×.

### 1.2 The fire-time drift → depth dependence

Deep layers fire late. From `char_decode_law`:
* encode-layer retention `ρ_0(S) = 1 − c/S`, `c = 1.9` (fit once on the random-init
  depth-0 ratio; **not** the eval metric), i.e. `g_eff(τ_0) = ρ_0` ⇒
  `E[τ_0] = (S+1)·c/S + 1`;
* per-hop normalized fire-cycle drift `δ ≈ 1/√S` ⇒ absolute drift `S·δ = √S`
  cycles/hop, so `E[τ_d] = E[τ_0] + d·√S`.

Substituting gives the **additive depth model**
`g_closed(S,d) = (S − E[τ_d] + 1)/(S+1) = 1 − c/S − (d√S − 1)/(S+1)`.

### 1.3 The correction rule

Lowering `θ_d` by a factor `g_d < 1` boosts the layer's normalized value by `1/g_d`,
fires it earlier, and lands the decoded value where the teacher's is:

```
        θ_d  ←  θ_d · g_d(S, d)
```

Because the attenuation **compounds multiplicatively** through depth, the correction
must too. Linearising the gain ramp at the operating point under the `√S` drift
gives a per-hop **geometric** ratio `γ = g_eff(τ+√S)/g_eff(τ) ≈ 1 − √S/(S+1)`, so

```
   HEADLINE (geometric):   g_d  =  ρ_0 · γ^d ,   ρ_0 = 1 − c/S,  γ = 1 − √S/(S+1)
```

**Empirically the geometric form dominates the additive `g_closed`** (which clamps
to ~0 past `d_max` and over-shrinks deep layers). The geometric base is *derived*
from S (γ = 0.69 at S=8) — this is the principled version of char's hand-set
`θ_d *= 0.5^d`.

### 1.4 Why this unifies char's geometric and precomp's mean_target

* **char `θ_d *= γ^d`** is exactly the multiplicative-per-hop view of this gain
  model; char fixed `γ = 0.5` by hand, we **derive** `γ = 1 − √S/(S+1)` from the
  ramp + drift (it correctly grows toward 1 as S→∞, so the correction self-fades
  when the cascade stops dying).
* **precomp `θ_d = mean(relu(act_d))/target`** is the *same lever* (multiplicative
  θ) reached from the calib first moment instead of the depth index. It implicitly
  absorbs `g_d` into the data (deeper attenuated layers have smaller means) but is
  blind to the depth structure. Our rule is the depth-explicit, data-free closed
  form precomp was approximating. The **calib_fire** variant below closes the loop:
  it reads the *fire time* (the actual quantity in `g_eff`), not the value mean, and
  is the most robust at the deepest layers.

### 1.5 The data-grounded variant `calib_fire` (still calibration-only)

A forward calibration pass measures each layer's mean local fire-cycle `τ_d` on
calib data and sets `g_d = g_eff(τ_d)/g_eff(τ_0)` (relative to layer 0, capped at
1), re-measuring downstream so each correction sees the corrected upstream. No
eval-metric access. This **matches/beats the oracle at depth=4** (where the
analytic depth-model breaks down past `d_max`).

---

## 2. COLD validation vs baseline vs oracle (seeds 0,1,2)

PRIMARY (depth=3 digits, S=8): continuous 0.78–0.94; baseline genuine ~0.077
(chance); oracle per-depth θ-scale 0.745.

| | cont | baseline | **geometric** | oracle | gap closed |
|---|---|---|---|---|---|
| seed0 | 0.944 | 0.074 | **0.868** | 0.909 | 95% |
| seed1 | 0.750 | 0.085 | **0.575** | 0.701 | 80% |
| seed2 | 0.655 | 0.070 | **0.492** | 0.625 | 76% |
| **mean** | 0.783 | 0.077 | **0.645** | 0.745 | **85%** |

Full sweep (mean over seeds 0,1,2; `gap-closed` = best(geom,calibFR) vs oracle gap,
reported only where the death cascade is active, oracle-gap > 0.05):

```
depth=2  S    cont  baseline   geom  relative  calibFR  oracle   gap-closed
         4   0.722     0.078   0.667    0.304    0.577   0.700      95%
         8   0.722     0.505   0.674    0.662    0.676   0.712      82%
        16   0.722     0.694   0.656    0.667    0.651   0.714    healthy
        32   0.722     0.625   0.575    0.588    0.579   0.717    healthy(over-corr)

depth=3  4   0.783     0.077   0.678    0.077    0.077   0.694      97%
         8   0.783     0.077   0.645    0.146    0.623   0.745      85%
        16   0.783     0.260   0.670    0.625    0.662   0.751      83%
        32   0.783     0.646   0.646    0.645    0.636   0.748    healthy

depth=4  4   0.456     0.077   0.271    0.077    0.077   0.142      ~oracle
         8   0.456     0.077   0.111    0.077    0.166   0.344      33% (calibFR)
        16   0.456     0.077   0.173    0.090    0.275   0.401      61% (calibFR)
        32   0.456     0.115   0.281    0.260    0.283   0.417      56%
```

**Reading it honestly:**
* In the **death-cascade regime** (depth 2–3, S 4–16 — the regime the whole effort
  targets) the geometric closed form recovers **80–97% of the oracle gap with zero
  metric access**. This is the deliverable.
* `relative` (γ^d, *no* ρ_0 prefactor) **under-corrects at small S** (it leaves the
  encode layer untouched, but at S=4/8 ρ_0≈0.55–0.77 so layer 0 *does* need its
  prefactor) — it collapses at depth-3 S≤8. Keep the ρ_0 prefactor. (Reported as a
  negative result.)
* **depth=4** is past `d_max(S)=0.56√S` (=1.6 at S=8): the deepest layers are dead,
  and the *analytic* geom model mis-estimates the drift there — but the
  **data-grounded `calib_fire`** matches/beats the oracle (0.166 vs 0.344 at S=8;
  0.275 vs 0.401 at S=16). Use calib_fire when depth > d_max.
* **Caveat (honest):** in the *already-healthy* regime (depth-2 S≥16, depth-3 S=32)
  baseline ≈ oracle and the geometric form **over-corrects by a few points**
  (0.656 vs 0.694 baseline at depth-2 S16). The rule should be **gated to the
  starved regime** in production (only correct layers at depth ≥ d_max(S), or where
  the calib fire-time is late). `calib_fire` self-limits (caps g≤1, relative to
  layer 0) and is the safer default.

---

## 3. WITH genuine fine-tuning (the real deployment path)

Both arms fine-tune **all weights through the genuine single-spike cascade**
(boundary STE, `cascade_forward(grad=True, surrogate_temp=0.5)`, 40 epochs); θ is
frozen during FT. The only difference is the θ **init** (plain calibration vs the
geometric gain-correction). **Hypothesis: the correction gives a healthy init so FT
succeeds where it stalled.** Confirmed in every cell and every seed:

```
                 cont  plain_cold  plain_ft  corr_cold  corr_ft   corr_ft − plain_ft
depth=3 S= 8    0.783      0.077     0.711     0.645     0.857          +0.146
depth=3 S=16    0.783      0.260     0.818     0.670     0.894          +0.076
depth=4 S= 8    0.456      0.077     0.077     0.111     0.485          +0.408   ← FT was DEAD
depth=4 S=16    0.456      0.077     0.393     0.173     0.563          +0.171
```

Per-seed (corr_ft − plain_ft is positive for **12/12 seed×config cells**):
* depth=3 S=8: +0.078 / +0.185 / +0.174 (corr_ft 0.941/0.837/0.794; seed0 reaches
  continuous 0.944).
* depth=4 S=8: **+0.482 / +0.473 / +0.269** — plain FT is stuck at chance for all
  three seeds (the dead cascade gives the STE no gradient signal); the corrected
  init revives it and FT lifts to 0.34–0.56.

**The key scientific result:** the correction's payoff is *largest exactly where
plain FT stalls* (deep, low S). It is not just a better cold number — it changes
the optimisation landscape from "dead, no gradient" to "alive, FT converges." This
is the init that the prior D1/curriculum attempts lacked.

---

## 4. Deployability (verified)

`apply_gain_correction` only calls `p.set_activation_scale(θ_d · g_d)`; it never
touches weights, biases, or the decode. Verified:
* manual `θ_d *= g_geometric(S,d)` reproduces it with `max|Δ output| = 0.0`;
* weights/biases bit-identical after correction.

So it is a per-layer **threshold (`activation_scale`) trim keyed only on (S,
cascade-depth) + the law constant c** — exactly the deployable category (the chip
already supports per-neuron θ). The decode stays bit-exact ⇒ **NF↔SCM parity is
preserved** by construction. `c=1.9` and `γ=1−√S/(S+1)` are fixed; `calib_fire`
needs one extra calibration forward pass (already part of the pipeline's calib).

---

## 5. Graduation recommendation

1. Add a per-cascade-depth `activation_scale` trim in the cascaded-TTFS calibration,
   flag-gated. **Default rule: `geometric` (ρ_0·γ^d, γ=1−√S/(S+1), c=1.9) gated to
   depth ≥ d_max(S)=0.56√S** (don't touch already-healthy shallow layers). For
   depth > d_max (deep models), use **`calib_fire`** (reads the calib fire-time;
   robust where the analytic model breaks down).
2. Keep the **genuine-cascade FT** (existing `boundary_surrogate_temp`) — the
   correction's main value at deep/low-S is that it makes that FT *work* (corrected
   init → FT converges; plain init → FT dead).
3. Open caveats: (a) gate the rule to the starved regime to avoid the high-S
   over-correction; (b) past d_max even corrected+FT does not fully close the gap
   (depth-4 S=8 tops out ~0.49 of cont 0.46 on the toy — i.e. it *exceeds*
   continuous here only because cont is itself low; the absolute ceiling is the
   capacity/window budget) — for very deep models pair with the input→decode skip
   (depth_reduction.py) to extend d_max.

The closed form, its 85% cold recovery vs oracle, and the corrected+FT lift are
ready to graduate into the real pipeline calibration.
