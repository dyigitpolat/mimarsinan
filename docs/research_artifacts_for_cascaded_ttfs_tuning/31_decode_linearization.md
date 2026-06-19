# Phase 3 — LINEARIZE the effective decode by warping the ENCODE

The death cascade (a per-DEPTH gain distortion) is now handled (G / the teacher-blend
genuine ramp revive it to ~0.93). **THE RESIDUAL** this phase attacks: after the death
cascade is gone, the genuine single-spike cascade still caps ~3.5pp below LIF (~0.94 vs
~0.975). Root-cause hypothesis (from the brief): the ramp decode gives an arriving spike
at local time `τ` a per-spike effective gain that is QUADRATIC in the remaining window,
a PER-SAMPLE nonlinearity in `τ` that a static per-neuron `θ` cannot invert; only
training approximates it. The capacity result says the IDEAL STAIRCASE (linear
single-spike decode) reaches LIF level, so the residual is the gap between the deployed
`(T-τ)²` ramp decode and the ideal LINEAR decode — NOT capacity.

**The fix tested here: choose the ENCODE (fire time `τ = φ(v)`, which is free) so the
deployed quadratic ramp DRIVE is LINEAR in `v` — invert the gain, encode-then-decode =
identity.** Prototype: `experiments/linearize.py` (CPU, ~21s, multi-seed, S∈{4,8,16,32},
depth∈{3,4}). Reuses the validated G machinery / flow cache from `closed_form.py`.

---

## 1. The mechanism, made exact (bit-exact with the genuine node)

A consumer double-integrates its arriving latched ramps. A SINGLE upstream spike at
consumer-local cycle `τ` (unit weight) drives the consumer membrane at the end of the
window to a TRIANGULAR number:

    R(τ) = Σ_{j=1}^{k} j = k(k+1)/2,   k = S − τ   (remaining window)

so the per-spike DRIVE is **quadratic** in `k`, whereas a faithful (teacher / staircase)
decode wants the value to enter **linearly**: the spike was emitted to carry
`v = (S−τ)/S = k/S` (uniform TTFS encode `τ = round(S(1−v))`). Normalising by the value
at `τ=0`:

    R_norm(τ) = k(k+1)/(S(S+1))   (deployed, quadratic)
    L_norm(τ) = k/S = v           (intended, linear)
    g_eff(τ)  = R_norm/L_norm = (S − τ + 1)/(S + 1)     ← the per-spike gain, exact

A consumer summing weighted inputs computes `Σ_i w_i · R_norm(τ_i)` but the teacher
wants `Σ_i w_i · v_i`. A static per-neuron `θ` only rescales the SUM; it cannot invert a
per-input nonlinearity in `τ`. That is the residual.

## 2. THE LINEARIZING ENCODE φ(v) (the closed form)

The encode is free: pick `τ = φ(v)` so the deployed quadratic drive equals `v`. Set
`R_norm(φ(v)) = v`:

    k(k+1)/(S(S+1)) = v   ⇒   k = φ_k(v) = (−1 + √(1 + 4·v·S(S+1))) / 2
    τ = φ(v) = round(S − φ_k(v))

Equivalently a **value PRE-WARP** `v → w(v) = φ_k(v)/S` fed to the STANDARD encoder
`τ = round(S(1−w))` lands the same fire time. So the linearizing encode is a per-neuron
**monotone value map**. (`§0` of the prototype prints φ; `R_norm(φ(v)) = v` is exact by
construction up to the integer-`τ` grid.)

**The single-spike tradeoff (`§1`).** Removing the gain trades a SYSTEMATIC distortion
for EXTRA quantization noise. The uniform encode has a large NEGATIVE drive bias (=the
death-cascade gain): `E[R_norm−v] = −0.125/−0.146/−0.156/−0.161` at S=4/8/16/32. The
linearizing encode zeros that bias (`+0.012/+0.003/+0.001/+0.000`) but the integer-`τ`
grid becomes NON-uniform (dense near `k=S`, sparse near `k=0`), so its zero-mean noise
sits at std `0.091/0.048/0.025/0.013`. Net: a single spike still has only `S+1` levels —
linearization MOVES the error from a coherent gain into incoherent quantization, which
the downstream weighted sum averages out far better (the gain is shared across inputs;
the quant noise is not).

## 3. THE HEADLINE — linearization closes the genuine→staircase decode residual

`§3b` isolates the per-neuron DECODE-LINEARITY effect with a layerwise value-snap
(faithful proxy: at high S it matches the genuine cascade — `§3a`: S=32 genuine 0.861 vs
proxy 0.870; at low S the genuine is lower by the SEPARATE latency death-cascade that G
handles). Each layer's decoded value is snapped to the code's decode; multi-seed mean:

| depth | S | continuous | ideal staircase | uniform (genuine value) | **linz single-spike** | dual-spike |
|---|---|---|---|---|---|---|
| 3 | 8  | 0.783 | 0.764 | 0.208 | **0.774** | 0.764 |
| 3 | 16 | 0.783 | 0.785 | 0.601 | **0.780** | 0.785 |
| 4 | 8  | 0.456 | 0.440 | 0.077 | **0.447** | 0.440 |
| 4 | 16 | 0.456 | 0.455 | 0.077 | **0.456** | 0.455 |

**The linearizing single-spike encode recovers ESSENTIALLY THE ENTIRE decode-linearity
residual at S≥8** (linz ≈ staircase to ±0.01), at both depths. At S=4 the grid is too
coarse (linz/dual both lose to the residual), confirming the `S+1`-level floor. So the
brief's root cause is correct AND fixable: the residual IS the `(T−τ)²` vs linear gap,
and a monotone encode warp inverts it.

## 4. DEPLOYABILITY — what is a real chip primitive (the load-bearing distinction)

A nonlinear `φ` is deployable ONLY where the value→fire-time map is a free host value
map; internal cascade neurons fire via the FIXED ramp+threshold and admit only `θ`
(scale) and bias — an **affine** value warp `v → v/g`, which is exactly the gain trim G.

| layer | knob | can express φ? |
|---|---|---|
| **encode / input** (`encoding=True` node: charge `V/θ`, fire `round(S(1−V/θ))`) | arbitrary monotone value map baked into calibration (host preprocessing) | **YES — arbitrary φ, free** |
| **internal cascade** neuron | `θ` (activation_scale), bias | only AFFINE (`v → v/g`) = G; **NOT** arbitrary φ |

`§4` measures the deployable scope: nonlinear φ at the ENCODE layer only, internal layers
on affine G, vs the oracle (φ everywhere = staircase). Multi-seed mean:

| depth | S | uniform | φ_enc only | **φ_enc + G (DEPLOYABLE)** | φ_all (oracle = staircase) |
|---|---|---|---|---|---|
| 3 | 8  | 0.208 | 0.720 | **0.741** | 0.774 |
| 3 | 16 | 0.601 | 0.767 | **0.772** | 0.780 |
| 4 | 8  | 0.077 | 0.108 | **0.435** | 0.447 |
| 4 | 16 | 0.077 | 0.292 | **0.450** | 0.456 |

**The fully-deployable single-spike recipe — nonlinear φ at the value-domain entry +
affine G on the internal layers — matches the φ-everywhere oracle within ~0.01–0.03 at
S≥8.** φ_enc ALONE is insufficient at depth (it linearizes only the entry; internal
layers still die); the COMBINATION φ_enc (decode-linearity at entry) + G (the latency
death-cascade internally) is what closes it. This is the deployable answer: **a monotone
input value-warp `w(v)=φ_k(v)/S` baked into the encode-layer calibration, composed with
the existing `ttfs_gain_correction` G.** No decode change, no extra spike → NF↔SCM parity
holds by construction.

## 5. DUAL-SPIKE — exact per-neuron linearization (the internal-deployable path)

Single-spike φ cannot be applied internally (only affine). The exact per-neuron
linearization that IS internally expressible is the DUAL-SPIKE code: two timed spikes per
neuron whose two triangular drives SUM to a near-linear decode (`§2`). Two free value
spikes (NOT one fixed anchor — anchor+value covers only the upper half-range) give a
codebook of ~`(S+1)²/2` levels with far finer, more uniform value coverage:

| S | #levels 1-spike | #levels 2-spike | 1-spike linz max err | **2-spike max err** |
|---|---|---|---|---|
| 4  | 5  | 14  | 0.213 | **0.100** |
| 8  | 9  | 39  | 0.115 | **0.056** |
| 16 | 17 | 123 | 0.060 | **0.029** |
| 32 | 33 | 424 | 0.031 | **0.015** |

End-to-end (`§3b` dual column) the dual-spike cascade ceiling = the ideal staircase at
S≥8 (and recovers MORE of the S=4 grid than single-spike at depth-3). **Spike-budget /
hardware cost:** 2 spikes/neuron (2× spike traffic) and a node that fires TWICE per
window with two independently-encoded fire times — a genuine encode + forward change (the
current substrate latches after one spike), so it is NOT free and NOT bit-exact with the
deployed single-spike HCM. It is the right lever ONLY where S cannot be raised, the grid
coarseness dominates (low S), and the 2× traffic is affordable.

## 6. THE HONEST CATCH — genuine FT already reaches the staircase on the toy (`§5`)

The residual the brief targets is the gap between the DEPLOYED recipe and the staircase.
On the toy, the deployed recipe is G init + a genuine STE fine-tune. Measured:

| depth | S | gen_cold | **plainG_ft** | staircase | dual | FT→staircase gap |
|---|---|---|---|---|---|---|
| 3 | 8  | 0.077 | **0.771** | 0.764 | 0.764 | **−0.006** |
| 3 | 16 | 0.260 | **0.872** | 0.785 | 0.785 | **−0.087** |
| 4 | 8  | 0.077 | **0.403** | 0.440 | 0.440 | +0.037 |
| 4 | 16 | 0.077 | **0.494** | 0.455 | 0.455 | **−0.039** |

**Genuine FT already reaches — and at S=16 EXCEEDS — the staircase.** This confirms the
brief's "training only APPROXIMATES it," but shows the approximation is, on the toy,
already complete: FT retrains the weights to exploit the cascade, beating a fixed-weight
perfect-linear decode. So **on the toy there is little genuine→staircase residual left
for the encode-warp to recover ON TOP OF a full genuine FT** — the linearization's value
is in (a) the COLD / weak-FT / very-deep / teacher-blend-absent regime, where it closes
the residual with ZERO training (`§3`/`§4`), and (b) as a sufficiency PROOF that the
residual is a correctable encode artifact, not capacity.

## 7. Verdict and how much of the ~3.5pp the linearization recovers

- **Of the genuine→IDEAL-STAIRCASE decode-linearity residual: ~100% recovered at S≥8**,
  cold, no training, by a DEPLOYABLE primitive — a monotone input value-warp
  `w(v)=φ_k(v)/S` at the encode layer composed with the existing G trim (`§3`/`§4`). This
  is a clean confirmation of the brief's root-cause hypothesis: the residual is the
  `(T−τ)²` ramp vs linear gap, and the encode is free to invert it.
- **Of the real-pipeline ~3.5pp (deployed-recipe → LIF): the recovery is likely SMALL**,
  because the real deployed recipe (teacher-blend genuine ramp = an "F") already supplies
  the genuine fine-tune that, on the toy, ALREADY reaches the staircase (`§6`). The
  encode-warp does not add on top of a complete FT. Its real-pipeline payoff is the
  cold / weak-FT / deeper-than-`d_max` / no-teacher-blend regimes.
- **Single-spike φ is the deployable lever (input warp + G); dual-spike is the exact
  internal lever but costs 2× spikes and a forward change.** Neither beats simply raising
  S where the budget allows (S=32 genuine ≈ staircase already).

## 8. Recommendation (next step)

Deploy the **encode-layer input value-warp** `w(v)=φ_k(v)/S` as a calibration-time
monotone value map (a per-neuron LUT / a closed-form pre-warp on the segment value-domain
entry), config-gated, composed with `ttfs_gain_correction`. It is parity-preserving
(decode unchanged), free (host preprocessing), and proven to close the cold decode
residual. Validate on the real mmixcore in the COLD / pre-teacher-blend regime (where
`§6` predicts it earns its keep); expect it to be NEUTRAL on top of the full teacher-blend
FT (which already reaches the staircase). Treat dual-spike as a documented fallback for
S-constrained deployments only.

Prototype: `experiments/linearize.py`. Run `python …/linearize.py` (cold, ~15s) or
`… ft` (+fine-tuning, ~21s).
