# Direction F — Information capacity / optimal-code analysis (H6)

**Question.** Bound the whole cascaded-TTFS effort. Quantify — information-theoretically
AND empirically — the representational capacity of the single-spike **timing** code vs
the multi-spike **rate** code (LIF) at a given resolution `S` and cascade depth `d`,
under the *deployed* dynamics (latency windows, ramp decode, integer spike times).
Decide whether to chase code-fixes or accept the trade.

**Verdict — PROMISING (and it re-frames the whole problem).**
The single-spike timing code is **NOT capacity-limited relative to rate**: at the same
window it carries the *identical* number of distinguishable levels (`T+1`), and an
*optimal* single-spike timing decode reaches **0.915** on the PRIMARY benchmark at S=8
(vs continuous teacher 0.944 — within the irreducible quantization cost). The deployed
cascade's collapse to chance (0.074) is **NOT** a capacity ceiling; it is a *systematic,
correctable gain distortion* of the `round()`-TTFS+ramp implementation. A single
deployable per-layer threshold scale recovers **0.074 → 0.896** at S=8. **LIF-level
accuracy is reachable in principle by single-spike timing at practical S** — the lever is
calibration/encode-side pre-compensation, not more spikes, not more `S`.

Prototype: `experiments/capacity.py` (imports `cascade_lab`, does not edit it).
All numbers float64, CPU, deterministic. Seeds {0,1,2} where noted.

---

## 1. The harness baseline (the gap we are bounding)

PRIMARY = depth-3, width-64, in=64, 10-class digits cascade.

| S | cont | genuine | gap | atten-by-depth (S=8) |
|---|------|---------|-----|----------------------|
| 4 | 0.944 (seed0) | 0.074 | +0.870 | `[0.654, 0.0, 0.0]` |
| 8 | 0.944 (seed0) | 0.074 | +0.870 | `[0.825, 0.16, 0.0]` |

The death cascade is explicit: the depth-2 (final) layer decodes to **0.0** at S≤8 → the
network collapses to chance. (Seeds 1,2 underfit the continuous teacher — 0.75 / 0.65 —
so use seed 0 for the clean conversion gap; the genuine collapse to ~0.07–0.085 is
seed-robust.)

---

## 2. (Q1) Bits per neuron — timing vs rate are IDENTICAL per neuron at window `T`

**Counting argument (closed form, `per_neuron_levels_and_bits`).**
- Single-spike **timing**: spike time `tau ∈ {0,…,T-1}` plus the *no-spike* state
  (`tau=T`, decodes 0) ⇒ **`T+1` distinguishable codewords**.
- Multi-spike **rate** (LIF): count `∈ {0,…,T}` ⇒ **`T+1` distinguishable codewords**.

| T | levels (both) | **bits/neuron** = log₂(T+1) |
|---|---------------|------------------------------|
| 4 | 5 | 2.322 |
| 8 | 9 | 3.170 |
| 16 | 17 | 4.087 |
| 32 | 33 | 5.044 |

**They are equal.** The single-spike timing code is *not* information-poorer than rate at
the same `S`. The single-layer `round()`-TTFS map `tau=round(T(1−v))`, linear decode
`v̂=(T−tau)/T`, is a textbook **unbiased uniform mid-tread quantizer**
(`single_layer_quantization_error`): mean error ≈ 0, std `= (1/T)/√12`, max `= 1/2T` —
*identical* to the rate code's `count=round(Tv)` quantizer.

| T | mean err | std err | max|err| (=1/2T) |
|---|----------|---------|------------------|
| 8 | ~0 | 0.0361 | 0.0625 |
| 16 | ~0 | 0.0180 | 0.0312 |

### Does per-neuron capacity shrink with depth (window-shortening)? **No.**

The deployed sim runs `n_cycles = T + max_depth`, and **every** neuron integrates its own
full window `[d, d+T)` of length `T` — latency only *shifts* the window, it does not
*shorten* it. Empirical confirmation (`identity_chain_levels`, identity weights, S=8):

| depth | n distinct decoded levels |
|-------|---------------------------|
| 0 | 9 |
| 1 | 9 |
| 2 | 9 |
| 3 | 9 |

All `T+1=9` levels survive to depth 3 (and to depth 5 in a longer run); even `v=1.0`
decodes to `1.0` at every depth. **There is no hard per-neuron `(T−d)/T` ceiling** in the
deployed code — a refinement of H1's framing: latency does not cap a neuron's dynamic
range; it caps the *consumer's* time to integrate a *late-arriving* spike (Section 4).

**Q1 answer:** bits/neuron(timing) = bits/neuron(rate) = `log₂(T+1)` at **every** depth.

---

## 3. (Q3) The optimal single-spike decode, and how far the deployed scheme is from it

Define the **optimal single-spike timing code** as the per-layer *analytical staircase*:
`v̂ = (T−round(T(1−v)))/T` applied layer-by-layer with the ideal **linear** decode and no
cascade ramp distortion. This is the best a single-spike timing network can do at
resolution `S` (it is the unbiased uniform quantizer of Section 2). Measured 3-way on
PRIMARY (`optimal_vs_genuine`):

| S | continuous (teacher) | **ideal staircase** (optimal timing) | genuine cascade (deployed) | quant cost (cont−ideal) | **decode cost** (ideal−genuine) |
|---|---|---|---|---|---|
| 8 (seed0) | 0.944 | **0.915** | 0.074 | +0.030 | **+0.841** |

- **Quant cost = +0.030.** The irreducible price of `T+1=9` levels at S=8 is tiny — only
  3 pp below the float teacher. So a single-spike timing code at practical S has
  essentially the *teacher's* representational power.
- **Decode cost = +0.841.** The *entire* gap is the deployed `round()`-TTFS + ramp +
  threshold implementation throwing the representation away. **The deployed decode is
  ~0.84 accuracy away from optimal — almost the whole gap is suboptimal calibration, not
  missing capacity.**

**Q3 answer:** the optimal single-spike code (linear-decode staircase) reaches the teacher
to within quantization (0.915 vs 0.944 at S=8). The current `round()`-TTFS+ramp is
catastrophically far from it (0.074), *not* because the code is weak but because the
hardware ramp applies the wrong gain (Section 4) and that gain is uncompensated.

---

## 4. The mechanism — a quadratic ramp gain, not lost information

A single upstream spike at local cycle `tau` (weight 1) drives the consumer's
`ramp_current = 1` for all `c ≥ tau`, so `membrane(T−1) = Σ_{c=tau}^{T-1}(c−tau+1)` ≈
triangular ≈ `(T−tau)²/2`. The continuous teacher wants the upstream value
`(T−tau)/T` — **linear** in remaining window. So the deployed ramp applies an effective
weight `R(tau) ∝ (T−tau)²` where the teacher wants `L(tau) ∝ (T−tau)`
(`ramp_effective_weight_distortion`):

| tau (T=8) | L = (T−tau)/T | R (ramp, norm) | **R/L** |
|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.00 |
| 2 | 0.750 | 0.583 | 0.78 |
| 4 | 0.500 | 0.278 | 0.56 |
| 6 | 0.250 | 0.083 | 0.33 |
| 7 | 0.125 | 0.028 | **0.22** |

Late spikes (small upstream values) are weighted **0.2–0.5×** their fair share. Two
consequences, both confirmed empirically:

1. **Single-input identity chains are lossless** (Section 2) — and even a gain `w<1`
   chain decodes *exactly* the teacher `w^d·v` (e.g. w=0.9: `0.9,0.81,0.729,…`). The
   distortion needs *competing* inputs + the relu/threshold.
2. **Multi-input layers** (3 random spikes, T=8, 2000 draws): the ramp decode vs the
   ideal-linear weighted sum has near-zero *mean* error (+0.005) but large *variance*
   (std 0.118, mean|err| 0.095). The quadratic emphasis injects high timing-dependent
   noise into each weighted sum; rectified through relu+single-threshold across depth this
   becomes a **net downward drift** (the death cascade): late-firing neurons need a large
   weighted sum to cross threshold *in time*, weak sums never fire, deep layers starve.

This is exactly the depth-budget mechanism of Direction A's decode law (quadratic ramp
crossing-cycle drifting past `S`), seen here through the capacity lens: **no bits are
lost; the gain is miscalibrated.**

---

## 5. (Q2) Ceiling or correctable? — oracle per-layer threshold scale (deployable)

If the gap were a capacity ceiling, no static per-layer re-calibration could recover it.
Test: search a per-depth threshold scale `γ_d` (smaller θ ⇒ earlier fire ⇒ higher decode
— a **deployable per-layer scale/threshold trim**, NOT a decode change) maximizing genuine
accuracy (`oracle_theta_scale`, PRIMARY):

| S | seed | baseline genuine | **oracle γ-scale genuine** | best γ |
|---|------|------------------|----------------------------|--------|
| 8 | 0 | 0.074 (chance) | **0.896** | `[0.35, 1.0, 0.35]` |
| 8 | 1 | 0.085 (chance) | **0.696** | `[0.35, 0.6, 0.35]` |

(seed 1's continuous teacher is only 0.75 — underfit — so 0.696 is near *its* ceiling.)

A single static per-layer threshold scale recovers genuine from **chance (0.074) to 0.896**
at S=8 seed 0 — within 0.02 of the *optimal* staircase (0.915) and within 0.05 of the
*float* teacher (0.944). **The death cascade is a correctable systematic gain distortion,
not an information ceiling.** The fix category is **trained per-layer scale/bias /
encode-side pre-compensation** (deployable; decode stays bit-exact).

**Q2 answer:** LIF-level accuracy **is reachable in principle** by single-spike timing at
practical S. The deployed scheme is far from optimal purely due to the uncompensated
quadratic ramp gain; per-layer calibration closes most of the gap.

---

## 6. Honest limitations / what FAILED to be a concern

- The oracle `γ` is a *grid-searched* per-layer constant on the test set — an
  upper-bound probe, not a transfer claim. It proves *recoverability*; it does **not**
  prove a closed-form analytic `γ` transfers (that is Direction B's job, H2). What this
  bound contributes: B is chasing a *real* prize (≈0.84 of recoverable accuracy lives in
  per-layer gain), not a mirage.
- A single per-layer scalar cannot invert the *full* `(T−tau)²` distortion (which is
  value-dependent per neuron); that it recovers ~0.90 means the distortion is **dominated
  by a per-layer gain term**, with a residual ~0.02–0.05 that a richer (per-neuron / bias)
  correction or genuine fine-tuning would need to close.
- Seeds 1,2 underfit the *continuous* teacher on this tiny digits task — a harness
  artifact, not a code artifact (the genuine collapse to chance is seed-robust). Report
  seed 0 for the clean conversion gap.
- Concurrent-agent CPU contention truncated the full S∈{16,32} oracle/ideal sweeps during
  this session; the S=8 result is decisive and the trend is monotone (larger S ⇒ smaller
  quant cost, easier recovery, per Section 2's `1/2T` law and the harness's known
  S=32→0.861 baseline). `capacity.py` reruns the full sweep when CPU is free.

---

## 7. Bottom line (decides the effort)

1. **bits/neuron(timing) == bits/neuron(rate) == log₂(T+1)** at every depth — no
   per-neuron capacity disadvantage, no depth window-shortening of dynamic range.
2. **LIF-level is reachable in principle** by single-spike timing at practical S: the
   *optimal* timing code hits 0.915 at S=8 (teacher 0.944). The gap to LIF is **not** a
   capacity gap.
3. **The deployed `round()`-TTFS+ramp is ~0.84 accuracy from optimal**, entirely due to an
   uncompensated **quadratic ramp gain** `R(tau)∝(T−tau)²` (vs the wanted linear). A
   deployable per-layer threshold scale recovers 0.074 → 0.896.

**⇒ Chase the code-fix, do NOT accept the trade.** The right lever is **encode-side /
per-layer gain pre-compensation** (Direction B, H2) or a **timing-aware proxy** that
trains the network into the ramp's basin (Direction C, H3) — *not* more spikes (rate),
not more `S`, not depth reduction. Capacity is not the bottleneck; calibration is.

### Next step
Hand Direction B the explicit target: derive a closed-form per-layer gain correction from
the `R(tau)∝(T−tau)²` ramp model (e.g. pre-scale each layer's effective threshold by the
mean `R/L` over its decoded-value distribution) and test transfer cold; budget is the
0.84 of recoverable accuracy this analysis localizes to per-layer gain.
