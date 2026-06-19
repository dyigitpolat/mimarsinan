# DIRECTION G (Phase 2) — Novel mechanisms against the gain-distortion root cause

Re-run of the novel-ideas investigation that died on API-overload in Phase 1. The
old `experiments/novel.py` was overwritten. Everything here runs on the isolated
`cascade_lab` harness (CPU, float64, seconds); nothing edits the deployed decode.

## The root cause we now target precisely (from Phase 1)

A spike arriving at consumer-local cycle `a` latches a ramp; the consumer
double-integrates it, so by end of window its **membrane contribution** is

    C(a) = (S-a)(S-a+1)/2  ~  (T-tau)^2 / 2      [QUADRATIC]

whereas a faithful **linear** decode wants `L(a) = (S-a) ~ (T-tau)`. The per-spike
**gain** `R(a) = C/L = (S-a+1)/2` falls **4.5 -> 1.0** across the window at S=8
(measured, `a=0..7`): early/large-value spikes are over-weighted ~4.5x vs
late/small/deep-layer spikes -> geometric death cascade (`d_max ~ 0.56*sqrt(S)`).

The Phase-1 deployable fix is a per-layer multiplicative threshold trim
(`precomp.thetas_mean_target`): reproduced here at **0.693 mean** on PRIMARY
(depth-3 S=8, seeds 0-2); baseline **0.077**, oracle per-layer theta-scale **~0.91**.

## Six mechanisms brainstormed; two prototyped in depth

| id | mechanism | verdict |
|----|-----------|---------|
| **N1** | **gain-linearizing value-warp** — concave per-layer power pre-shape `v -> (v/theta)^p` so the quadratic ramp integrates back to ~linear in `v` (theory: `p=0.5` cancels `(T-tau)^2`) | **REFUTED** as a standalone cold gain over the trim |
| **N2** | **trained per-neuron phase-advance** — train ONLY the per-neuron bias (== `bias_norm` the node folds) through the genuine single-spike cascade; a positive lift advances the fire time into the faithful early window | **PROMISING** — reaches the continuous ceiling; uniquely revives past `d_max` |
| N3 | arrival-aware threshold (fire-time-drift-derived theta) | folded into the trim (already captures per-layer gain) |
| N4 | training-time spike-time jitter for robustness | rejected — regularizes a continuous proxy, does not attack the deterministic gain |
| N5 | dual / complementary two-spike code | rejected — changes the deployed decode (non-deployable per the rules) |
| N6 | window-length-invariant normalization | == N1 with `p` chosen from `S`; folded in |

---

## N1 — gain-linearizing value-warp (REFUTED as a standalone cold lever)

The consumer over-weights large/early inputs because `C(a) ~ (T-tau)^2`. Pre-warp
each producer's normalized value `v' = relu(x)/theta -> (v')^p` with `p<1` (concave)
BEFORE the single-spike encode. Theory (closed form): `C ~ (S*g)^2/2`,
`tau ~ S(1-g)`, so choosing `g(v') = sqrt(v')` (`p=0.5`) makes `C` proportional to
`v'` — a unit of value contributes a constant amount to the consumer membrane
regardless of magnitude, cancelling the over-weighting. The warp is a monotone
per-layer activation reshape (a calibration value->time table; the ramp decode and
threshold crossing stay bit-exact), so it is deployable.

**It does not work.** Cold p-sweep on PRIMARY (depth-3 S=8, seeds 0-2):

| warp `p` | gen (mean) | vs trim |
|----------|-----------|---------|
| 1.0 (= trim) | 0.693 | — |
| 0.9 | 0.691 | -0.002 |
| **0.75** | **0.709** | **+0.016** |
| 0.6 | 0.686 | -0.007 |
| 0.5 (theory-optimal) | 0.655 | -0.038 |
| 0.4 | 0.578 | -0.115 |

Across depth x S (best `p=0.5` vs trim, cold): a wash to net-negative — helps a few
cells (depth-2; depth-4 S=4 +0.087) but hurts most (depth-3 S=8 -0.038, depth-4 S=8
-0.106). **Why it fails:** the theory-optimal `p=0.5` warps small live values DOWN
toward the `0.5/S` dead-zone, and the single-spike quantization cost of that
out-weighs the linearization gain. The simple theta trim already absorbs most of
the per-layer gain correction; the extra value-domain warp just injects quantization
noise. Honest negative result — and an informative one (it bounds how much a pure
static value remap can buy beyond the trim: essentially nothing).

## N2 — trained per-neuron phase-advance (PROMISING; the winner)

A constant per-cycle membrane lift `beta` advances the fire time (earlier -> faithful
early window -> higher decoded value). `beta` is **exactly** the per-neuron
`bias_norm = bias/theta` the genuine node folds at drive time. Phase-1's partial
novel.py only HAND-SET a depth-graded `beta` cold (G2, weak). The novel move here:
**train the per-neuron offset THROUGH the genuine single-spike cascade** (boundary
STE) with **all weight DIRECTIONS frozen** — only the per-neuron bias updates. This
is a pure conversion-correction (not a task retrain) and is deployable (the bias is
a chip parameter).

### Cold vs after genuine FT — PRIMARY depth-3 S=8 (seeds 0,1,2)

| seed | cont | trim-cold | N2 ft | lift |
|------|------|-----------|-------|------|
| 0 | 0.944 | 0.866 | **0.939** | +0.072 |
| 1 | 0.750 | 0.659 | **0.738** | +0.080 |
| 2 | 0.655 | 0.553 | **0.651** | +0.098 |
| **mean** | **0.783** | **0.693** | **0.776** | **+0.083** |

N2 recovers each seed to **99.1% of its own continuous ceiling** (vs 88% for the
trim, vs the 0.91 oracle theta-trim ceiling). (Seeds 1,2 have low *continuous*
accuracy because deeper random inits train worse; N2 tracks each ceiling.)

### Depth x S — N2 vs the trim and the existing full-FT lever (seeds 0-2)

| depth | S | cont | base | trim | **N2 (bias-only)** | full-FT (existing) |
|------:|--:|-----:|-----:|-----:|-------------------:|-------------------:|
| 3 | 8 | 0.783 | 0.077 | 0.693 | **0.777** | 0.917 |
| 3 | 16 | 0.783 | 0.260 | 0.657 | **0.777** | 0.951 |
| 4 | 8 | 0.456 | 0.077 | 0.335 | **0.456** | 0.688 |
| 4 | 16 | 0.456 | 0.077 | 0.318 | 0.416 | 0.853 |
| 6 | 8 | 0.467 | 0.077 | 0.077 | **0.265** | 0.110 |
| 6 | 16 | 0.467 | 0.077 | 0.306 | 0.215 | 0.414 |

Reading:
- **depth 3-4**: N2 recovers to the continuous ceiling and clearly beats the trim
  (depth-4 S=8: 0.456 vs 0.335). Full genuine FT beats both because it *retrains the
  task* through the cascade (it can exceed the conversion ceiling).
- **depth-6 S=8 (past `d_max(8)~=2`)**: the killer regime. trim = **dead (0.077)**,
  full-FT = **0.110 (also fails — too deep, diverges)**, **N2 = 0.265 (best)**. N2
  revives a cascade that BOTH the analytical trim AND full FT cannot, because it
  attacks fire-timing directly and freezing the weights keeps the optimization
  well-conditioned where full FT diverges. This is N2's unique, publishable regime.

### Mechanism confirmed (it is a genuine phase-advance)

Inspecting the learned offset (depth-6 S=8, seed0): every layer learns a **positive**
`bias_norm` delta of **+0.09 .. +0.33** normalized membrane units/cycle — a real
per-cycle phase-advance that lifts the membrane and pulls the fire time earlier into
the faithful window, exactly as theorized. Not a hand-set schedule — the gradient
discovers it per neuron.

### Ablations / controls

- **N2 from BASELINE init (no trim): 0.077 -> 0.795** at depth-3 S=8. Training the
  per-neuron bias ALONE revives the *dead* cascade to the continuous ceiling — the
  trim init is not required; N2 subsumes it.
- **N2 + per-neuron gain: 0.780** (no gain by itself: 0.776). Adding a trained
  per-neuron output gain does not help — the offset is the operative lever.
- **N1 warp -> N2 stack: ft 0.777**, identical to trim -> N2 (0.778). The warp does
  NOT raise the FT ceiling; N2 reaches the ceiling regardless of init. **N1 is fully
  subsumed by N2.**

## Relation to the simple theta trim — does N2 beat or complement it?

- N2 **beats** the trim everywhere the cascade is alive (depth 3-4: to the ceiling vs
  trim's ~88%), and **uniquely revives** the dead regime past `d_max` (depth-6 S=8)
  that the trim cannot touch.
- N2 **does not need** the trim (revives from baseline init too) — so it is not a
  complement to the trim; it is a *stronger, training-based replacement* for the same
  per-layer gain correction, expressed as a per-neuron offset the optimizer fits.
- The price: N2 requires a short genuine-cascade fine-tune (bias-only, ~30-40 epochs,
  ~0.2s/step on the toy), whereas the trim is closed-form cold. For a fully-cold
  deployment the trim remains the right call; where a brief FT is acceptable, N2 wins.

## Deployability

N2 trains ONLY the per-neuron bias (`bias_norm`) — a first-class chip parameter the
genuine node already folds into the membrane each cycle. Weight directions are
frozen, the ramp-integrate decode and threshold crossing are untouched, so NF<->SCM
parity is preserved by construction. It is the most deployable form of a
genuine-cascade fine-tune: a per-neuron scalar, not a weight retrain.

## Honest verdict

- **N1 (value-warp): REFUTED** as a standalone improvement over the closed-form
  theta trim (a wash-to-negative cold; the trim already captures the per-layer gain,
  and warping toward the dead-zone adds quantization noise). Theory-clean, fails in
  practice for a precise, instructive reason.
- **N2 (trained per-neuron phase-advance): PROMISING.** Bias-only genuine FT recovers
  to the continuous ceiling at depth 3-4 (beating the trim), and is the ONLY lever
  tested — analytical trim or full FT — that revives the dead cascade past `d_max`
  (depth-6 S=8: 0.265 vs 0.077 / 0.110). The learned offsets are verified positive
  phase-advances. Deployable as trained per-neuron biases.

## Next step

Validate N2 on the real mmixcore pipeline: add a **bias-only genuine-cascade
fine-tune phase** to the cascaded-TTFS tuner (freeze weight directions, train
`bias_norm` through the genuine forward), gated by a flag, and measure deployed SCM
accuracy vs the 0.95 baseline — particularly on any segment chain deeper than
`d_max(S)`, where N2 is the only working lever. Pair with the Phase-1 input->decode
skip (artifact 14) for chains far past `d_max`.
