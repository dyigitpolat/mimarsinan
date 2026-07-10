# Cascaded TTFS first-crossing vs complete-sum: exact formalization and a guard-ramp transformation

**Question.** The cascaded (`ttfs_cycle_based`, schedule `cascaded`) forward fires each
neuron ONCE at the first threshold crossing of a causally-integrated ramp membrane,
while the analytic TTFS staircase that calibration targets is the *complete-sum*
solution. Premature firing — crossing before late/cancelling arrivals are in — is
measured to dominate ~79–94% of the cascaded rate error
(`mimarsinan_research/docs/research_artifacts_for_cascaded_ttfs_tuning/55_healthy_conversion_fast_ft.md:33-40`),
and the tier-0 `casc_collapse` group reads 0.77–0.87 or collapses
(`docs/research/findings/tier0_group_diagnoses.json`, e.g. casc_deepmlp cells at
0.7667/0.7698). This memo derives the exact discrepancy, enumerates every
transformation expressible in the EXISTING chip/mapping parameter space, proves what
each can and cannot do, and prototypes the best one against the repo's deployed kernel.

**Headline results (prototype, MNIST MLPs, zero training-loop involvement):**

- The deployed baseline reproduces the casc_collapse band: raw cascaded accuracy
  0.82/0.80/0.77 (3-layer, S=8/16/32) and 0.86/0.84/0.80 (5-layer) vs staircase
  0.91–0.95, with per-hop premature-fire fractions 0.37–0.86 that GROW with S.
- A per-neuron **threshold-ramp guard** `theta_eff(t) = theta + M_j*(S-1-t)` —
  implemented exactly by `hw_bias_j += M_j`, `theta_j += M_j*S`, i.e. two existing
  parameters — with hazard-calibrated `M_j` (mean 0.2–2.5) drives measured premature
  fractions to **0.00–0.01 at every hop** and lifts deployed accuracy by **+9 to +44 pp**
  at S≥16 (3-layer S=32: 0.61→0.89; 5-layer S=64: 0.45→0.89), restoring the
  "raising S helps" monotonicity that the baseline inverts.
- The worst-case-sound guard (`M_j = Wpos_j`) provably eliminates premature firing for
  EVERY arrival pattern, but its exact price is a per-hop resolution factor
  `S_eff = S/(1+M)`; with measured `Wpos ≈ 2.3–3.8` this is resolution-fatal
  (accuracy 0.09–0.10). This is a quantitative no-free-lunch: **inside the
  (theta, bias-ramp, weight-scale) space, adversarially-sound premature-fire
  elimination costs a (1+Wpos) resolution factor per hop** — the parameter-space
  analogue of why only schedule changes (doc 54's D1/D2 complete-sum deferral) are
  lossless.

Prototype: `/tmp/claude-1005/-home-yigit-repos-research-stuff/11224c9e-f926-4cb5-a527-2d0211f4bd25/scratchpad/m1/`
(`casc_first_crossing.py` = semantics + worst-case guard; `casc_guard_calibrated.py`
= calibrated guard sweep). The genuine forward is driven THROUGH the repo kernel
`ttfs_cycle_contribute_and_fire`; a numpy closed-form twin is asserted equal to the
kernel on random configs before use.

---

## 1. Ground-truth semantics (code-anchored)

Local cycle `t = 0..S-1` inside a core's active window `[L, L+S)`;
`ChipLatency` enforces consumer `L >= max(source L) + 1`
(`src/mimarsinan/mapping/latency/chip.py:145-175`). The executor gates fills/steps to
the window and, for the single-spike policy, diffs latched trains into single spikes
(`src/mimarsinan/models/spiking/hybrid/lif_step.py:103-106,115-148`). Because the fill
at global cycle `c` reads producer buffers written at `c-1`, a producer that fires at
its local cycle `k_p` is seen by the consumer at consumer-local time `a = k_p`:
**consumer-local arrival time = producer-local fire cycle**, and with the TTFS decode
`v = (S-k)/S` the code gives exactly `a_i = S*(1 - v_i)` on the integer grid
(`v_i = 0` ⇒ silent, `a_i = S`).

Membrane recursion (`src/mimarsinan/models/spiking/ttfs_cycle_step.py:36-40`):
`ramp += W @ spikes; memb += ramp; memb += hw_bias`, so

```
m(t) = sum_i w_i * (t + 1 - a_i)_+  +  beta * (t + 1),        beta = hw_bias      (1)
```

Fire once at `t_g = min{ t : m(t) >= theta }` with the inclusive default comparator
(`src/mimarsinan/models/nn/ttfs_cycle_kernels.py:21-23`;
`DEFAULT_THRESHOLDING_MODE = "<="`,
`src/mimarsinan/chip_simulation/spiking_semantics.py:131-132`). The output decode
latches the single spike and counts `S - t_g` inside the source window
(`lif_step.py:165-196`). The bias enters as a RAMP: a per-neuron `hw_bias` added every
active cycle (equivalently the always-on bias axon spikes once at window start,
`src/mimarsinan/models/spiking/hybrid/stage_io.py:146`), contributing `beta*(t+1)`.

**End-of-window identity.** For any arrival pattern,
`m(S-1) = sum_i w_i (S - a_i) + beta*S = S * z` with `z = sum_i w_i v_i + beta` —
the membrane is value-correct at the window end; ONLY the commit time is wrong
(re-confirming doc 54 §"The mechanism").

**Complete-sum reference** (what calibration targets; also the synchronized schedule's
per-window rule): `k* = ceil(S*(1 - z/theta_v))`, fires iff `k* < S`, decode
`(S-k*)/S` (`src/mimarsinan/chip_simulation/ttfs/ttfs_cycle_genuine.py:23-29`;
torch/numpy twins `src/mimarsinan/models/spiking/wire_semantics.py:11-37`). The
synchronized schedule realizes it by giving each latency group its own full window on
an `S x groups` timeline (`ttfs_cycle_genuine.py:47-49,66-69`) — complete information
before commitment, by construction.

The NF genuine node (`src/mimarsinan/models/nn/activations/ttfs_spiking.py:170-182`)
implements the same ramp/fire-once semantics, so everything below is trainable-through
if promoted. Chip parameters exist end-to-end: per-neuron `hardware_bias` array
(`src/mimarsinan/mapping/ir_mapping_class_emit.py:65-88`, exported
`src/mimarsinan/mapping/export/chip_export.py:68-111`, SANA-FE per-neuron `bias` attr
`src/mimarsinan/chip_simulation/sanafe/neuron_model.py:105-119`); threshold is a
per-HardCore scalar (`src/mimarsinan/mapping/packing/softcore/hard_core.py:21`,
`stage_io.py:247-250`) packed by threshold groups
(`src/mimarsinan/mapping/packing/canonical.py:96-101`).

## 2. Exact formalization of the discrepancy

### 2.1 Two entangled error mechanisms in deployed coordinates

Deployed cascaded parameters are `theta = 1`, `beta = b` (the folded value bias).
Write `A = sum_i w_i + b` (the completed ramp slope) and let `l(t)` be the completed
line — the trajectory's final segment extended to all `t`:

```
l(t) = A*(t+1) - S*(sum_i w_i - z + b).
```

(i) **Gain error.** Even when every arrival is in before the crossing, the fire time
solves `l(t) = theta`, giving decode `v_g = (z + (A - theta)/S) / A` — an affine
per-neuron distortion unless `A = 1`. This is the error family the per-layer theta
trims / DFQ bias matching already fight (docs 21, 55 pillar 1); it is NOT
first-crossing-specific.

(ii) **Timing (premature/late) error.** The crossing may occur while arrivals are
pending. This is the first-crossing-specific error and the subject of this memo.
In deployed coordinates (i) and (ii) cannot even be separated — the reference
`k* = ceil(S(1-z))` is not the crossing of `l` unless `A = 1`.

### 2.2 The exact fold (T1) that orthogonalizes them

Choose per neuron `j` (both are existing parameters):

```
beta_j  = 1 - sum_i w_ij                      (bias ramp; slope-normalizes A to 1)
theta_j = 1 + S*(1 - sum_i w_ij - b_j)        (value bias moves into the threshold)  (T1)
```

**Theorem 1 (exactness for non-premature neurons).** If no crossing occurs before the
last arrival, the fold makes the fire condition collapse to `t >= S*(1 - z)`, i.e.
`t_g = ceil(S(1-z))` — bit-identical to the ceil-staircase rule, ties included
(both are `min integer t >= S(1-z)` under `"<="`). *Proof:* with slope
`sum w + beta = 1`, `l(t) >= theta` ⇔ `(t+1) - S*sum(w) + S(z - b) >= 1 + S - S*sum(w) - S*b`
⇔ `t >= S(1-z)`. ∎

This is Stanojevic et al.'s exact ReLU→TTFS map (Nat. Commun. 2023, the doc-54
literature anchor) expressed in this pipeline's `(theta, hw_bias)` space.

**Measured pathology.** `theta_j <= 0` ⇔ `sum_i w_ij + b_j >= 1 + 1/S`, which holds
for 21–47% of hidden neurons per layer after standard conversion (measured on the
prototype MLPs: `sum(w)+b > 1` fraction per layer 0.21–0.47). Those neurons fire at
`t = 0` unconditionally — the fold is missing Stanojevic's second-regime `t_min` wait,
which the pipelined cascade cannot express. T1 alone therefore COLLAPSES in practice
(measured 0.10–0.53) even though it is exact on paper for non-premature neurons.
The guard below repairs exactly this (it makes `theta` positive again).

### 2.3 The pending-mass identity and the exact premature-fire condition

Define the pending-mass credit

```
D(t) = m(t) - l(t) = sum_{i : a_i > t} w_i * (a_i - t - 1).                       (2)
```

In T1 coordinates `l(t) - theta = t - S(1-z)`, so with `t* = S(1-z)`:

```
PREMATURE  <=>  exists t < t* :  D(t) >= t* - t.                                  (3)
firing-time error:  t* - t_g = max{ t* - t : D(t) >= t* - t },  else 0.
```

**Sign structure (inverts the naive intuition).** Pending EXCITATORY synapses
contribute `+w_i (a_i - t - 1)` to `D` — maximal (`w_i (S-1-t)`) for SILENT inputs
(`v_i = 0`, the ReLU-sparse majority). Pending INHIBITORY synapses contribute
negatively — they DELAY firing, i.e. they are protective in these coordinates. The
familiar deployed-coordinates story ("early positive ramps cross before late negative
arrivals", doc 54/55) is the same failure set seen from coordinates where `theta`
pre-counts nothing; the fold moves the accounting so that the hazard is exactly the
mass whose contribution the complete-sum rule pre-credits but the causal membrane has
not yet seen. Consequence: any guard sized by inhibitory mass alone
(`M_j = sum_{w<0}|w_ij|`, the candidate suggested by the deployed-coordinates
intuition) is NOT sound — measured: the `wneg` guard leaves up to 14% premature at
the entry hop where the same-sized `wpos` guard leaves exactly 0
(`casc_first_crossing.py` output, `T1+guard[wneg]` rows).

**Distributional form.** For arrival survival functions `S_i(t) = P(a_i > t)`:
`E[D(t)] = sum_i w_i * sum_{s > t} S_i(s)`; premature probability is
`P( sup_{t<t*} [D(t) - (t*-t)] >= 0 )` with the worst case
`D(t) <= Wpos * (S-1-t)`, `Wpos = sum_{w_i>0} w_i`. The empirical hazard is far below
the worst case because arrivals are correlated and much of the excitatory mass has
either arrived by the crossing region or is silent-and-self-cancelling; measured
per-neuron sufficient guards concentrate at 0.2–2.5 while `Wpos` is 2.3–3.8
(mean) / up to 5.5 (max).

## 3. The transformation space (what the existing parameters can and cannot do)

Realizable per-neuron membrane/threshold terms: weights give `w_i (t+1-a_i)_+` per
arrival; `hw_bias` gives the ONLY free time-dependent term, `beta*(t+1)` — affine in
`t`. Hence every expressible effective threshold is affine in `t`:
`theta_eff(t) = theta - beta_extra * t`. The full enumeration:

| mechanism | exists as | (a) preserves complete-sum time for non-premature? | (b) reduces premature? |
|---|---|---|---|
| per-neuron `theta` (via per-core scalar + threshold groups, or row-scaling fold) | `hard_core.py:21`, `canonical.py:96-101` | static trim only — fixes gain (i), cannot see time | no (a static raise trades premature vs dead one-for-one; doc 55 §2 measured this saturating) |
| per-neuron `hw_bias` ramp | `ir_mapping_class_emit.py:65-88`, `ttfs_cycle_step.py:39-40` | **T1 fold: YES, exactly (Thm 1)** — but theta≤0 pathology unguarded | alone: no |
| **guard ramp = hw_bias + theta jointly** (`theta_eff(t) = theta + M(S-1-t)`) | both above | YES up to a deterministic, invertible per-neuron affine (`(1+M)`-staircase, Thm 3) | **YES — provably zero at `M >= Wpos` (Thm 2); measured zero at hazard-calibrated M** |
| per-core / per-channel weight scales | mapper-owned matrices; precedent `node_output_shifts` (`hybrid_types.py:47-48`) | the compensation currency for the guard's affine (fold `a_j` into consumer columns, `c_j` into consumer bias) | n/a |
| arrival scheduling (per-core integer latency offsets) | `ChipLatency` (`chip.py:145-175`), `_align_shiftable_cores` | uniform delay `d` of a producer core = uniform arrival shift = value shift `-d/S` on ALL its consumers (bias-compensable, but clips the small-value tail to silence — lossy) | weak: a delayed *excitatory duplicate* of a producer (2x neurons/axons) buys a `d`-cycle inhibition lookahead, protection only against inhibitors arriving ≤ d cycles late; the guard dominates it at zero resource cost (it is the per-cycle-slope generalization) |
| complement re-encode (make all weights positive via `v -> 1-v` on inhibitory fan-ins) | not realizable | — | needs a mirrored spike time `S - a` from a fire-once producer: impossible without a second spike (this is doc 54's refuted D3 in disguise) |
| sync grid-snap / partial hold inside the cascade | not realizable | — | the window length and decode are tied to the single global `S` (`lif_step.py:50,186-189`); a hold-then-catch-up is non-affine in `t`. The guard IS the affine relaxation of the two-phase hold: `M -> inf` recovers listen-then-emit behavior but saturates the decode; D2 pays the honest `2T` window instead |
| encode-layer host value warp | free host map (doc 31) | orthogonal (fixes encode nonlinearity, not first-crossing) | no |

### 3.1 The guard ramp, exactly

Mechanism (deployed coordinates): `hw_bias_j += M_j`, `theta_j += M_j * S` implements

```
fire at first t with  m(t) >= theta + M_j * (S - 1 - t)                            (T2)
```

— the threshold starts `M_j*(S-1)` high and decays linearly to exactly `theta` at the
last cycle, so any neuron that should fire at the window end still can. In T1
coordinates use `theta_j += M_j*(S+1)` instead; then:

**Theorem 2 (soundness).** With T1 + guard `M_j >= Wpos_j`: for EVERY arrival pattern,
every fire satisfies `t_g >= S(1-z) + M_j > t*`. *Proof:* a fire at `t` means
`l(t) + M(t+1) + D(t) >= theta + M(S+1)`; by (2) `D(t) <= Wpos*(S-1-t) <= M*(S-1-t)`,
so `l(t) >= theta + M(S+1) - M(t+1) - M(S-1-t) = theta + M`, i.e. `t >= S(1-z) + M`. ∎
Zero premature fires; silent stays silent (`z <= 0` ⇒ no crossing in-window).

**Theorem 3 (what the guard does to correct fires).** When all arrivals are in by the
crossing, the guarded fire time is `ceil(S*(1 - z/(1+M)))` — the SAME ceil staircase
with threshold `(1+M)` instead of 1: a pure per-neuron value scaling `v -> v/(1+M)`,
exactly invertible downstream by scaling the consumer's weight columns by `(1+M_j)`
(0 maps to 0; no bias correction needed). With pending mass at the crossing the fire
lands in the interval `[S(1-z)+M, ceil(S(1-z/(1+M)))]` — a ONE-SIDED (attenuating,
order-friendly) residual bounded by `(z - 1/S) * (M + Wneg_pend)/(1 + M + Wneg_pend)`
in value units, absorbed in the mean by a calibrated per-neuron affine readout.

**Theorem 4 (the exact price).** The guarded neuron resolves values on the grid
`(1+M)/S`: effective window `S_eff = S/(1+M)` per hop. Worst-case `M = Wpos ≈ 2.3–3.8`
⇒ `S_eff <= S/3.3`, and the downstream decompensation multiplies the NEXT layer's
`Wpos` by up to `(1+M)`, compounding `prod_l (1+M_l)` through depth. **Corollary
(no-free-lunch):** in the existing parameter space, adversarially-sound premature
elimination costs a `(1+Wpos)` resolution factor per hop — the quantitative reason the
pipelined cascade cannot be patched losslessly by parameters (matching doc 54: only
complete-sum deferral via schedule change — synchronized D1, two-phase D2 — is
lossless, at `T*(depth+1)` or `~2T/hop` latency respectively).

### 3.2 The deployable operating point

Two calibration-time choices turn the guard family into a working transformation:

1. **Hazard-calibrated `M_j`** (per neuron): the smallest guard preventing a crossing
   at `t` is `M >= (m(t) - theta) / (S - 1 - t)` (deployed coords; `/(S - t)` in T1
   coords). Take the max over pre-target cycles per calib sample and a quantile `q`
   over samples. Measured `M_j` at `q = 0.99–1.0`: mean 0.2–2.5 — i.e. the empirical
   hazard is 2–10x below the adversarial bound.
2. **Per-neuron affine readout** `(a_j, c_j)`: least-squares fit of the guarded hop's
   decoded value against the staircase target on calib data, folded into the consumer
   (`W_col_j *= a_j`, `b += W @ c`) — the same fold family as the existing
   `node_output_shifts` pre-correction; host-side per-class affine at the output.
   Calibration proceeds layer-by-layer input→output so each fit sees the actual
   guarded upstream stream (one forward per layer; conversion-time only).

Deployed coordinates (skipping T1) turn out to be the robust default: the affine
readout absorbs the per-neuron gain diversity that T1 would have normalized, WITHOUT
inheriting T1's `theta <= 0` pathology. T1 + hazard guard is competitive at shallow
depth / high S (best 3-layer S=32/64 cell) but degrades at depth 5.

## 4. Prototype and numbers

Setup: MLPs trained on MNIST (3 epochs / 4 epochs, Adam), standard percentile
(q=0.999) activation-scale conversion, 2000 test / 4000 calib samples, float64.
Genuine hops run through `ttfs_cycle_contribute_and_fire` (repo kernel); the numpy
closed-form twin is asserted bit-equal first. `S = one hop window`; hop latency 1.

**Baseline premature-fire fractions per hop** (measured in the running cascade against
each hop's complete-sum target given its actual arrivals) — 5-layer MLP:

| S | L0 | L1 | L2 | L3 | L4 (out) |
|---|---|---|---|---|---|
| 8 | 0.37 | 0.34 | 0.18 | 0.05 | 0.00 |
| 16 | 0.51 | 0.59 | 0.58 | 0.48 | 0.15 |
| 32 | 0.57 | 0.68 | 0.71 | 0.67 | 0.33 |
| 64 | 0.60 | 0.74 | 0.86 | 0.84 | 0.51 |

Premature firing GROWS with S (more cycles to cross early) — this is why the baseline
cascade gets WORSE as S rises, inverting the intended resolution/accuracy trade.
With the hazard guard (q=0.99/1.0) every entry above drops to **0.00–0.01**.

**End-to-end accuracy** (`casc_guard_calibrated.py`; "affine" = per-neuron affine
readout, present in all rows including the control, so the deltas isolate the guard):

3-layer MLP 784-128-128-10 (float 0.9485):

| variant | S=8 | S=16 | S=32 | S=64 |
|---|---|---|---|---|
| analytic staircase (ceiling) | 0.9135 | 0.9435 | 0.9475 | 0.9470 |
| genuine baseline (raw, no affine) | 0.8235 | 0.8025 | 0.7690 | — |
| deployed baseline + affine (control) | 0.8600 | 0.7795 | 0.6095 | 0.5020 |
| deployed + const guard M=1 + affine | 0.8360 | 0.8025 | 0.8350 | 0.8420 |
| **deployed + hazard guard q=1.0 + affine** | 0.7970 | 0.8635 | **0.8915** | **0.9090** |
| deployed + hazard guard q=0.99 + affine | 0.8355 | **0.8700** | 0.8760 | 0.8890 |
| T1 + hazard guard q≈1 + affine | 0.7875 | 0.8415 | 0.8865 | 0.8920 |
| T1 + worst-case guard (M=Wpos) | 0.0875 | 0.0875 | 0.0895 | — |

5-layer MLP 784-128(x4)-10 (float 0.9555):

| variant | S=8 | S=16 | S=32 | S=64 |
|---|---|---|---|---|
| analytic staircase (ceiling) | 0.7190 | 0.9460 | 0.9510 | 0.9555 |
| genuine baseline (raw, no affine) | 0.8570 | 0.8385 | 0.7965 | — |
| deployed baseline + affine (control) | 0.6660 | 0.8340 | 0.6690 | 0.4515 |
| deployed + const guard M=1 + affine | 0.8535 | 0.7985 | 0.8060 | 0.7750 |
| **deployed + hazard guard q=1.0 + affine** | **0.8580** | **0.8580** | 0.8485 | **0.8880** |
| deployed + hazard guard q=0.99 + affine | 0.7600 | 0.8500 | **0.8835** | 0.8820 |
| T1 + hazard guard q=0.99 + affine | 0.3245 | 0.7790 | 0.8515 | 0.8710 |
| T1 + worst-case guard (M=Wpos) | 0.1035 | 0.1085 | 0.1025 | — |

Readings:

- The raw baseline lands exactly in the tier-0 `casc_collapse` band (0.77–0.87), and
  its S-trend is inverted (worse at higher S) — premature firing is the mechanism.
- The guard restores S-monotonicity (guarded accuracy RISES with S: 0.80→0.91 and
  0.86→0.89) because the guard's value cost is `~M/S` (shrinks with S) while the
  premature error it removes grows with S. The casc_collapse cells are S=16/32 — the
  regime where the guard buys the most (+9 to +28 pp over the affine control; +12 to
  +44 pp at S=64).
- The worst-case-sound guard collapses to chance — Theorem 4 measured. Sound-but-
  useless vs calibrated-and-working is the whole design space of this mechanism.
- Depth-5 S=8 note: the guarded genuine (0.858) BEATS the analytic staircase (0.719)
  — the double-ceil staircase itself floor-compounds to death at low S/depth, and the
  affine-read guarded cascade preserves more signal; the staircase is not always the
  right ceiling at low S.
- Residual gap to the staircase at S≥16 is ~5–7 pp: the one-sided pending-mass spread
  of Theorem 3 that a per-neuron affine cannot fully invert. It is order-friendly and
  concentrated where fan-in sign mixing is high, i.e. exactly the FT-recoverable shape
  (cf. doc 21's corrected-init → FT-unlock result); guard+FT is the natural next
  measurement.

## 5. Hardware realizability

- **`hw_bias += M` (per neuron):** exists on every backend — torch executor adds it
  each active cycle (`ttfs_cycle_step.py:39-40`), HCM/nevresim export carries the
  per-neuron array (`chip_export.py:68-111`), SANA-FE emits per-neuron `bias` soma
  attrs for the cascaded soma (`sanafe/neuron_model.py:120-136`,
  `ttfs_cascade_model_attributes`). No new mechanism; the guard is a positive bias.
- **`theta += M*S`:** threshold is per-HardCore (`hard_core.py:21`); the
  **constant-M-per-core variant needs zero new support** (and already delivers
  premature ≈ 0 and most of the win, e.g. 0.835 at 3L S=32). The per-neuron `M_j`
  variant needs either (a) the row-scaling fold — divide neuron j's weight row and
  `hw_bias_j` by `theta_j`, set the core threshold to 1; fire-time invariant since
  membrane and threshold scale together — which must precede weight quantization and
  costs re-quantization error; or (b) backends with per-neuron thresholds (SANA-FE
  attrs are already per-neuron; nevresim threshold is per-core in the current
  export — unverified whether per-neuron is expressible there).
- **Affine readout fold:** consumer weight-column scales + bias shift; same transform
  class as the shipped `node_output_shifts` pre-correction
  (`mapping/packing/hybrid_types.py:47-48`, applied `lif_step.py:263-283`). Output
  layer: host-side per-class affine at decode (where the `* float(T)` scaling already
  happens for the analytic path, `models/spiking/hybrid/ttfs_step.py:139`).
- **Timing/energy:** spike COUNT is unchanged (still ≤ 1 per neuron); fires move later
  within the same window; no cycle-count or schedule change; `ChipLatency` untouched.
- **Parity:** only existing parameters change, so HCM↔nevresim↔SANA-FE cross-sim
  parity is mechanically unaffected. The NF genuine node models per-neuron bias/theta
  (`ttfs_spiking.py:170-182`), so the guarded forward is trainable-through.
  CAVEAT: the analytic-staircase interpretation of the SAME parameters is no longer
  the semantic target — verification comparing analytic-vs-cascaded readouts must
  apply the affine-corrected decode, and the guard must remain a cascaded-schedule-
  only transform (`is_cascaded_ttfs`, `spiking_semantics.py:100-102`).

## 6. Integration points (if promoted; nothing here is implemented in src/)

1. **Calibration pass** (hazard `M_j` + affine `(a_j, c_j)`, layer-sequential, one
   forward per layer on calib data): belongs with the existing cascaded conversion
   calibration family (doc 55 pillar-1 seam, `match_activation_distributions` /
   `calibration_pipeline.py` in `tuning/`), gated to `is_cascaded_ttfs`.
2. **Parameter application:** per-core `threshold += M*S` and per-neuron
   `hardware_bias += M` on the packed mapping (post-packing, pre-`ChipLatency` is
   fine); affine folds into IR weights BEFORE weight quantization
   (`mapping/` emit path, `ir_mapping_class_emit.py`).
3. **Config:** one knob in `pipelining/core/deployment_plan.py` (default OFF) routed
   through the cascaded recipe in `tuning/orchestration/conversion_policy.py`.
4. **Verification:** keep the existing NF↔SCM per-neuron gates on the guarded
   parameters; add the affine-corrected readout to any analytic-vs-cascaded
   comparison; a unit lock asserting Theorem 2 (zero premature on a fixed random
   config at `M = Wpos`) and the Theorem 3 fire-time law would pin the semantics.

## 7. Honest limitations

- Prototype is MLP/MNIST with a simplified percentile conversion; mmixcore/deepcnn and
  the real pipeline's conversion are untested. That `casc deepcnn` already passes
  tier-0 is consistent with the theory (hazard scales with mixed-sign fan-in mass;
  doc 55 measured Pearson +0.60–0.75 between neg-fan-in and over-fire).
- Hazard-quantile soundness is calibration-distribution-bound (probabilistic); the
  adversarial guarantee costs `(1+Wpos)` resolution (Theorem 4) and is not deployable.
- At S=8 / shallow depth the guard can slightly regress (0.860→0.836); gate by
  S/measured hazard.
- No FT interaction measured. The residual is one-sided and localized — the shape doc
  21/55 showed to be FT-recoverable — so guard-as-init for the genuine FT is the
  highest-value next experiment.
- The affine readout fit here is least-squares on calib; the production form should
  reuse the DFQ/distribution-matching machinery already in the pipeline.

## 8. Relation to prior artifacts

- Doc 54 (`54_transformation_fixes.md:20-36`): mechanism confirmed; this memo adds the
  exact parameter-space theory (Theorems 1–4) showing WHY only schedule changes are
  lossless, and a parameter-only transformation that is measurably close (premature →
  0, +9 to +44 pp) without any schedule/forward change.
- Doc 55 (`55_healthy_conversion_fast_ft.md:33-40`): the 79–94% premature dominance is
  reproduced (per-hop 0.37–0.86); the guard is the missing calibration-only lever for
  mode (b) that §0 declared "calibration cannot touch" — it can, at the cost of the
  one-sided Theorem-3 residual instead of the two-sided scramble.
- Doc 10 (`10_decode_model_and_law.md`): `rho(S) = 1 - 1.9/S`, `d_max ≈ 0.56*sqrt(S)`
  describe the OLD latched-double-integral semantics; the current kernel is the linear
  ramp (1). The guarded effective window `S_eff = S/(1+M)` composes with any such
  depth-budget law by substituting `S -> S_eff`.
