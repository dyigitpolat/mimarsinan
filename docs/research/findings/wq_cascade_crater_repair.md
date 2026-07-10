# Why 5-bit weight quantization craters the genuine cascaded-TTFS forward — and the closed-form repair

**Date:** 2026-07-10.
**Question.** The 5-bit WQ projection collapses the genuine (cascaded, fire-once)
TTFS forward — campaign t01_18 casc lenet5 S=16: FT exit 0.9808 → WQ entry read
0.6689 (bbce29c3 stdout:243,264-267); t0_17 casc lenet5 S=32: 0.94 → 0.62 —
while the SAME projection family costs the value-trained cells ±1 pp (ttfsq
t0_11 raw WQ projection −0.64 pp) and casc deepcnn nothing
(`docs/research/findings/tier0_group_diagnoses.json`, casc_collapse group). The
WQ endpoint recovery then trains DOWNHILL to chance in ~2000 steps and keep-best
clamps exit=entry; the retention gate trips INSIDE Weight Quantization
(0.6813 < 0.8338). This memo derives the crossing-time perturbation calculus
that explains the asymmetry, locates where the 5-bit error actually
concentrates on the real vehicle, and derives + prototypes two closed-form,
training-free repairs.

**Headline results.**

- The WQ install re-solves NOTHING for the cascaded semantics: it rounds the
  effective (θ-normalized) weights AND bias onto one shared per-perceptron
  max-abs grid and stamps `parameter_scale`; θ, wire scales, and biases'
  sub-grid DFQ corrections are never re-solved (§1).
- First-crossing calculus (§2): a weight/bias perturbation moves the fire time
  by `Δt* = −[Σ_A δw̃·elapsed + δb̃·(t*+1) − δθ]/S` — amplified by the
  reciprocal of the PARTIAL crossing slope `S` (small/mixed-sign at the greedy
  operating point: measured median full-sum slopes are NEGATIVE at every hop),
  by elapsed-time factors up to T (the bias error is worth `δb̃·T`, matching
  S=32 cratering deeper than S=16), and made O(1)-discontinuous by the
  fire-once latch. The complete-sum staircase passes the same perturbation
  once through a 1-Lipschitz composition — bounded, no crossing-slope divisor.
- Grid forensics on the real lenet5 FT artifacts (§3): the shared (w,b) grid is
  SET BY THE BIAS on both fc perceptrons (|b̃|max ≈ 2–8× |w̃|max); the median
  weight gets 0.13–0.32 grid levels, **58–95 % of fc weights round to exactly
  zero**, and the median per-neuron slope-sum error reaches |ΔS/S| = 0.95. The
  sub-grid DFQ bias corrections that bought the float cascade's health
  (distmatch 0.9377→0.9680, bbce29c3:213) are rounded away.
- Repairs (§4), both closed-form, zero training:
  1. **Two-scale projection** (weight grid from max|w̃| alone; bias on its own
     5-bit grid): recovers float EXACTLY on every substrate measured — real
     artifact A 0.9440→0.9685 (float 0.9675), real artifact B 0.9600→0.9645
     (float 0.9645), controlled 7-hop cascade-FT MLP 0.9121→0.9531 (float
     0.9517, T=16) and 0.8855→0.9318 (float 0.9331, T=32).
  2. **Crossing-interval θ̂ re-solve** (per-neuron threshold matching the float
     firing time on calibration statistics): the general post-projection law;
     restores timing systematics (fc1 mean|Δt| 3.92→3.47 cycles on artifact A,
     1.20→0.80 on B) and is worth +0.1…+0.9 pp on the joint grid, but is
     feasibility-bounded: channels whose quantized ramp is non-positive at the
     float crossing (66/120 on artifact A's fc1 — the §3 annihilation class)
     admit no positive threshold. The naive sum-ratio rescale
     `θ·Σŵ/Σw` is exact only for degenerate arrival patterns and measured
     ACTIVELY destructive (0.95→0.50) — active sets are input-dependent.
- The two are one law at two levels: the calculus prices every error in
  grid-steps-times-elapsed-cycles; two-scale shrinks the grid step where the
  price is highest (the bias row), θ̂ absorbs what remains at the crossing.

Prototype (nothing under `src/` was modified):
`/tmp/claude-1005/-home-yigit-repos-research-stuff/11224c9e-f926-4cb5-a527-2d0211f4bd25/scratchpad/m2/`
— `artifact_repair.py` (real lenet5 artifacts, hook-instrumented, framework
forward + framework NAPQ class), `proto2.py` (controlled 7-hop cascade-FT MLP,
kernel-faithful reimplementation, sanity-asserted against the differentiable
form), results JSONs alongside.

---

## 1. What the WQ install actually does — and what its reads see

**The install.** `WeightQuantizationStep.process`
(`src/mimarsinan/pipelining/pipeline_steps/quantization/weight_quantization_step.py:37-57`):
LIF half-step fold (LIF-only, no-op here, :59-75), guarded starved-bias
canonicalization (:77-95), `compute_per_source_scales` refresh (:42), norm
freeze (:43-49), then the NAPQ tuner. The transform itself
(`src/mimarsinan/transformations/normalization_aware_perceptron_quantization.py:30-64`)
operates on the **effective** parameters — the θ-and-source-scale-normalized
chip coordinates: `get_effective_weight = per_input_scales · w · u / activation_scale`
and `get_effective_bias = ((b−μ)u+β)/activation_scale`
(`src/mimarsinan/transformations/perceptron/perceptron_transformer.py:106-124`).
It clips OFF-saturated bias (:34), takes ONE scale per perceptron over weight
AND bias, `scale = q_max / max(|w̃|,|b̃|)` (:39-46), stamps
`parameter_scale` (:48), and rounds/clamps BOTH w̃ and b̃ on that grid
(:54-64). The recipe's rungs are projection-only — `wq_fast_rates=[0.5,1.0]`,
`wq_fast_steps_per_rate=0`
(`src/mimarsinan/tuning/orchestration/conversion_policy.py:16-20`;
`src/mimarsinan/tuning/tuners/normalization_aware_perceptron_quantization_tuner.py:33-38`).

**The reads.** The gate's entry read is `full_transform_acc_on_clone` — the
rate-1.0 projection FORCED on an isolated clone
(`src/mimarsinan/tuning/orchestration/mbh_gate.py:89-99` →
`src/mimarsinan/tuning/orchestration/mbh_ledger.py:170-182` →
`src/mimarsinan/tuning/orchestration/fast_ladder.py:249-267`, replica apply
`normalization_aware_perceptron_quantization_tuner.py:57-63`) — and every rung's
`full_acc` is the same full-transform read. Evaluation goes through
`trainer.model(x)` (`src/mimarsinan/model_training/basic_trainer_eval.py:154-168`),
and for cascaded mode `model.forward` is the genuine single-spike segment
cascade kept installed at FT finalize
(`src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py:586-589`,
`_SegmentSpikeForward` :42-63 → `TTFSSegmentForward`
(`src/mimarsinan/models/spiking/training/ttfs_segment_forward.py:22-58`) →
`SegmentForwardDriver` + `TtfsSegmentPolicy`). **So the probes DO read the
deployed cascade** — the campaign's three identical reads
(entry 0.668872, rate 0.5 0.668872, rate 1.0 0.668872, bbce29c3:264-267) are
three evaluations of the SAME full projection, not evidence that rate mixing is
inert.

**What is never re-solved.** Nothing per-neuron and nothing timing-aware: no
θ (`activation_scale`) re-solve, no DFQ bias pass, no boundary re-propagation
anywhere in the step or the transform. Contrast the conversion install, whose
cascade health is explicitly calibrated: θ from the teacher's activation
quantile + iterated DFQ per-neuron bias correction through the deployed cascade
(`src/mimarsinan/spiking/distribution_matching.py:140-162`,
`src/mimarsinan/spiking/dfq_bias_correction.py:92-172`,
`src/mimarsinan/spiking/lif_distribution_matching.py:31-57`). The endpoint
stage that follows the projection only TRAINS
(`normalization_aware_perceptron_quantization_tuner.py:69-88` →
`src/mimarsinan/tuning/orchestration/frontier/endpoint_recovery.py:58`), which
the campaign measured as actively destructive from a cratered entry (val
0.668→0.09-chance over 2000 steps; keep-best clamps exit=entry;
`tier0_group_diagnoses.json`, casc group, t01_18 row).

## 2. The firing-time perturbation calculus

**Semantics** (deployed effective coordinates; matches
`src/mimarsinan/models/nn/activations/ttfs_spiking.py:158-183`,
`src/mimarsinan/models/spiking/ttfs_cycle_step.py:36-44`,
`src/mimarsinan/models/nn/ttfs_cycle_kernels.py:10-26`, window gating
`src/mimarsinan/spiking/segment_policy_ttfs.py:251-273`; same form as
`casc_first_crossing_transformation.md` §1 eq. (1)):

```
m(t) = Σ_j w̃_j · (t + 1 − a_j)_+  +  b̃ · (t + 1),      fire once at
t* = min{ t : m(t) ≥ θ },        decode v = θ_dec · (T − t*)/T.
```

`a_j` = arrival cycle of axon j (single spike), `w̃, b̃` = the effective
parameters — exactly what NAPQ rounds. The encoding perceptron is analytic
(value-domain compute, fires at `ceil(T(1−v/θ))`; `ttfs_spiking.py:143-156`),
so hop 0's quantization error is value-typed; hops ≥ 1 are ramp integrators.

**First-order law.** Perturb (w̃, b̃, θ) → (w̃+δw, b̃+δb, θ+δθ) and
differentiate the crossing condition. With `A = {j : a_j ≤ t*}` (the ARRIVED
set) and crossing slope `S = Σ_A w̃_j + b̃`:

```
Δt* = −[ Σ_A δw_j (t*+1−a_j)  +  δb (t*+1)  −  δθ ] / S            (1)
Δv  = (θ_dec /(T·S)) · [ Σ_A δw_j (t*+1−a_j) + δb (t*+1) − δθ ]     (2)
```

Three amplifiers, none of which exist in the value domain:

- **1/S crossing-slope gain.** `S` is the PARTIAL sum over arrived axons plus
  bias — not the trained value-Jacobian. Measured on the real artifact
  (t0_17 FT state), the FULL-sum slopes `Σ_j w̃_j + b̃` have NEGATIVE medians at
  every hop (−0.23 / −2.56 / −1.28 / −0.57): the FT solution parks most neurons
  in the greedy regime — they fire on early positive mass and would never fire
  on the complete sum (the premature-firing regime formalized in
  `casc_first_crossing_transformation.md` §2.3). Such neurons' function has no
  value-domain equivalent, and their crossing slopes are small positive partial
  sums: high 1/S gain by construction.
- **Elapsed-time weighting.** Each δw_j is scaled by its elapsed ramp cycles
  (≤ T) and δb by (t*+1) ≤ T: **rounding the bias to grid step g injects up to
  (g/2)·T of membrane error**, versus a single δb in the value forward. This
  is why the S=32 cell (t0_17, 0.94→0.62) craters deeper than the S=16 cell
  (t01_18, 0.98→0.67) under the identical projection.
- **Fire-once discreteness.** The latch (`ttfs_cycle_kernels.py:20-26`) makes
  v(w̃) piecewise-constant with O(θ/T) jumps at every crossing-cycle boundary
  and an O(θ) jump at the fire/never-fire boundary (greedily ignoring
  everything after the crossing; window exit ⇒ decoded 0). The 5-bit
  perturbation is a large structured step (half a grid per element,
  correlated), not a within-basin wiggle: measured per-hop fire-flip rates at
  5 bits are 10–16 % on the real vehicle, mean |Δt*| up to 3.9 of 32 cycles.

**Hop compounding.** Inside a segment, hop k's output-time shift IS hop k+1's
arrival shift: `Δm_{k+1}(t) = Σ_j w̃_j Δa_j` — per-hop gain ~ Σ|w̃·Δa|/margin,
routinely > 1 in deep intra-segment chains (the 9-hop mmixcore). Segment
boundaries (lenet5 = 3 segments of intra-depth 1, bbce29c3:212; deepcnn sched
likewise) re-anchor times to decoded values
(`segment_policy_ttfs.py:165-183, 275-288`) and stop TIMING compounding, but
each hop's crossing amplification (1)–(2) still applies per hop.

**Why the complete-sum staircase is immune.** The analytic forward
`v = θ·stair(clamp(relu(w̃ᵀx + b̃)))`
(`src/mimarsinan/models/spiking/wire_semantics.py:26-38`) passes δ ONCE through
a 1-Lipschitz composition plus a ≤ θ/T rounding: no 1/S divisor, no elapsed
factors, no latch. Campaign check on the same projection family: ttfsq WQ
deltas are ±1 pp (t0_11 raw projection −0.64 pp) vs −31 pp for casc lenet5 at
identical vehicle/bits (`tier0_group_diagnoses.json`).

## 3. Where the 5-bit error concentrates: the bias-set shared grid

Grid forensics on the two real casc-lenet5 FT artifacts (effective parameters,
exactly what NAPQ rounds; `q_max = 15`):

Artifact A = `generated/t0_17_casc_lenet5_wqaq_s32_phased_deployment_run/TTFS Cycle Fine-Tuning.model.pt`;
Artifact B = `generated/t01_18_casc_lenet5_wq_s16_phased_deployment_run/…` (local reproduction, 2026-07-10).

| perceptron | grid set by | |w̃|max | |b̃|max | median w levels | weights → 0 | median \|ΔS/S\| |
|---|---|---|---|---|---|---|
| A features_0_full (enc) | weight | 0.840 | 0.190 | 3.33 | 8.7 % | 0.21 |
| A features_3_full | weight | 0.379 | 0.191 | 2.54 | 11.5 % | 0.011 |
| A classifier_0 (fan-in 784) | **BIAS** | 0.093 | **0.734** | **0.13** | **94.7 %** | **0.95** |
| A classifier_2 | **BIAS** | 0.184 | 0.500 | 0.32 | 64.9 % | 0.12 |
| B classifier_0 | **BIAS** | 0.149 | 0.362 | 0.17 | 74.7 % | 0.26 |
| B classifier_2 | **BIAS** | 0.181 | 0.224 | 0.32 | 57.9 % | 0.10 |

(ΔS = quantization-induced error of the per-neuron slope sum Σ_j w̃_j + b̃.)

The fc perceptrons' effective bias — inflated by the θ-normalization and by the
DFQ corrections that repaired the float cascade — is 2–8× the largest weight,
so it sets the shared grid; the per-input weights of a 784-fan-in row are tiny
by construction, land below half a grid step, and are annihilated en masse.
This is the LIVE-bias generalization of the OFF-channel starvation NAPQ
already guards against (`normalization_aware_perceptron_quantization.py:31-34`,
`src/mimarsinan/transformations/perceptron/bias_saturation.py`,
`bias_canonicalization.py` at the step entry) — those clips only reach
functionally-unobservable bias mass; here the bias is live. It is also the WQ
sibling of the AQ scalar-θ starvation diagnosed for the mixer
(`mixer_column_scale_pathology.md` §"Per-hop scalar-θ grid starvation"): one
shared scale spanning magnitudes that differ by decades.

**Why some draws crater and others don't.** The campaign configs' `seed` is
never applied to torch (`numerical_boundary_consistency.md` §"missing global
seed"), so every run trains a different network. Both locally-available lenet5
draws carry the same structural fragility (table above) yet lose only
0.5–2.4 pp under the joint projection, because their function does not ride the
annihilated mass; the campaign draw (FT exit 0.9808 → 0.6689) sat in the tail
where it does — its float health was visibly bought by sub-grid DFQ bias moves
(distmatch probe 0.9377 → 0.9680, bbce29c3:213) that the (g/2 ≈ 0.02–0.05)
grid rounds away with a T-amplified membrane effect (§2). A local end-to-end
re-run of t01_18 (same config, different draw) went 0.9714 → 0.9788 through WQ
— crater absent, structure present. The repair below therefore targets the
structure, not one draw.

## 4. The repair

### 4.1 Exact scale-space identity (and why the naive ratio fails)

The membrane is linear in (w̃, b̃), so `(w̃, b̃, θ) → (αw̃, αb̃, αθ)` leaves
t* invariant for EVERY input. Hence any per-neuron multiplicative error
component of quantization is exactly cancellable by `θ ← αθ` alone — and since
firing TIMES are restored, downstream wire/decode interpretation stays valid
without touching any other parameter. `θ̂ = θ·(Σŵ̃+b̂̃)/(Σw̃+b̃)` (the
sum-ratio) is exact iff the perturbation is a pure common rescale, or all
arrivals are simultaneous (only the slope matters). On real inputs neither
holds — the arrived set is input-dependent (ReLU-sparse: most axons never
spike) — and the full-sum ratio is measured ACTIVELY destructive on the
controlled prototype (0.9517 → 0.5398 at T=16; 0.9331 → 0.5041 at T=32).
Least-squares α (`⟨ŵ,w⟩/‖w‖²`) is safe but negligible (+0.2 pp): rounding has
no systematic per-row gain; the systematic error lives at the crossing.

### 4.2 The crossing-interval θ̂ law (general closed form)

The float first crossing at cycle t* means `A(t*−1) < θ ≤ A(t*)`. Reproducing
t* under the quantized parameters requires

```
θ̂  ∈  ( Â(t*−1),  Â(t*) ],      Â(t) = Σ_j ŵ̃_j (t+1−a_j)_+ + b̂̃ (t+1)   (3)
```

with Â evaluated under the FLOAT arrivals a_j. Per calibration event take the
interval midpoint; per output channel take a robust statistic (median wins over
mean/quantiles in all measurements). Expanding (3):

```
θ̂_i = θ_i + stat[ Σ_A δw_j (t*+1−a_j) + δb (t*+1) ]                  (4)
```

— an additive threshold correction equal to the expected quantization-induced
membrane excess at the float crossing. It subsumes §4.1 (δ ∝ (w̃,b̃) recovers
θ̂ = αθ exactly), absorbs the T-amplified bias-rounding term at the per-channel
mean crossing, and needs NO gradients: one calibration forward of the float
cascade recording arrivals and fire times, plus one linear pass with the
quantized parameters. Two structural boundaries, both measured:

- **Discreteness:** matching at `Â(t*)` alone (no interval) inflates θ̂ by the
  mean per-cycle overshoot (+25 % measured on the first prototype iteration and
  its accuracy REGRESSED) — the interval form is required.
- **Feasibility:** events with `Â(t*) ≤ 0` admit no positive threshold — the
  quantized ramp is annihilated at the float crossing. On artifact A's
  classifier_0, 66/120 channels are infeasible: exactly the §3 starvation
  class. θ̂ is a TIMING repair; it cannot restore destroyed slope mass.

### 4.3 The projection-side repair: two-scale grids

Equation (2) prices every quantization error in grid steps × elapsed cycles /
crossing slope; §3 shows the grid step g itself is inflated 2–8× by one row
(the bias), whose own rounding error is then T-amplified. The repair one level
up is therefore: **weight grid from max|w̃| alone; bias on its own 5-bit
grid** (`scale_w = q_max/max|w̃|`, `scale_b = q_max/max|b̃|`). Same bit budget,
closed-form, zero training.

### 4.4 Measured results

Genuine cascaded forward, 2000-sample MNIST test reads, 5 bits everywhere.
Real artifacts run THROUGH the framework (`TTFSSegmentForward`, framework NAPQ
class); the controlled MLP is a kernel-faithful reimplementation (7 hops,
width 128, BN-folded effective params, cascade surrogate-FT, θ from the 0.99
activation quantile per `distribution_matching.py:140-144`), whose
hard/differentiable forwards are assert-identical.

| substrate | float casc | joint NAPQ (shipped) | + θ̂ (4.2) | two-scale (4.3) | two-scale + θ̂ |
|---|---|---|---|---|---|
| artifact A (lenet5, S=32) | 0.9675 | 0.9440 | 0.9410 | **0.9685** | — |
| artifact A, bias kept float (control) | | | | 0.9680 | |
| artifact B (lenet5, S=16) | 0.9645 | 0.9600 | 0.9565 | **0.9645** | — |
| MLP 7-hop, T=16 | 0.9517 | 0.9121 | 0.9131 | **0.9531** | 0.9509 |
| MLP 7-hop, T=32 | 0.9331 | 0.8855 | 0.8944 | **0.9318** | 0.9311 |
| MLP 7-hop T=16, θ̂ = sum-ratio (control) | | | 0.5398 | | |
| MLP 7-hop, healthy grid (no BN fold), T=16 | 0.9543 | 0.9513 | 0.9536 | — | — |

Readings:

- **Two-scale recovers float exactly everywhere** — the bias-kept-float control
  shows the two-scale bias grid loses nothing further, i.e. at 5 bits the
  ENTIRE measured WQ damage on this vehicle is the shared-grid starvation
  channel, in the timing domain and (measured on artifact A: value-forward
  0.9675→0.9435 under the same projection) in the value domain alike.
- **θ̂ restores the timing systematics** — artifact A fc1 mean|Δt*| 3.92→3.47,
  fc2 2.59→2.20 cycles; artifact B fc1 1.20→0.80 — and is worth +0.1…+0.9 pp
  on the joint grid, bounded by the feasibility class (it cannot repair the
  55 % of fc1 channels the joint grid annihilates). On a healthy grid, per-hop
  crossing errors are sub-cycle (mean|Δt| 0.26–0.50) and the cascade absorbs
  the projection: the healthy-grid MLP row loses only 0.3 pp, which is why
  crater-vs-no-crater tracks the §3 forensics, not depth alone.
- The controlled substrate reproduces the crater's dependence on T (−4.0 pp at
  T=16 vs −4.8 pp at T=32 from the same weights) — the δb̃·(t+1) term of (2).

## 5. Where it integrates (no `src/` changes made here)

**Two-scale projection (primary).**
`NormalizationAwarePerceptronQuantization._transform`
(`normalization_aware_perceptron_quantization.py:35-64`): compute the weight
scale from `w_max` alone, keep `parameter_scale = q_max/w_max` (:46-48), and
quantize the bias in `apply_effective_bias_transform` with its own
`q_max/b_max` scale. Downstream contract points:
- `hardware_bias` is already a SEPARATE per-neuron array on the weight bank
  (`src/mimarsinan/mapping/ir_mapping_class_base.py:150-157`,
  `src/mimarsinan/mapping/ir/types.py:28,100`) — but chip export currently
  quantizes it with the SAME scale and bounds as the weights
  (`src/mimarsinan/mapping/export/chip_quantize.py:78-85`); the second scale
  must be carried there and in `verify_quantization`
  (`src/mimarsinan/transformations/chip_quantization.py:8-21`;
  `src/mimarsinan/pipelining/pipeline_steps/quantization/quantization_verification_step.py:25-34`).
- Backend audit: SANA-FE emits per-neuron float `bias` soma attributes (see
  `casc_first_crossing_transformation.md` §5); for a backend whose contract
  truly forces one grid, the bias decomposes into `ceil(|b̃|·scale_w/q_max)`
  always-on grid axons (lenet5 cores run at 61 % axon waste — headroom exists).

**θ̂ re-solve (secondary / post-hoc knob).** Belongs INSIDE the WQ install, per
projection: after the transform in `_apply_rate` / `_apply_rate_to`
(`normalization_aware_perceptron_quantization_tuner.py:52-63`) or once at the
rate-1.0 commit, BEFORE `_post_stabilization_hook` (:69-88) so endpoint
recovery trains from the repaired state. Reuse map:
- calibration reads through the deployed forward with a recorder side-channel,
  like `_cascade_channel_means` / `_lif_cascade_channel_means`
  (`distribution_matching.py:83-94`, `lif_distribution_matching.py:14-28`),
  extended to record per-hop arrivals + fire cycles (the prototype does this
  with forward hooks only);
- the per-channel θ container is `promote_activation_scale_per_channel`
  (`src/mimarsinan/spiking/theta_cotrain.py:19-40`); the mutation precedent is
  `apply_gain_at_rate` (`src/mimarsinan/spiking/gain_correction.py:86-97`).
- **Coupling hazard (load-bearing):** `perceptron.activation_scale` is BOTH the
  spike-node threshold and the denominator of the effective weights
  (`perceptron_transformer.py:106-113`) and the boundary decode/wire scale
  (`segment_policy_ttfs.py:70-94`). Mutating it after the projection would
  silently move the just-quantized weights off-grid and rescale decoded
  values. The repair must therefore rebind ONLY the spike-node threshold
  (`TTFSActivation.activation_scale` on the node instances — the prototype's
  `apply_threshold_repair`), and must NOT call
  `propagate_boundary_input_scales` afterwards: restored firing times mean the
  old wire/decode interpretation is the correct one.
- Hardware: per-neuron thresholds are native in nevresim
  (`nevresim/include/simulator/chip_weights.hpp:15-19`,
  `NeuronWeights.threshold_`); the current export writes one scalar
  `node.threshold = scale` (`chip_quantize.py:48-50,62`) — a per-neuron
  threshold vector `scale·θ̂_norm` is the θ̂-ready generalization.
- Config: one knob, default OFF, routed through the cascaded/WQ recipe knobs in
  `conversion_policy.py:16-20,90-107`.

## 6. Genericity and limitations

- **Vehicle/dataset-agnostic.** Both repairs use only the fire-once ramp
  semantics and the per-perceptron grid — no workload constants. Conv
  perceptrons are handled per output channel (the prototype instruments the
  real conv lenet5 through the functional-conv node path,
  `src/mimarsinan/mapping/mappers/conv2d_mapper.py:126-160`). Cost at
  tier-1/2 scale: two-scale is free; θ̂ is one calibration forward per
  projection commit (the WQ ladder has two projection rungs and one commit).
- **Mode scope.** The law is derived for first-crossing (cascaded) semantics;
  ttfsq/analytic cells neither need nor are harmed by two-scale (value-domain
  rounding error only shrinks when the weight grid refines). LIF's WQ
  craters are training-recoverable in-campaign (positive WQ endpoint deltas in
  `tier0_group_diagnoses.json`) and are out of scope here.
- **What this memo does NOT fix.** The mmixcore casc cells collapse in the FT
  hop ladder BEFORE WQ (t01_12/t0_16 rows) — that is the premature-firing /
  scalar-θ territory of `casc_first_crossing_transformation.md` and
  `mixer_column_scale_pathology.md`. casc deepcnn passes WQ consistently with
  this analysis (healthy grids + boundary-dominated segments).
- **Draw variance.** The exact campaign crater (0.9808→0.6689) is not
  bit-reproducible locally because run seeds are never applied
  (`numerical_boundary_consistency.md`); validation here is (i) exact-recovery
  and control experiments on two real artifacts spanning both campaign S
  values, (ii) a controlled substrate where the starved-grid crater is
  reproduced and repaired at will, (iii) the mechanism's fingerprints in the
  campaign logs (T-ordering of crater depth; ttfsq-vs-casc asymmetry at equal
  projection; destructive endpoint from a cratered entry).
- **θ̂ feasibility boundary is fundamental:** no threshold repairs an
  annihilated ramp. If the shared-grid contract is immutable on some backend,
  θ̂ plus the (grid-legal) DFQ-on-grid bias nudge is the remaining post-hoc
  space; the honest fix is the grid.
