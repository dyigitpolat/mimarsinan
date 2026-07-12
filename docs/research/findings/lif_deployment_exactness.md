# LIF deployment exactness: the commutation theorem, its violated preconditions, and the analytical correction series

**Question.** Deployed LIF (rate-coded, cycle-based) reads below its trained NF
across the campaign — lif mixer t0_01 S=4 crater to 0.8336 in-stage (final
0.9078), t01_01 S=8 −1.91 pp, t01_02 S=16 −0.68 pp; the prior frozen-NF healing
curve ran 0.32@T4 → 0.82@T8 → 0.90@T16. This memo derives, from the exact
deployed kernels, the arithmetic identity the LIF chain implements, proves the
commutation condition under which (quantize to S levels → spike) ≡ (spike →
count), enumerates every site where the implementation or a config choice can
break it (file:line), derives the complete correction series — of which the
shipped `lif_half_step_bias` is exactly one term — and prototypes each
correction through the repo kernels. Everything is arithmetic-consistency or
calibration-statistics; no training lever appears anywhere.

**Headline results (prototype on the repo kernels, MNIST chains, seeded, zero
training):**

- The deployed LIF chain is *already* a near-exact composed integer staircase:
  per-hop CONDITIONAL timing violations measure 10⁻⁴–10⁻³ rate units (the
  deployed count equals the staircase of its actual integer charge to that
  tolerance at every hop), and end-to-end the deployed arm bit-matches the
  floor twin where the twin collapses (0.1010 = 0.1010 at S=4) and sits
  0–2.8 pp under the nearest twin elsewhere — a gap the exact corrections
  below close. The LIF crater is therefore NOT a timing mystery; it
  decomposes into four named, separately-corrected terms.
- **Floor-kernel compounding is the catastrophic term** and the half-step bias
  is its exact fix: 9-hop norm-free chain at S=4 reads 0.1010 without it
  (bit-matching the analytical floor staircase, 0.1010) and 0.9180 with it
  (nearest staircase 0.9460). The measured floor curve 0.10@4 / 0.24@8 /
  0.83@16 / 0.97@32 reproduces the healing-curve shape.
- **The missing exact term is the readout**: logits are spike counts on the
  {0..S}·θ/S grid. The residual-charge identity `Q_T = θ·c_T + m_T` makes a
  membrane-augmented output decode EXACT (recovers the un-quantized,
  sign-carrying pre-activation): +2.6 pp at chain9 S=4 (0.9180 → 0.9440),
  +2.3 pp at mlp3 S=4 (0.9515 → 0.9745, ABOVE the staircase reference 0.9605).
  With it, deployed LIF sits at-or-above its commutation target in every cell.
- **Back-loading is real but small and separable**: the emission-cap deficit
  grows along depth (≤0.6 % rate/hop at S=4) and is killed value-losslessly by
  per-hop re-timing (boundary re-encode): +1.9 pp at chain9 S=4. Mixer-class
  vehicles already get this for free at their ComputeOp boundaries.
- **The casc-style threshold guard is REFUTED for multi-spike LIF** (−13.6 pp
  at chain9 S=4): the subtractive reset self-corrects transient overshoot
  (Theorem 3), so the guard's deficit cost has nothing first-order to buy back.
  Also refuted by measurement: V0=θ/2 initial-charge placement of the
  half-step, per-channel encode phase stagger, and stochastic (Bernoulli)
  encoding (−19 pp vs its uniform twin: the √S-noise law vs Uniform's deterministic 1/S law).
- **Statistical laws measured**: floor drift per hop −1/(2S)·(live mass);
  nearest-kernel residual noise obeys an exact 1/S collapse (hop-8 MAD × S =
  0.135–0.167 across S=4..32) with T-independent per-hop growth ratios; the
  dead-zone bias E[Q(x)−x] < 0 accumulates along depth and is absorbable by a
  per-channel affine fold (full affine helps up to +1.6 pp; bias-only folds can
  HURT −4.2 pp — mean-matching without gain is the wrong estimator at coarse
  grids).
- **Two any-vehicle exactness hazards found in the executor semantics**:
  (i) unequal-depth fan-in inside one neural segment (residual joins, or
  packing that fuses unequal-depth softcores) both DROPS the shallow branch's
  early spikes and RE-READS its stale final spike — measured 0.10 mean rate
  error on 30 % of join neurons at S=4; exact fix = depth-balancing relay
  cores (zero new mechanisms); (ii) under the strict `"<"` comparator an
  exact-θ charge never fires, so a unit-weight identity relay is DEAD — a
  boundary-convention trap for any integer-lattice backend.

Prototype scripts and raw outputs archived at
`docs/research/findings/lif_deployment_exactness_artifacts/`
(`lif_lab.py` = twin checks + main arms, `lif_lab2.py` = comparator/reset/WQ/
phase/V0, `lif_lab3.py` = drift law + Novena repair, `lif_lab4.py` = re-encode
+ stochastic; `results*.json`; working copies in scratch
`/tmp/claude-1005/-home-yigit-repos-research-stuff/11224c9e-f926-4cb5-a527-2d0211f4bd25/scratchpad/L1/`). All hops run THROUGH the repo kernels
(`lif_core_contribute_and_fire`, `lif_fire_and_reset`, `to_uniform_spikes`);
the closed-form count law is asserted bit-equal to the kernels before use
(float64 = `COMPUTE_DTYPE`; seed 0; 2000 test / 4000 calib samples; RTX PRO
6000, repo env, 2026-07-10).

---

## 1. Ground-truth deployed semantics (code-anchored)

Wire domain. `PerceptronTransformer` defines the effective (wire) parameters
the mapper emits: `W~ = per_input_scales · W / theta_out`
(`transformations/perceptron/perceptron_transformer.py:106-113`) and
`b~ = b / theta_out` (`perceptron_transformer.py:115-125`), with
`per_input_scales = theta_in` stamped by scale propagation
(`mapping/mappers/scale_propagation.py:76-102`). IR cores carry
`threshold = 1.0` (`mapping/ir_mapping_class_emit.py:97`); weight quantization
rescales to integers with `threshold = scale` (`mapping/export/chip_quantize.py:36-62`),
which cancels — the wire threshold is 1.

Executor (`models/spiking/hybrid/lif_step.py`, torch twin of the deployed
backends; cross-sim parity machinery pins HCM↔nevresim↔SANA-FE):

- **Encode** (segment entry): value → rate `r = clamp(v/θ, 0, 1)`
  (`models/spiking/hybrid/rate_forward.py:82`), then Uniform spikes
  `n = round(r·T)` placed by `to_uniform_spikes`
  (`chip_simulation/recording/spike_modes.py:32-41`); external backends use
  the same math (`chip_simulation/recording/_spike_encoding.py:11-24`).
  Exactly `n` spikes land in the window, and EVERY live channel pulses at
  cycle 0 (verified, `lif_lab.py` twin A1).
- **Cycle loop**: per active cycle, `memb += W~ @ s(t) + b~`; fire per
  comparator; subtractive reset (`Default`) or zero reset (`Novena`)
  (`models/spiking/lif_core_step.py:22-32`,
  `models/nn/lif_kernels.py:7,25-37`). The per-cycle `hw_bias` (equivalently
  the always-on bias axon firing every cycle,
  `models/spiking/hybrid/stage_io.py:146`) contributes `T·b~` per window.
  Comparator: `"<"` means fire iff `memb > θ` strictly; the tier-0 lif cells
  pin `"<"` (`test_configs/tier0/t0_01_lif_mmixcore_wq_s4.json:54`); the
  taxonomy default is `"<="` (`chip_simulation/spiking_semantics.py:132`).
- **Windows**: core latency = longest live path
  (`mapping/latency/chip.py:42-46`), invariant `consumer ≥ max(source)+1`
  (`chip.py:145-179`); a core steps and fills only inside `[L, L+T)`
  (`lif_step.py:117-129,131-159`). Fills read producer buffers written the
  PREVIOUS cycle, so for a gap-1 edge the consumer's T-cycle window covers the
  producer's T fire cycles exactly — no head or tail loss.
- **Decode**: per-window counts → `counts/T` (`spiking/segment_boundary.py:173-178`),
  per-producer (and per-channel, `ttfs_theta_cotrain`) theta divisors at host
  boundaries (`segment_boundary.py:51-88`,
  `mapping/support/activation_scales.py:66-88`); the network output is
  `rates × T` = raw spike counts (`rate_forward.py:199`). **Logits are
  integers in [0, T].**
- **Half-step**: `lif_half_step_bias` (default ON,
  `tuning/orchestration/conversion_policy.py:64`) folds `+θ/(2T)` into the
  effective bias before the WQ QAT
  (`mapping/support/bias_compensation.py:73-102`,
  `tuning/shift_calculation.py:1-7`,
  `pipelining/pipeline_steps/quantization/weight_quantization_step.py:89-93`)
  — i.e. `b~ += 1/(2T)` wire units, `+1/2` charge per window.

The NF training node (`models/nn/activations/lif.py:171-216`) runs the same
dynamics on `x/θ` with an IFNode at threshold 1 — the reference the pipeline
trains against; this memo's reference is the mode-independent one: the
composed staircase ANN below.

## 2. The commutation theorem

Fix a hop with wire weights `W~`, per-cycle bias `b~`, window `T`, θ=1,
subtractive reset. Input `i` delivers `n_i ∈ {0..T}` spikes at cycles
`A_i ⊆ {0..T−1}`. Define the charge process and terminal charge

```
Q_t = Σ_i w~_i · |A_i ∩ [0, t]|  +  (t+1)·b~ ,      Q ≡ Q_{T−1} = Σ_i w~_i n_i + T·b~ .
```

Define the comparator's staircase `F(x) = ⌊x⌋` for `"<="` and
`F(x) = ⌈x−1⌉` for `"<"` (number of thresholds passed).

**Theorem 0 (charge conservation).** At every cycle,
`Q_t = θ·c_t + m_t` exactly: the subtractive reset moves charge between the
count and the membrane and never destroys it. *(Induction on the update; this
identity is the basis of correction C2.)* Novena breaks it: each fire discards
`m − θ ≥ 0` (V7).

**Theorem 1 (fire rule).** The neuron fires at cycle `t` iff
`c_{t−1} < F(Q_t)`; hence `c_t = c_{t−1} + 1[c_{t−1} < F(Q_t)]` — a unit-speed
counter chasing the staircase of the running charge. *Proof:* fire ⟺
`m_t ▷ θ` ⟺ `Q_t − c_{t−1} ▷ 1` ⟺ `c_{t−1} < F(Q_t)` by the definition of F. ∎

**Theorem 2 (commutation).** If

1. **(window coverage)** every arrival lands inside the consumer's window —
   holds for every gap-1 edge by the latency invariant + previous-cycle
   buffer reads (§1); violated by gap>1 fan-in (V6);
2. **(no terminal overshoot)** `F(Q_t) ≤ F(Q)` for all `t`; and
3. **(chase completion)** any burst debt `F(Q_t) − c_t` drains before the
   window ends (sufficient: `F(Q_t)` rises ≤ 1 per cycle once above `c_t`),

then `c_T = clamp(F(Q), 0, T)` **exactly** — the deployed count is the
staircase ANN evaluated on the exact integer charge. Composing hops (intra-
segment counts pass as integers; boundary decode/re-encode is count-preserving
since `round((c/T)·T) = c`), the deployed network is the composed integer
staircase

```
c_out = clamp( F( Σ_i w~_i c_in,i + T·b~ + ½·[half-step] ), 0, T ),
```

with the half-step fold turning `F` into nearest rounding
(`F(x+½)` = round-half-up for `"<="`, round-half-down for `"<"`).
Violations are one-sided and bounded: overcount
`O ≤ F(max_t Q_t) − F(Q)` (rectified transient — c never decrements), deficit
`D` only from bursts landing too close to the window end to be emitted at
1/cycle.

**Theorem 3 (self-correcting reset; guard refutation).** Under subtractive
reset an overshoot-driven early fire debits the membrane by θ; later charge
must repay the debit before the count can grow again, so a transient
`Q_t > Q` inflates `c_T` only if the trough never recovers — `O` is
second-order. Any threshold-guard `θ_eff(t) = θ + h·(T−1−t)/T` (the casc
first-crossing lever, realized as `hw_bias += h/T`, `θ += h`) suppresses `O`
but delays every fire, converting the suppression into a first-order deficit
near the window end plus a `(1+h)` resolution factor. For the fire-once
cascade kernel the trade wins (casc memo Thm 2-4); for multi-spike LIF it has
nothing first-order to buy back. **Measured:** per-hop `O` ≤ 7·10⁻⁴ rate
units at baseline; hazard-calibrated guard (q=0.99, mean h 0.16–0.67) reads
0.7825 vs 0.9180 baseline at chain9 S=4 — refuted.

**Twin validation.** `lif_lab.py` asserts on the repo kernels: exact Uniform
spike counts and cycle-0 anchoring (A1); `c_T = clamp(F(dT),0,T)` bit-exact
for constant drives over both comparators and T ∈ {4,8,16,32} (A2); the
vectorized executor bit-equal to `lif_core_contribute_and_fire` loops,
membranes included (A3). All pass.

## 3. The violation ledger (every site that breaks commutation)

Vehicles: `mlp3` = 784-128-128-10 (float 0.9800), `chain9` =
784-128×8-10, ReLU, norm-free — the mixer-class chain shape (float 0.9770);
count-quantile θ (q=0.99, pipeline default,
`transformations/activation_scale_policy.py:11-33`); entry layer host-subsumed
(`encoding_layer_placement: subsume`). Per-hop numbers below are chain9.

| # | violation | site | S=4 magnitude | class |
|---|---|---|---|---|
| V1 | **floor kernel** (no half-step): terminal count = `F(Q)`, mean drift −1/(2S)/hop | fixed by `conversion_policy.py:64` default | acc 0.1010 (= analytic floor twin 0.1010); drift/hop −0.024..−0.10 | exact-fix shipped (B3) |
| V2 | **readout count quantization**: logits = counts ∈ {0..S}, ReLU-positive, ties argmax-first | `rate_forward.py:199`, `lif_step.py:199-217` | −2.6 pp (chain9), −2.3 pp (mlp3) vs membrane decode | exact fix missing (C2) |
| V3 | **back-loading deficit**: charge arriving late + 1 spike/cycle emission cap | executor physics (`lif_step.py:131-159`) | deficit 0.0003→0.006 rate/hop along depth; −1.9 pp vs re-timed | exact-fix by re-timing (C3) |
| V4 | **transient overcount** (rectified timing noise) | same | ≤7·10⁻⁴ rate/hop | negligible (Thm 3); guard refuted |
| V5 | **dead-zone / saturation bias**: nearest zeroes mass < θ/(2S); clamp at r=1 one-sided | encode `rate_forward.py:82` + kernel | residual signed drift −0.001→−0.044/hop with depth | statistical (C4) |
| V6 | **unequal-depth fan-in**: head-drop of early shallow-branch spikes AND stale-buffer re-read of its last spike | `lif_step.py:60-64,117-148` (buffers only overwritten while active; fills read them unconditionally); latency `chip.py:145-179`; `_align_shiftable_cores` (`chip.py:114-143`) only rescues input-only branches | join demo: mean rate error 0.102, 31.5 % neurons differ (T=4); stale-repeat dominates (join counts > balanced) | exact fix: relay balancing (C5) |
| V7 | **Novena zero-reset**: discards `m−θ` per fire (Theorem 0 broken) | `lif_kernels.py:26-27`; guarded by `firing_strategy.py:75-90` | −1.7 pp (S=4) to −10.4 pp (S=8, 0.8640 vs 0.9680) | expectation-repair only (C6) |
| V8 | **encoder noise family**: Bernoulli (Stochastic) gives per-hop CLT noise ∝ 1/√S vs Uniform's deterministic ∝ 1/S | `spike_modes.py:10-11` vs `:32-41` | stochastic 0.7430 vs uniform 0.9370 (S=4) | config invariant: keep Uniform |
| V9 | **strict-`"<"` tie loss**: exact-θ charge never fires (a unit-weight relay is dead; any backend that snaps θ onto the weight-integer lattice loses one level at exact grid points) | `lif_kernels.py:7`, t0_01 config | float weights: 0 measured (`<` ≡ `<=` in all cells); hazard is lattice-specific | boundary convention |

Cross-checks against the campaign: the floor curve 0.101/0.240/0.833/0.969
(S=4/8/16/32) reproduces the frozen-NF healing-curve shape (0.32/0.82/0.90);
the corrected (nearest) chain's crater −5.9 pp at S=4 → −0.9 pp at S=8 →
−0.15 pp at S=16 matches the campaign's 1/S crater scaling (−6.91/−1.91/−0.68
pp); the deployed arm equals its staircase twin to ≤0.5 pp everywhere —
consistent with t0_01's LIF entry 0.8336 sitting a few pp under the mixer's
one-shot nearest-S4 read 0.868 (mixer memo §2), the difference being V2+V3+V5.
The remaining gap to float on starved-θ vehicles is the shared scalar-θ grid
pathology — mixer memo territory (equalization, per-channel θ), NOT re-derived
here.

## 4. The correction series (complete, in existing parameter space)

The realizable per-neuron terms are: per-cycle bias (`hw_bias`), threshold
(per-core scalar; per-neuron via exact row scaling), consumer weight
columns/bias (fold currency, same family as `node_output_shifts` and the
negative-shift bakes, `bias_compensation.py:204-230`), window/latency
structure, and the readout decode. The complete series, each term with its
exactness class and measured value:

**C1 — half-step fold (shipped).** `b~ += θ/(2T)`: exact floor→nearest
identity on the terminal count (Theorem 2). Worth +81.7 pp at chain9 S=4.
Placement matters and the shipped per-cycle ramp is the RIGHT placement:
the same total charge as initial membrane `V0 = θ/2` front-loads the
transient and measures WORSE (0.9085 vs 0.9180 at S=4; 0.9645 vs 0.9680 at
S=8) — and V0 is not in the parameter space anyway (bias axons fire every
cycle). The residual C1 leaves: V2, V3, V5 below.

**C2 — membrane-augmented readout (exact, missing).** By Theorem 0,
`Q = θ·c_T + m_T` at window end; decoding output cores as
`logits = θ_out·(c_T + m_T/θ)/T − θ_out/(2T)` (half-step charge removed)
recovers the exact, unquantized, SIGN-CARRYING pre-activation — infinite
logit resolution and no ReLU-positivity at the readout, from state the
executor already holds.

| vehicle, S | counts readout (a1) | + membrane readout | staircase ref |
|---|---|---|---|
| mlp3 S=4 | 0.9515 | **0.9745** | 0.9605 |
| mlp3 S=8 | 0.9720 | 0.9760 | 0.9750 |
| chain9 S=4 | 0.9180 | **0.9440** | 0.9460 |
| chain9 S=8 | 0.9680 | 0.9745 | 0.9710 |
| chain9 S=16/32 | 0.9755 / 0.9755 | 0.9760 / 0.9760 | 0.9765 / 0.9760 |

With C2 the deployed network sits at-or-above its commutation target in every
cell — the LIF-specific residual is closed.

**C3 — per-hop re-timing (value-exact, structural).** Boundary decode +
Uniform re-encode is count-preserving (`round((c/T)·T) = c`), and resets
arrival timing, killing the back-loading deficit V3. Measured: chain9 S=4
0.9180 → 0.9370 (+1.9 pp), S=8 0.9680 → 0.9725; nil at S≥16 and on shallow
vehicles. Mixer-class vehicles already re-time at every ComputeOp boundary —
per-hop segmentation is only a (mapping-level) choice for deep single-segment
chains at S ≤ 8. Encode-phase dithering — the other conceivable re-timing —
is refuted (0.9130 vs 0.9180: coincidence structure is not the mechanism; the
executor's O/D are already tiny).

**C4 — per-channel affine fold (statistical, calibration-only).** The
dead-zone/saturation bias (V5) and Novena's discard (V7) shift and shrink
per-channel means; the generic absorber is the consumer-side affine
`r → a⊙r + c` folded exactly as `W~[:,j] *= a_j`, `b~ += W~ @ c` (host
per-class affine at the readout), fitted per channel by least squares on
calibration data, layer-sequential. Measured (on top of C1):

- full affine: chain9 S=4 0.9180 → 0.9340; +C2: 0.9475; elsewhere ±0.2 pp
  (already commutation-tight);
- **bias-only folds are the wrong estimator at coarse grids**: chain9 S=4
  0.9180 → 0.8760 (−4.2 pp) — mean-matching a 5-level variable shifts many
  staircase decisions coherently; always fit gain+bias;
- Novena repair: S=8 chain9 0.8640 → 0.9450 (+8.1 pp); S=4 it overfits the
  5-level grid (0.9010 → 0.8845) — gate on S ≥ 8. Keep `Default` reset where
  the choice exists.

**C5 — depth-balancing relays for joins (exact, any-vehicle).** For every
intra-segment edge with latency gap > 1, insert identity relay cores on the
shallow branch (or equivalently fix the executor's stale-buffer read — a src
change, out of scope here). Join demo (gated executor semantics, repo
kernel): rate error 0.102/0.045/0.022 at T=4/8/16 on ~30 % of join neurons,
restored exactly by relays. Caveat V9: a relay's identity weight must exceed
θ under `"<"` (1+ε, or a θ−ε threshold; exact-1.0 is silent).

**C6 — variance-optimal gain (statistical, bounded value).** The per-hop
nearest noise is fresh, symmetric, ±θ/(2S); the LS gain of C4 is the greedy
per-hop MSE minimizer, and no per-layer gain can beat the 1/S law below
(deterministic, not CLT). Where the mixer memo's starved-θ regime applies,
scale surgery (equalization / per-channel θ) is the effective "gain"
correction — shared pathology, not LIF-specific.

**Refuted terms** (measured, so the series is provably complete in this
space): threshold-guard ramp (Theorem 3; 0.7825 at chain9 S=4), V0 half-step
placement, encode phase stagger, stochastic encoding, bias-only folds.

## 5. Statistical laws

**Floor drift (why the healing curve looks the way it does).** Per hop,
`E[F(Sz)/S − z] = −1/(2S)·P(z live)` (dead channels cannot drift). Measured
drift/hop at S=4: −0.024..−0.103 (bound −0.125); the drift compounds through
the ReLU chain into the 0.10@S4 collapse. The half-step cancels this term
identically — the corrected chain's residual drift is 10–30× smaller.

**Nearest residual (dead-zone) bias.** With C1, the remaining signed error is
`E[N(Sx)/S − x] = −∫₀^{Δ/2} x dP(x) + (tail terms)`, Δ = θ/S — strictly
negative for ReLU-heavy-tailed channels (the sub-half-step mass the mixer
memo measures at 51–80 % on starved hops) and compounding along depth:
measured −0.0010 (hop 1) → −0.0437 (hop 8) at S=4, −0.0006 → −0.0049 at S=32.
This is the closed form behind C4's bias component (fold `−W~·μ` into the
consumer, μ from calibration stats) — but deploy it only inside the full
affine (see C4).

**Noise compounding law (the 1/S law).** Each hop injects fresh quantization
noise uniform on ±θ/(2S) (σ² = θ²/(12S²)) plus timing noise (O+D); the
deployed-vs-twin per-hop MAD grows with T-INDEPENDENT ratios
(1.19/1.31/1.06/1.67/0.99/1.27/1.37 along chain9 — structure, not time,
determines amplification) and collapses exactly as 1/S:

| S | hop-8 MAD (rate) | MAD × S |
|---|---|---|
| 4 | 0.0416 | 0.167 |
| 8 | 0.0208 | 0.166 |
| 16 | 0.0094 | 0.150 |
| 32 | 0.0042 | 0.135 |

End-to-end distortion ≈ (θ/2√3·S)·Σ_k γ^{L−k} with measured γ ≈ 1.2–1.4/hop;
accuracy heals when this falls below the class margin — the analytical form of
the healing curve and of the campaign's 1/S crater scaling. Per-hop SNR under
Uniform encoding is deterministic ∝ S (NOT the Bernoulli √S law — measured:
stochastic encode loses 19 pp at S=4, 0.7430 vs 0.9370); nothing in the correction space can
beat 1/S per hop except more levels per channel — the shared starved-θ lever.

**WQ orthogonality.** 5-bit symmetric weight quantization on the wire moves
every cell by ≤ 0.6 pp (chain9 S=4: 0.9235 vs 0.9180) and `"<"` ≡ `"<="`
holds in all measured cells (float thresholds keep ties measure-zero; V9 is a
lattice hazard, not a current defect).

## 6. Prototype summary (chain9, the mixer-shaped vehicle; float 0.9770)

| S | floor twin | nearest twin | a0 floor | a1 +half-step | a3 +affine | a4 guard | a1+C2 memb | a3+C2 | C3 re-time | C3+C2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.1010 | 0.9460 | 0.1010 | 0.9180 | 0.9340 | 0.7825 | 0.9440 | **0.9475** | 0.9370 | 0.9440 |
| 8 | 0.2645 | 0.9710 | 0.2395 | 0.9680 | 0.9720 | 0.9685 | 0.9745 | 0.9745 | 0.9725 | 0.9720 |
| 16 | 0.8285 | 0.9765 | 0.8325 | 0.9755 | 0.9765 | 0.9735 | 0.9760 | 0.9760 | 0.9750 | 0.9760 |
| 32 | 0.9685 | 0.9760 | 0.9685 | 0.9755 | 0.9755 | 0.9735 | 0.9760 | 0.9755 | 0.9765 | 0.9765 |

mlp3 mirrors it shallowly (S=4: a1 0.9515 → a1+C2 0.9745 vs twin 0.9605;
S≥8 within noise of float 0.9800). Every arm is a single seeded run driven
through the repo kernels; the twins are closed-form and kernel-verified.

## 7. Integration points (nothing implemented in src/)

1. **C2 membrane readout**: the torch executor already holds the final
   membranes — extend the output-span accumulation
   (`models/spiking/hybrid/lif_step.py:199-217`) to optionally return
   `counts + memb/θ` for output-source cores, decoded at
   `rate_forward.py:113,199` (the per-channel theta divisor plumbing exists,
   `segment_boundary.py:51-88`). Backend parity: nevresim/SANA-FE need a
   final-potential read of output neurons (simulator-side state, chip-
   plausible readout); gate `is_lif` only, output cores only, and keep the
   count decode for parity gates until cross-sim carries the same read.
   *R8 adjudication (2026-07-13, measured): nevresim exports counts only
   (no membrane read port), so the shipped decode is a torch-side DIAGNOSTIC
   excluded from every deployed-read metric — deployed claims may not include
   C2 until a backend exports final membrane; see
   `lossless_refinement_ledger.md` §2F.1.*
2. **C4 affine folds**: a calibration pass in `tuning/`, same fold family as
   the negative-shift bake (`mapping/support/bias_compensation.py:204-230`)
   and the mixer memo's consumer-column scaling; must run BEFORE weight
   quantization; one recipe knob in
   `tuning/orchestration/conversion_policy.py` (`_LIF_RECIPE_KNOBS`), full
   affine only, Novena arm gated S ≥ 8.
3. **C3 re-timing**: a mapping-level option to split deep single-segment
   chains into per-hop (or per-k-hop) neural segments at S ≤ 8 — the hybrid
   stage machinery already decodes/re-encodes at every boundary
   (`rate_forward.py:110-113`, `segment_boundary.py:181+`) and the transcode
   is count-exact.
4. **C5 join safety**: a post-`ChipLatency` assertion that every live
   intra-segment edge has gap ≤ 1, with relay insertion (weights 1+ε under
   `"<"`) as the exact remedy; alternatively (src change, flagged only) zero
   producer buffers after window end in `lif_step.py` and un-gate the fills.
5. **Verification locks**: (i) Theorem-2 lock — a fixed random config where
   the executor count must equal `clamp(F(Q),0,T)`; (ii) Theorem-0 lock —
   `Q = θc + m` on the executor state; (iii) a gap-1 invariant test on every
   tier-0 mapping; (iv) the A1 encode locks (exact count, cycle-0 anchor).

## 8. Genericity and honest limits

**Generic by construction.** Every correction is either an arithmetic
identity of the deployed kernel (C1, C2, C3, C5 — no workload constants, no
training) or a per-channel calibration statistic (C4, C6 — same statistics
the pipeline already collects for gauges/negative shifts). The laws (§5) are
distribution-level, measured to collapse across S and depth; nothing is tuned
to MNIST beyond the models themselves.

**Limits.**

- The shared scalar-θ grid starvation (mixer memo) is untouched by
  construction: with C1+C2+C4 the deployed chain EQUALS its integer staircase
  twin, and the twin's distance to float is the starved-grid distance. The
  mixer's S=4 cells need the equalization / per-channel-θ levers first.
- C2 changes what the readout reports (membrane + counts); any
  analytic-vs-deployed comparison and the NF↔SCM parity gates must adopt the
  same decode on both sides (the NF node's mean-rate output corresponds to
  counts; its pre-threshold membrane is available in the cycle-accurate
  forward).
- The prototype vehicles are FC chains; convs/pooling ride the same kernel
  and boundary code paths but were not separately measured. The mixer's
  permute boundaries were argued (count-exact transcode) and indirectly
  measured via C3, not end-to-end.
- The Novena affine repair is expectation-level (Theorem 0 is genuinely
  broken); at S=4 it overfits. Default reset is the semantically clean mode.
- V9 (strict-comparator lattice ties) is a hazard statement, not a measured
  current defect; it binds only if a backend snaps thresholds onto the
  weight-integer lattice.

## 9. Relation to prior artifacts

- `mixer_column_scale_pathology.md`: owns the shared per-hop starved-θ
  component and its exact scale-migration fix; this memo owns the residual —
  with C1+C2+C4 deployed-LIF ≥ nearest-staircase in every measured cell, so
  the two memos together decompose the whole LIF crater.
- `casc_first_crossing_transformation.md`: the guard family transfers in
  parameter space but NOT in mechanism — Theorem 3 + measurement close that
  branch for multi-spike LIF (the reset physics differ: fire-once has no
  self-correction; subtractive LIF does).
- `research_artifacts_for_cascaded_ttfs_tuning/10_decode_model_and_law.md`:
  the ρ(S)=1−c/S retention and √S depth-budget laws describe the OLD latched
  TTFS semantics; the LIF analogues derived here are the −1/(2S) floor drift
  (cancelled by C1) and the 1/S noise collapse with structure-determined
  depth gain.
- `[5v B3]` (`conversion_policy.py:58-64`): the half-step fold's design —
  trainable entry fold before the WQ QAT — is validated by the V0-placement
  refutation; this memo adds the terms B3 does not cover (C2–C6).
