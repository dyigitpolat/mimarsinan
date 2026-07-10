# Synchronized-TTFS deployment exactness: the composition theorem, its violated preconditions, and the analytic corrections

**Date:** 2026-07-10.
**Question.** The seeded (2026-07-10) tier-0/0.1 synchronized mixer cells sit
in a pinned band below the acceptance bar: **every S=8 pruned10 lever variant
reads exactly 0.9527** with byte-identical stage trajectories (t0_21,
t01_09_e4, t01_24_floor — pre 0.9614 → prune 0.9509 → AQ 0.9472 → WQ 0.9527 →
every deployment step +0.0000;
`~/.slurmech/workspaces/mimarsinan-xlog1/runs/20260710-025531-9267ffde` etc.),
0.9525 at S=4 (t01_05, AQ 0.9317), 0.9641 at S=16 (t01_04) — while every
non-mixer sync cell passes easily (0.977–0.9942). This memo derives the exact
end-to-end composition the deployed sync forward computes, proves it equals
the `sync_exact_qat` train-forward and names the preconditions that carry the
equality, derives the statistical law of the composed staircase error, and
measures a ladder of ANALYTIC corrections — arithmetic-consistency and
calibration-statistics fixes only, zero training anywhere — on the real
kernels and the real tier-0 vehicles.

**Headline results.**

1. **Deployment is exact; the drop is the composition itself.** Under
   preconditions P1–P6 (§2) the deployed SCM forward equals the exact-QAT NF
   forward neuron-by-neuron (Theorem 1); the seeded campaign confirms it at
   machine precision (torch↔deployed-sim parity 1.0000/256, all post-WQ
   deltas +0.0000). Sync robustness is therefore a property of the quantized
   composition's arithmetic and statistics, not of any simulator seam — and
   the sound levers are exactly (a) the remaining in-composition arithmetic
   inconsistencies and (b) statistically optimal (θ, offset) placement at
   fixed S.
2. **Found: the subsumed encoder is a floor-staircase hop with NO half-step**
   — `apply_half_step_entry_fold` skips `is_encoding_layer`
   (`src/mimarsinan/mapping/support/bias_compensation.py:80-83`), but under
   `encoding_layer_placement: subsume` the encoder ComputeOp runs the
   perceptron module itself, ceil staircase included
   (`src/mimarsinan/mapping/mappers/perceptron_mapper.py:29-32,56-90`). The
   entry of every network carries a systematic −θ_enc/(2S) floor bias, in
   train and deploy alike (parity-invisible, statistically costly). Extending
   the fold to subsumed encoders — one predicate — is worth **+15.9/+29.8/
   +4.8 pp** on the mixer at S=4/8/16 over the shipped half-step state, and
   +30.5 pp on top of the deflated S=4 state (§6, A1e/A2e rows).
3. **Found: the readout is float only by mapping accident.** A `Linear` with
   no following activation converts to a host ComputeOp
   (`src/mimarsinan/torch_mapping/converter_handlers/linear_mixin.py:48-59`),
   so both tier-0 vehicles' classifiers bypass the staircase. Staircasing the
   readout would cost −10.1/−2.0/−0.5 pp (mixer, S=4/8/16). This must become
   an explicit contract: any vehicle whose last layer carries an activation
   silently loses its logit resolution to an S-level grid (P6).
4. **The composed-error law is drift-then-saturate, not √L·(1/S).** The ceil
   kernel is a floor quantizer in value space (Identity 1): without the
   half-step every hop's error mean is strictly negative and same-sign
   (measured −0.26…−0.44 grid steps, neg-fraction up to 0.95), compounding
   through positive-gain ReLU paths into whole-population flooring (shipped
   pre-B1 state A0: mixer at CHANCE for every S ≤ 16; lenet at chance at
   S=4). The half-step (mid-tread conversion, Identity 2) cancels the
   uniform-density part; measured residual means drop to −0.005…−0.067 steps.
   Centered errors are weakly correlated across hops (0.03–0.13) and the
   composed logit RMSE SATURATES (mixer 0.61→1.35 by hop ~5, flat after):
   the first hops carry almost all the damage — which is why the two entry
   fixes (encoder fold, θ loading) dominate everything else.
5. **At fixed S the free statistical knobs are (θ, offset); offsets are
   already solved, θ is not.** Per-channel MSE-optimal offsets equal the
   uniform half-step to 4 decimals on both vehicles (B3 ≡ B1: the +θ/(2S)
   fold is already per-channel-MSE-optimal — a genuinely useful negative
   result). The win is θ loading: the shipped full-quantile θ (q=1.0) is the
   worst plausible choice at low S; the measured optimal quantile FALLS with
   S and with depth (mixer S=4: 0.995→0.95 along depth; lenet S=4 fc hops:
   0.90). Per-hop accuracy-guided quantile selection (B1) and per-channel θ
   (A5/B4, exactly realizable on perceptron→perceptron edges today, §4) plus
   the sequential first-moment fold (B2) give, with zero training:
   **mixer 0.0974 → 0.8674 (S=4), 0.0974 → 0.9401 (S=8), 0.1047 → 0.9502
   (S=16)** against float 0.9602; lenet ≥ 0.9879 everywhere (float 0.9916).
   The stack survives 5-bit two-scale WQ within −0.9 pp.
6. **First-moment folding has a sign trap worth recording:** matching
   deployed-vs-float PRE-activation means naively cancels the half-step it
   rides on (the measured mean gap ≈ +0.5 grid step IS the intentional
   compensation) and collapses accuracy (0.93→0.59 at S=8). The correct
   closed form excludes each hop's own offset and folds only the propagated
   upstream error — then it is worth +0.1…+0.9 pp (§3.2).
7. **The two-scale WQ bias lattice carries the folded half-step only
   coarsely** — at 5 bits the bias grid step is 0.08–1.41× the half-step on
   these vehicles (§5) — but measured end-to-end damage at 5 bits is ≤0.5 pp
   and re-folding the exact half-step after projection is NOT uniformly
   better (−1.6 pp at S=16 on one draw). Verdict: a watch-item with a ready
   exact repair (comparator-side half-step), not a current binder.

Prototype (nothing under `src/` modified):
`/tmp/claude-1005/-home-yigit-repos-research-stuff/11224c9e-f926-4cb5-a527-2d0211f4bd25/scratchpad/S1/`
(`proto_sync.py`, `results.json`; RTX PRO 6000, repo env, seed 0). All
quantization arithmetic goes through the repo wire kernels
(`ttfs_quantized_staircase`, `ttfs_grid_quantize`; torch/numpy twins asserted
bit-equal before use), vehicles are the real classes (`TorchMLPMixerCore` in
the exact t0_21 shape, `LeNet5`), trained with the tier-0 recipe (AdamW 3e-3,
cosine+warmup, 4 epochs) and hop-graph-asserted against the float64 model
forward to max|Δlogit| ≤ 1.2e-14. Envelopes reproduce the campaign band
(mixer 0.9602 vs campaign 0.9614; lenet 0.9916).

Scope notes (mechanisms owned elsewhere, not re-derived): first-crossing
premature firing is cascaded-only — the synchronized schedule grid-snaps
whole windows and has no crossing-time hazard
(`casc_first_crossing_transformation.md`); the WQ shared-grid bias crater is
repaired by the committed two-scale projection, which the sync recipe carries
(`wq_cascade_crater_repair.md`;
`src/mimarsinan/tuning/orchestration/conversion_policy.py:93-96`); the
inter-channel scale spread is repaired by the M4 cross-layer migration
(`mixer_column_scale_pathology.md`) — this memo's target is the residual
AFTER those: the within-channel, within-composition distortion.

---

## 1. The exact deployed composition (code-anchored)

### 1.1 The two kernels and their value-space identities

All sync quantization is built from two wire kernels
(`src/mimarsinan/models/spiking/wire_semantics.py:11-37,68-95`):

```
Q_S(r) = (S − clamp(⌈S(1−r)⌉, 0, S−1))/S   if ⌈S(1−r)⌉ < S, else 0   (ceil kernel)
N_S(x) = (S − round(S(1−clamp(x,0,1))))/S  if round(·) < S, else 0   (grid snap)
```

**Identity 1 (the ceil kernel is a floor quantizer).** With ⌈a⌉ = −⌊−a⌋:
`S − ⌈S − Sr⌉ = ⌊Sr⌋` for `Sr ∉ ℤ` and `= Sr` on grid points; the fire
condition `⌈S(1−r)⌉ < S ⇔ r ≥ 1/S`. Hence

```
Q_S(r) = 0                 r < 1/S            (dead zone: one FULL grid step)
       = min(⌊Sr⌋, S)/S    r ≥ 1/S, Sr ∉ ℤ    (floor: −1/(2S) mean bias under
       = min(r, 1)         Sr ∈ ℤ              locally-uniform mass)
```

Grid points are fixed points (ties-to-self) — this is what lets on-grid wire
values pass every boundary snap exactly (§1.2).

**Identity 2 (the half-step turns floor into mid-tread round).** The
`sync_entry_half_step` fold adds +θ/(2S) to the effective pre-activation
(`bias_compensation.py:73-110`; magnitude = `calculate_activation_shift` =
(θ/2)/S, `src/mimarsinan/tuning/shift_calculation.py`), so the hop computes
`Q_S(r + 1/(2S))` — round-half-up: unbiased under locally-uniform within-cell
mass, dead zone halved to [0, 1/(2S)), max error halved. `N_S` differs from
it only at exact half-ties (round-half-even).

### 1.2 The deployed forward, stage by stage

The deployed sync read is the SCM float64 numpy path
(`src/mimarsinan/models/spiking/hybrid/ttfs_step.py:59-139`; sync disables
nevresim — `conversion_policy.py:248-251` — so this IS the deployment):

- **Neural stage.** Assembled inputs (+ node output shifts) are grid-snapped
  as a whole — synchronized only: `ttfs_input_grid_quantize(seg_in, S)` = N_S
  (`src/mimarsinan/chip_simulation/ttfs/ttfs_executor.py:187-194`,
  `ttfs_encoding.py:18-24`). Cores execute in latency order:
  `v = a_in @ W_core.T + hw_bias`, then `Q_S(v/θ_core)`
  (`src/mimarsinan/chip_simulation/ttfs/ttfs_segment.py:155-173`) with one
  scalar threshold per hard core (`segment_arrays.py:117`) — 1.0 in effective
  coordinates (`src/mimarsinan/mapping/ir_mapping_class_emit.py:97,171`);
  the θ-division lives in the effective weights
  `per_input_scales · W · u / θ_out`
  (`src/mimarsinan/transformations/perceptron/perceptron_transformer.py:106-124`).
  Core-to-core wires inside a stage carry RAW grid values
  (`ttfs_segment.py:143-160`): **an S-level output feeds the next hop's
  S-level input exactly — no rescaling, no re-rounding**; the producer's
  decode θ is folded into consumer weights, never applied on the wire.
- **Host stage (ComputeOp).** Decode by a scalar `in_scale`, compute in
  absolute units, re-normalize by a scalar `out_scale`
  (`src/mimarsinan/chip_simulation/hybrid_run/hybrid_execution.py:253-272`;
  scales are mean-collapsed per node,
  `src/mimarsinan/mapping/support/activation_scales.py:49-63`).
- **Structural ops are free.** Permutes/reshapes rearrange IR sources — axon
  wiring, no host stage, no snap
  (`src/mimarsinan/mapping/mappers/structural.py:64-75`). The mixer's
  token/channel permutes keep its 8 fc hops inside one neural segment.
- **The subsumed encoder** runs the perceptron MODULE on host — including its
  installed ceil-staircase decorator — normalized by θ_enc
  (`perceptron_mapper.py:29-32,56-90`; wrapped-scale resolution
  `activation_scales.py:33-46`). The consumer stage's N_S snap is then an
  identity on the already-on-grid output.
- **The readout.** A `Linear` with no absorbed activation converts to a host
  ComputeOp (`linear_mixin.py:48-59`); `mean(dim=1)` likewise
  (`src/mimarsinan/torch_mapping/mapper_graph_converter.py:219-220`). Both
  run float; the final output is scaled by T (`ttfs_step.py:139`),
  argmax-invariant.

The full deployed composition of the tier-0 mixer is therefore:

```
x → [host] patch_embed+BN+ReLU → Q_S(·/θ_enc)·θ_enc      (encoder: NO half-step)
  → 8 × [chip] θ_h·Q_S((W̃_h a + b̃_h)/θ_h + 1/(2S))      (one segment, exact wires)
  → [host] mean over patches → classifier → logits·T      (float, never staircased)
```

and LeNet5's: conv1 encoder hop; {conv2, fc120, fc84} staircase hops
(max-pool preserves the grid — a max of grid values is a grid value); fc10
float. The synchronized-specific N_S boundary snap is **inert on both
vehicles** — every value reaching it is a grid fixed point; it becomes live
only when a host op emits off-grid values into a neural stage (avg-pool
between on-chip layers, residual joins of different-θ branches, per-channel
normalizers — see P1/P2).

### 1.3 Theorem 1 (train/deploy composition equality)

The `sync_exact_qat` NF trains, per perceptron, `θ·Q_S(x/θ)` under STE
(`src/mimarsinan/models/nn/decorators/clamp_quantize.py:30-50` →
`TTFSStaircaseFunction`, `src/mimarsinan/models/nn/activations/autograd.py:57-70`
— the forward IS the wire kernel), plus N_S entry quantizers on exactly the
`segment_entry_perceptrons` set (`autograd.py:111-131`; install
`src/mimarsinan/tuning/orchestration/adaptation_manager.py:81-114`; set
definition `src/mimarsinan/torch_mapping/encoding_layers.py:81-107`).

**Theorem 1.** Under P1–P6 below, the deployed SCM forward equals the NF QAT
forward for every input, neuron by neuron, in exact arithmetic.
*Proof sketch:* induct over the hybrid stage schedule. Host stages run the
same modules on both sides (the subsumed encoder literally IS the NF
perceptron). Neural stages: the effective-coordinate fold makes the deployed
`v/θ_core` equal the NF's `x/θ` (both sides read
`perceptron_transformer.py:106-124`); `Q_S` is the same function; intra-stage
wires are raw grid values on both sides; entry snaps apply the same N_S at
the same seams; grid fixed points make snaps of neural-to-neural values
identities. ∎

The seeded campaign realizes the conclusion at machine precision (parity
1.0000/256; +0.0000 through SCM/HCM/verification). The theorem's real content
is its preconditions — each a live seam that can silently break equality in
another regime.

## 2. The preconditions (P1–P6), and where each is enforced or violated

- **P1 — scalar θ per hop == the wire/decode normalizer.** Holds today
  (θ is a per-perceptron scalar). Per-channel θ breaks it at the three
  scalar-collapse seams of §4.3 while the perceptron→perceptron path already
  supports it exactly.
- **P2 — QAT entry-quantizer set == deployed snap seams.**
  `segment_entry_perceptrons` (first on-chip core after Input/ComputeOp) vs
  the executor's every-neural-stage snap (`ttfs_executor.py:193-194`). They
  agree because snaps are identities on on-grid values; any change that makes
  a boundary off-grid on one side only (P1 violations, new host ops) breaks
  the pairing.
- **P3 — the half-step is folded exactly once, and the mapping-time +0.5/Tq
  compensation is skipped for exact-QAT models.** Enforced: all-or-none
  marker (`adaptation_manager.py:58-78`) + the skip with assertion at
  `src/mimarsinan/pipelining/pipeline_steps/mapping/soft_core_mapping_step.py:428-446`.
  The RESIDUAL inconsistency is coverage: the fold skips `is_encoding_layer`
  (`bias_compensation.py:80-83`) — correct for `offload` placement (the chip
  input encode is the round `ttfs_spike_time`, already mid-tread) but WRONG
  for `subsume`, where the encoder is itself a ceil-staircase hop. Both
  sides agree (parity-invisible); the composition eats a floor-biased entry.
  Measured cost: §6, A1→A1e.
- **P4 — the WQ bias lattice carries the folded half-step.** After two-scale
  projection the effective bias lives on `g_b = ratio/scale_w`
  (`src/mimarsinan/transformations/normalization_aware_perceptron_quantization.py:67-83`);
  the folded 1/(2S) rounds by up to g_b/2. Measured `g_b/(1/(2S))`: mixer
  0.21–1.41 (S=8), 0.08–0.70 (S=4); lenet 0.22–0.39 — the lattice represents
  the half-step only to ±(10–70)% on the worst hops. Train/deploy stay
  bit-consistent (the QAT owns the rounded value): a statistical wound, not
  a parity wound. §5 quantifies; it does not currently bind.
- **P5 — dtype tie stability.** QAT computes f32/GPU, deployment f64/numpy;
  `⌈·⌉` flips a full level on a 1-ulp disagreement at a grid tie, and ties
  are STRUCTURALLY common (wires are exact grid values). Measured: 0 argmax
  flips from f32-cast parameters on both vehicles at S=8. Non-binding on
  these draws; the only precondition with no structural guarantee — keep it
  measured (the existing parity gate does exactly this), never assumed.
- **P6 — the readout stays float.** Not a contract anywhere — it falls out of
  `linear_mixin.py:48-59` ("no absorbed activation ⇒ host ComputeOp"). An
  on-chip staircased readout has S+1 distinguishable logit levels per class
  and index-order argmax ties; measured cost −10.1/−2.0/−0.5 pp (mixer),
  −1.5/−0.5/−0.1 pp (lenet) at S=4/8/16 (§6, NEG row). A vehicle whose final
  layer HAS an activation gets this silently today.

## 3. The statistical law of the composed staircase

### 3.1 Per-hop error decomposition and the first moment

For hop h, channel c, normalized pre-activation `r = z/θ` with calibration
marginal F, the deployed activation (half-step folded) has LOCAL error

```
ε(r) = −r                  r ∈ [0, 1/(2S))          dead zone
     = round(Sr)/S − r     r ∈ [1/(2S), 1+1/(2S))   granular (±1/(2S))
     = 1 − r               r ≥ 1 + 1/(2S)           overload (clip)        (3)

μ_c = E[ε]  — one calibration pass, per channel.                           (4)
```

Without the half-step, granular becomes floor (−1/(2S) mean under uniform
mass) plus a FULL-width dead zone: strictly negative, same-sign at every hop.
Measured local means per hop (grid-step units, S=8, deployed prefix):

| state | mixer hops (enc → deep) | lenet hops |
|---|---|---|
| no half-step (A1-state ref) | −0.26 … −0.44, neg-frac 0.56–0.95 | −0.05 … −0.27 |
| half-step everywhere (A2e state) | −0.019, −0.007, −0.015, −0.031, −0.033, −0.067, −0.018, −0.007, −0.005 (std 0.16–0.24) | −0.058, −0.009, −0.003, −0.004 |

The half-step is a first-moment correction and it nearly completes: residual
means are ≤0.07 steps, from decreasing (ReLU-sparse) within-cell mass at
precisely the hops the A6 gauge flags
(`src/mimarsinan/tuning/orchestration/install_resolution/gauges.py:13-46`).

### 3.2 Propagation: drift, then saturation — and the exact fold

With J_h the float Jacobian past hop h: `Δz_L ≈ Σ_h J_h diag(θ_h) ε_h`.

- **First moment (drift).** `E[Δz_L] = Σ_h J_h diag(θ_h) μ_h`. Floor-kernel
  μ_h are all negative and ReLU path gains are predominantly positive: drift
  compounds same-sign in L and floors whole channel populations at low S —
  the shipped pre-half-step state (A0) is at CHANCE on the 9-hop mixer at
  every S ≤ 16 and on 4-hop lenet at S=4 (§6). The closed-form correction is
  the **sequential first-moment fold**: input→output, per hop, fold
  `b_h ← b_h − Ê[z_h^deployed − z_h^float]` with the expectation measured
  through the already-corrected prefix — this subsumes Eq. (4) propagated
  through all upstream hops, encoder and boundary biases included (the
  TTFS-sync instance of the DFQ family the cascade path already uses,
  `src/mimarsinan/spiking/dfq_bias_correction.py`).
  **Sign trap (measured, load-bearing):** `z_h^deployed` must EXCLUDE hop
  h's own intentional offset — the raw pre-activation mean gap is ≈ +1/(2S)
  BY DESIGN (it is the half-step), and folding it away removes the mid-tread
  compensation at every hop: measured collapse 0.9294→0.5948 (mixer S=8).
  With the own-offset exclusion the fold is worth +0.85/+0.12/+0.19 pp
  (mixer S=4/8/16) and ~0 on lenet (whose first moment is already ~0).
- **Second moment (saturation).** Centered per-hop errors are weakly
  correlated across hops (consecutive per-sample mean-error correlations
  0.03–0.13 mixer, 0.005–0.17 lenet) — variance composes near-additively —
  but the composed logit RMSE SATURATES because every staircase re-quantizes
  its input: prefix-quantization law (quantize hops < k only, S=8) mixer
  0.61 → 1.06 (k=3) → 1.30 (k=5) → 1.35 (k=10, flat); lenet 0.29 → 0.70.
  Neither √L nor linear: concave with early-hop dominance. Consequence: the
  entry-side corrections (encoder fold, first-hop θ loading) dominate — and
  measured gains confirm (A1e alone: +30 pp at mixer S=8).

### 3.3 The fixed-S optimal placement problem

The decode grid {kθ/S} is hardware-fixed; reproduction points cannot be
Lloyd-optimized. The free per-channel knobs are exactly scale and offset:

```
min_{θ_c, c_c}  E[(θ_c·Q_S((z + c_c)/θ_c) − relu(z))²]                    (5)
```

solvable per channel by 1-D scans on calibration stats. Measured resolution
of (5):

- **Offsets are solved:** the per-channel MSE-optimal `c_c` equals the
  uniform half-step to 4 decimals on both vehicles and every S (B3 ≡ B1).
  Do not spend a lever here.
- **θ is not:** the shipped q=1.0 scale
  (`conversion_policy.py:67-69,77` `activation_scale_quantile: 1.0`) hands
  the grid to one outlier; the A6-gated 1.0→0.99 deflate
  (`src/mimarsinan/pipelining/pipeline_steps/adaptation/activation_analysis_step.py:218-245`)
  is the first rung of the true solution. The measured optimal quantile falls
  with S and with depth (mixer S=4: 0.995 at entry → 0.95 at depth 8; lenet
  S=4: fc hops at 0.90; S=16: mostly 0.995–1.0) — matching classic uniform-
  quantizer loading theory (granular ∝ θ²/S² vs clip tail mass): **the right
  θ is an (S, marginal)-dependent quantile, not a constant.**
- **Greedy per-hop MSE ≠ end-to-end optimal:** per-channel MSE-θ (B4) can
  lose to accuracy-guided scalar selection at S=4; layerwise distortion
  ignores what downstream amplifies. The cheap end-to-end-aware form is
  coordinate descent on the deployed calibration accuracy (B1: L hops × 5
  quantile candidates × one deployed calib eval each — no gradients).

## 4. Level starvation at low S and the per-channel-θ reallocation

### 4.1 The capacity condition

A staircase hop's channel carries at most `L_c = min(S, S·q_max,c/θ)` usable
levels; the A6 gauge's `median_effective_levels` (`gauges.py:49-58`) is its
empirical form, and the seeded t0_21 gauge read 7/9 hops starved at S=8
(θ 2.95–22.53, starved_mass 0.59–0.74; stdout:223-230). After M4's
cross-layer migration equalizes BETWEEN channels, the residual starvation is
WITHIN channels (heavy-tailed ReLU mass under one grid step; starved_mass
stays ~0.5–0.6 post-equalization, M4 §4) — the regime that Eq. (5) θ-loading
and the half-step dead-zone halving address, and whose hard floor at S=4 is
informational: even the full analytic stack leaves the mixer at 0.8674 vs
float 0.9602 (−9.3 pp), against −2.0 pp at S=8 and −1.0 pp at S=16. The
capacity wall is real at S=4 on a 9-hop starved chain; per-channel θ is the
sanctioned lever, and the QAT's remaining role there is level ASSIGNMENT
(which codes get which inputs), not statistics.

### 4.2 Exact per-channel reallocation (scale-space identity)

`Q_S(z/θ)` depends on (z, θ) only through z/θ: per-channel `θ_c` with decode
`θ_c·Q_S(z_c/θ_c)` refines the grid per channel and is exactly realizable in
the CURRENT deployed arithmetic on perceptron→perceptron edges, with no new
mechanism:

- effective weights already divide per OUTPUT channel when
  `activation_scale` is a vector (`perceptron_transformer.py:100-113`);
- the consumer's `per_input_scales` carries a per-channel vector verbatim
  when lengths match (`src/mimarsinan/mapping/mappers/scale_propagation.py:62-74`);
- SCM core thresholds are already 1.0 in effective coordinates
  (`ir_mapping_class_emit.py:97,171`); threshold-group packing keys exist
  for the export side (`src/mimarsinan/mapping/packing/canonical.py:96-101`);
- the NF container exists: `promote_activation_scale_per_channel`
  (`src/mimarsinan/spiking/theta_cotrain.py:19-40`).

Measured value (zero training): per-channel θ at q=0.99 (A5) reads
0.8633/0.9393/0.9514 on the mixer — the best SINGLE lever at S=8/16 — vs
scalar-θ B1 0.8197/0.9294/0.9418.

### 4.3 The three scalar-collapse seams that break it (file:line)

1. **ComputeOp scale aggregation** mean-collapses vectors to one float:
   `_scalar_node_scale` / `_aggregate_source_scales`
   (`activation_scales.py:49-63`) and the `float(...)` casts in
   `resolve_stage_compute_scales` (`hybrid_execution.py:266-272`).
2. **Boundary-snap normalizers are scalar:** the NF entry quantizer binds the
   scalar `input_activation_scale` (`adaptation_manager.py:104-113`), which
   `perceptron_boundary_scale` mean-folds from sources
   (`scale_propagation.py:105-110`; `src/mimarsinan/spiking/scale_aware_boundaries.py:31-38`
   documents the mean-collapse as today's parity contract).
3. **Axis-flipped (weight-shared) consumers:** where the producer's channel
   axis is the consumer's lane axis (mixer token-fc2 → channel-fc1),
   `assign_per_input_scales` repeats/mean-folds on length mismatch
   (`scale_propagation.py:76-94`) — semantically wrong for per-channel θ.
   Needs a lane-aware flat-axon scale vector; the container supports it
   (length == in_features), the propagation does not compute it.

Landable now without touching 1–3: all channel-mixer hops + token-fc1 (5 of
the mixer's 8 trunk hops — the same adjacency set as M4's exact migration,
because it is the same condition).

## 5. The two-scale WQ lattice × half-step interaction

Two-scale puts the effective bias on `g_b = ratio/scale_w`,
`ratio = ⌈b̃_max·scale_w/q_max⌉`
(`normalization_aware_perceptron_quantization.py:67-83`) — integer-ratio-
snapped so quantized biases stay on the weight lattice (the chip parity
contract). The folded half-step `1/(2S)` is bias mass with projection error
≤ g_b/2. Measured `g_b/(1/(2S))` at 5 bits: mixer S=8 hops 0.21–1.41
(b0_tok_fc1 1.41, b2_tok_fc1 0.95↑), S=4 0.08–0.70; lenet 0.22–0.39.
End-to-end at 5 bits (on B1): mixer −3.9/−1.6/−0.5 pp (S=4/8/16), lenet
≤−0.2 pp — and re-folding the exact half-step after projection changes
+0.6/+0.5/**−1.6** pp: NOT uniformly better; the erosion is not the current
binder (weight rounding is). It scales as `S·g_b ∝ S·b̃_max/q_max`, so it
becomes structural at higher S or fewer bias bits; the exact lattice-free
repair then is to carry the half-step in the comparator/threshold —
`Q_S((z+θ/(2S))/θ)` ≡ the same ceil compare with the threshold shifted half
a grid step — a per-core scalar θ edit, exact for every backend whose
threshold is not integer-locked (SCM: float64; nevresim: n/a for sync).
Two-scale vs one-scale WQ measured identical here (0.7803/0.7821 etc.):
post-AQ effective biases are small, so the M2 crater channel is absent on
these draws — consistent with M2's finding that the crater is draw- and
DFQ-history-dependent.

## 6. Prototype and numbers

Setup: §0. Corrections are CUMULATIVE only where stated; every row is a
one-shot install + closed-form calibration passes — **no training step
anywhere**. Calibration: 4–8k train samples; eval: full 10k MNIST test.

### 6.1 The correction ladder (accuracy)

Mixer (t0_21 shape, float 0.9602):

| # | state | S=4 | S=8 | S=16 |
|---|---|---|---|---|
| A0 | shipped pre-B1: floor kernel, θ = q1.0 | 0.0974 | 0.0974 | 0.1047 |
| A1 | + half-step on non-encoder hops (shipped [5v B1(ii)]) | 0.1475 | 0.4896 | 0.8831 |
| A1e | + half-step on the subsumed ENCODER (**E1, new**) | 0.3061 | 0.7872 | 0.9309 |
| A2 | θ = q0.99 + half-step (shipped B1(i), all-hop proxy) | 0.5332 | 0.8828 | 0.9332 |
| A2e | A2 + encoder half-step | 0.8379 | 0.9234 | 0.9408 |
| B1 | greedy per-hop θ quantile (accuracy-guided) + E1 | 0.8197 | 0.9294 | 0.9418 |
| B2 | B1 + sequential first-moment fold (own-offset-excluded) | 0.8282 | 0.9306 | 0.9437 |
| B3 | B1 + per-channel MSE offsets (replaces half-step) | 0.8197 | 0.9294 | 0.9418 |
| A5 | per-channel θ (q0.99) + E1 | 0.8633 | 0.9393 | 0.9514 |
| B4 | B1 + per-channel MSE-θ (quantile candidates) | 0.8671 | 0.9363 | 0.9502 |
| STACK | B1 + per-channel MSE-θ + first-moment fold | **0.8674** | **0.9401** | 0.9502 |
| NEG | on-chip staircased readout (control, on B1) | 0.7192 | 0.9097 | 0.9369 |
| WQ | 5-bit two-scale on B1 / on STACK | 0.7803 / 0.8602 | 0.9133 / 0.9331 | 0.9370 / 0.9415 |

LeNet5 (float 0.9916):

| # | state | S=4 | S=8 | S=16 |
|---|---|---|---|---|
| A0 | floor, q1.0 | 0.0892 | 0.9085 | 0.9902 |
| A1 | + half-step | 0.9706 | 0.9884 | 0.9915 |
| A1e | + encoder half-step | 0.9823 | 0.9899 | 0.9910 |
| A2e | q0.99 + both folds | 0.9893 | 0.9917 | 0.9921 |
| B1 | greedy θ | 0.9902 | 0.9918 | 0.9915 |
| STACK | full stack | 0.9879 | 0.9907 | 0.9912 |
| NEG | on-chip readout control | 0.9748 | 0.9873 | 0.9903 |
| WQ | 5-bit two-scale on B1 | 0.9900 | 0.9906 | 0.9915 |

Readings:

- The shipped half-step ([5v B1(ii)]) is the single survival lever: +88 pp on
  lenet S=4, +39 pp on mixer S=8 — consistent with the t0_21 campaign
  history (sync AQ entry 0.10 → 0.85 when B1 landed).
- The encoder fold (E1) is the largest UNSHIPPED arithmetic fix:
  +15.9/+29.8/+4.8 pp over A1 on the mixer, +30.5 pp over A2 at S=4;
  +1.2 pp on lenet S=4. One predicate change, QAT-owned like the existing
  fold, no new mechanism.
- θ loading beyond the binary deflate is worth up to +2.4 pp more (B1/A2e vs
  A2 at S=8/16); per-channel θ another +1.0–1.1 pp (A5/B4/STACK).
- The full analytic stack reaches 0.9401 at S=8 vs the campaign's
  QAT-trained 0.9527 (16k-step WQ endpoint; envelope 0.9614 vs this draw's
  0.9602) — i.e. closed-form corrections recover most of what the trained
  ladder recovers, from a 0.10 base, and they COMPOSE with the QAT (they
  raise its entry, its keep-best anchor, and its fixed-S ceiling).
- Greedy B1 at mixer S=4 (0.8197) trails A2e (0.8379): 4k-sample greedy
  selection noise — the ladder should keep the deflate as its floor
  candidate (the fold-in is trivial: include q0.99-pooled in the candidate
  set, which B4/STACK effectively do).

### 6.2 Diagnostics (S=8, A2e state)

Per-hop local error (grid-step units) and error correlation: §3.1–3.2 tables.
Prefix law: mixer 0.61/0.72/1.06/1.20/1.30/1.28/1.30/1.32/1.35/1.35 (k=1..10);
lenet 0.29/0.54/0.61/0.70/0.70. Dtype tie test: 0 argmax flips (f32-cast
parameters vs f64), both vehicles. Greedy-chosen quantiles: §3.3.

## 7. Corrections classified by exactness, and integration points

**Class E — arithmetic-consistency (convention fixes; parity-safe because
both sides move together, QAT reconciles):**

- **E1 — encoder half-step under subsume placement.** Seam:
  `apply_half_step_entry_fold`'s skip predicate
  (`bias_compensation.py:80-83`) becomes placement-aware — skip only
  encoders that do NOT run a staircase (offload/rate-encode); the arming
  site stays the AQ tuner init
  (`src/mimarsinan/tuning/tuners/activation_quantization_tuner.py:51-61`).
  Measured +4.8…+30 pp. Genericity: any vehicle, any S — it is the same
  mid-tread convention every other hop already gets.
- **E2 — explicit readout contract.** Make "readout = host decode" a stated
  invariant (today an accident of `linear_mixin.py:48-59`); add a
  representability check that a final-layer activation does not silently
  staircase the logits (measured cost up to −10 pp at S=4). Alternative for
  strict all-on-chip deployments: per-class bias centering + margin-aware
  θ_cls (not prototyped here).
- **E3 — comparator-side half-step under WQ (watch-item).** When
  `g_b/(1/(2S))` approaches 1 (measure at projection time; §5), carry the
  half-step in the per-core threshold instead of the bias lattice — exact,
  zero bit cost. Do not blanket-refold: measured non-uniform (−1.6 pp once).
- **E4 — per-channel scale plumbing.** The three seams of §4.3 (vector-aware
  ComputeOp scales, vector boundary-snap normalizers, lane-aware
  `per_input_scales`). Pure plumbing; unlocks S2 everywhere.
- **E5 — dtype tie guard.** Keep the existing parity gate as the P5 monitor
  (measured clean today; structurally unguaranteed).

**Class S — statistical (closed-form from calibration stats; no training):**

- **S1 — quantile-descent θ (per hop).** Generalizes the shipped binary A6
  deflate (`activation_analysis_step.py:218-245`) to a per-hop candidate
  scan {0.90, 0.95, 0.99, 0.995, 1.0} scored on the deployed calib forward,
  sequential input→output; keep the current deflate value in the candidate
  set as the floor. Integration: the same `install_resolution` capture that
  feeds A6 (`src/mimarsinan/tuning/orchestration/install_resolution/capture.py`)
  already carries per-channel q99 + the forward. Worth up to +2.4 pp beyond
  the deflate; makes θ selection S-aware (the measured optimum falls with S
  and depth), removing the per-S hand constant.
- **S2 — per-channel θ (level reallocation).** Container exists
  (`theta_cotrain.py:19-40`); realizable today on matching-axis edges
  (§4.2/4.3); +1.0–1.1 pp on the mixer at every S, +4.4 pp at S=4 vs scalar
  B1. Fold into consumers via `per_input_scales`; export via threshold
  groups or row scaling before WQ.
- **S3 — sequential first-moment fold.** The exact Eq.-(4) fold, one
  deployed + one float calib pass per hop, MUST exclude each hop's own
  offset (§3.2 sign trap). Integration: the DFQ family seam
  (`spiking/dfq_bias_correction.py` / `distribution_matching.py`), gated
  `is_synchronized_ttfs`, run once at AQ install (before endpoint recovery,
  so the QAT trains from the corrected state). Worth +0.1…+0.9 pp; also
  corrects encoder/boundary systematics generically.
- **S4 — offsets beyond the half-step: CLOSED (negative result).** The
  uniform +θ/(2S) is per-channel-MSE-optimal on both vehicles at every S
  (B3 ≡ B1 to 4 decimals). No lever here; document to prevent re-derivation.

**Ordering note.** E1/S1/S2/S3 all act at AQ-install time, BEFORE the
16k-step endpoints; they raise the entry, the keep-best anchor, and the
fixed-S ceiling the endpoint converges to. Nothing here spends training
budget or touches A1–A6 contracts; every fold is idempotent-markable exactly
like the existing half-step flags.

## 8. Genericity and limits

- **No workload constants.** Every correction is derived from the two kernel
  identities plus calibration statistics of the vehicle at hand; the
  mode-gates are the existing predicates (`is_synchronized_ttfs`,
  `uses_ttfs_floor_ceil_convention`,
  `src/mimarsinan/chip_simulation/spiking_semantics.py:105-118`). Measured
  on two vehicles with opposite failure profiles (9-hop starved mixer;
  4-hop healthy lenet): the corrections lift the fragile vehicle by tens of
  pp and cost the healthy one nothing (lenet stays ≥0.988 in every corrected
  row).
- **Any S:** the ladder was measured at S ∈ {4, 8, 16}; every correction's
  gain is monotone in starvation (largest at S=4) and none regresses at
  S=16 beyond noise. The S=4 residual (−9.3 pp to float on the mixer) is the
  §4.1 capacity wall — the honest boundary of statistics at fixed S; the
  levers there are per-channel θ (S2) and M4's migration, then QAT's level
  assignment.
- **Any regime:** Theorem 1 pins deployment ≡ trained composition; the P1–P6
  seams are the complete list of ways that equality can silently break, each
  now named with its guard or its measured cost. The corrections are
  install-time and compose with (never replace) the shipped QAT ladder.
- **Honest limits.** (i) The prototype's value-domain forward mirrors the
  deployed stage schedule but not the packing (coalescing/splitting is
  scale-neutral, so this should be exact; unverified here). (ii) The
  campaign's 0.9527 includes pruning (−1.05 pp) and the pretrain envelope
  (0.9614 < bar) — both out of scope by mandate (no train-more levers);
  this memo's corrections address the conversion loss component and the
  install floor. (iii) Greedy S1 measured with 4k calib samples has ~0.7 pp
  selection noise (the B1<A2e inversion at mixer S=4); production form
  should score on the full calib cache. (iv) MNIST-band evidence only;
  the mechanisms (kernel identities, moment law, capacity condition) are
  data-free, but the pp-sized gains are not — tier-1/2 replication is the
  next measurement.
