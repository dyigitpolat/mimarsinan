# The conversion boundary algebra: one wire currency, derived everywhere

**Question.** On the offloaded ViT (12 host-LayerNorm → on-chip fc1 seams) the
ANALYTIC staircase composition reads 0.79 while every TEMPORAL representation
(NF walk, HCM torch twin, nevresim) reads ~chance — and the temporal
representations agree with each other bit-for-bit in argmax (measured: 32/32
agreement, logit corr 1.0000 on identical inputs). Tier-0 subsume cells are
simultaneously bit-exact across all representations. This memo derives, from
the deployed kernels, the algebra every representation must share at a
host↔chip boundary; identifies the exact convention disagreements in the
code; proves why they are invisible on subsume/post-ReLU topologies and
catastrophic on offload/signed ones; and states the unification the code must
reflect. Everything is arithmetic-consistency; no training lever appears.

## 1. Objects: the two scales of a boundary edge

A hybrid deployment graph alternates host ops H (float, signed, arbitrary
scale — LayerNorm, attention, bare Linear/Conv, pooling, relays) and on-chip
segments C (spiking cores; codes are non-negative and bounded). Per node n
there are TWO scales, and every defect found is a confusion between them:

- **κ_fold(n) — the consumer-fold currency.** The scale the consumer's weight
  fold multiplies back in: the mapper emits `W~ = per_input_scales · W / θ_out`
  (transformations/perceptron/perceptron_transformer.py:85), so a consumer of
  n expects its input in units of `value/κ_fold(n)`. The propagation
  (perceptron → θ_out; every other node → aggregate of sources) exists twice,
  consistently: NF-side `read_boundary_out_scales`
  (spiking/scale_aware_boundaries.py:20-40) and IR-side
  `compute_node_output_scales` (mapping/support/activation_scales.py:66-88).
- **κ_buf(n) — the buffer gauge.** What the runtime buffer actually stores:
  `state_buffer[n] = value(n)/κ_buf(n)`. Neural cores store counts/T
  (κ_buf = θ_out); host value-ops store the value itself (κ_buf = 1). Today
  κ_buf is only IMPLICIT, reconstructed by `boundary_normalization_scales`
  (spiking/segment_boundary.py:51-88) — and reconstructed wrongly for a plain
  host op fed by a neural producer (§4, V-B).

Boundary operators, in terms of the two scales and the sign shift σ
(value-domain, per-channel, s = clamp(−min v, 0), calibrated over validation
data — mapping/support/negative_boundary.py:71-88):

```
encode (host value v → wire):   E(v)   = grid_T( clamp( v/κ_fold + σ/κ_fold, 0, 1 ) )
decode (wire r → host value):   D(r)   = r · κ_buf                      (gather-lift)
entry  (wire → neural charge):  wire   = buffer / divisor,  divisor = κ_fold/κ_buf
bias bake (σ absorption):       B'     = B − W_eff · (σ/κ_fold)
```

## 2. The identities (exactness = commutation)

Deployment is exact iff all three hold at every site, in every representation
(train-QAT entry op, NF walk, HCM torch twin, IR/nevresim):

- **I1 (seam identity).** The TRAINED entry function equals the deployed
  round-trip: `train_entry(v) = κ·(grid_T(clamp(v/κ + σ/κ)) − σ/κ)` — same κ,
  same σ, same grid, same rounding. In-range it is the identity up to grid
  noise; out-of-range effects (saturation, negative deletion) are identical
  between training and deployment, so the QAT adapts to the true deployed
  function.
- **I2 (hop identity).** The T-step temporal dynamics of an on-chip hop equal
  the analytic staircase: `temporal_T(W̃, b̃, train) = Clamp_T(F(Σ w̃ᵢnᵢ + T·b̃))`
  under the Theorem-2 preconditions (uniform trains, subtractive reset,
  half-step fold, window coverage, no terminal overshoot — see
  lif_deployment_exactness.md §2-3). The violation ledger there measures these
  terms at ≤ few pp, one-sided — hop dynamics are NOT the chance-collapse
  term.
- **I3 (host identity).** Every host op computes on the domain it was trained
  on. Host parameters (LN γ/β, attention projections) were trained on VALUES;
  LayerNorm and softmax are not positively homogeneous for per-channel scales:
  LN(r) ≠ LN(r·κ) unless κ is a single positive scalar. Only for positively
  homogeneous f (f(αx) = α·f(x): MaxPool, relays, concat, bias-free Linear
  with scalar κ) may a representation legally feed the RATE.

## 3. Statistical composition laws (why violations differ in kind)

- **Grid noise** per seam is bounded, |ε| ≤ κ/(2T), and symmetric (mid-tread
  round) ⇒ zero-mean; across K independent seams the composed perturbation
  scales ~√K·κ/(2T). At T=32, K=12 this is percent-level — the benign
  residual family measured on tier-0.
- **Convention mismatches** (currency, normalization, σ) are DETERMINISTIC
  per-seam biases: feeding LN a rate saturates/rescales every channel;
  clamping signed values at 0 deletes E[min(v+σ, 0)] — for zero-mean LN
  outputs ≈ half the signal mass — at EVERY seam. Deterministic biases compose
  linearly (and multiplicatively through the following nonlinearity), so
  K=12 seams take the composition to chance. This is the observed signature:
  all temporal representations agree with each other (they share the same
  wrong convention) and disagree with the analytic one.

## 4. The measured violations (code-anchored)

| # | Violation | Analytic / train side | Temporal side |
|---|---|---|---|
| V-A | Host-op input currency (I3) — PRIMARY | host ops receive VALUE = rate·θ (spiking/segment_policies.py:184-204) | host ops receive RATE = counts/T with (1,1) compute scales (chip_simulation/hybrid_run/hybrid_execution.py:266-272 `resolve_stage_compute_scales(apply_ttfs=False)`; segment_policies.py:155,167; simulation_runner/hybrid.py:255-263) |
| V-B | Entry normalization κ (I1) | ChipInputQuantizer divides by θ_in before clamp (models/nn/activations/autograd.py:197-204); the weight fold expects value/θ_in (mapping/mappers/scale_propagation.py:97-102) | `boundary_normalization_scales` returns IDENTITY for a plain host producer (segment_boundary.py:77-84: only wrapper mappers carry `perceptron_wrapped_activation_scale`) ⇒ clamp(raw value). The rate-normalizer table and the weight-fold currency are two different propagation functions that disagree exactly at plain-host-op boundaries |
| V-C | NF `train_of` clamp domain (I1) | SSOT `normalize_boundary_value` divides first (spiking/compute_boundary.py:18-35); the encoding-perceptron branch divides first (segment_policies.py:126-134) | segment_policies.py:105-108 clamps the RAW value then multiplies by θ after |
| V-D | σ asymmetry (I1) | training: hard clamp-at-0, no σ (autograd.py:203) | every temporal path adds +σ (value-domain; segment_forward.py:166-169, models/spiking/hybrid/lif_step.py:279-299, hybrid.py:227-232, ttfs_executor.py:194) — calibrated and baked at Soft Core Mapping, AFTER all training (soft_core_mapping_step.py:509-532); and where κ_fold ≠ 1 the raw σ is applied post-normalize, off by the κ factor |

Why the hop dynamics are exonerated: the temporal representations agree with
each other bit-for-bit while disagreeing with the analytic family — a hop-
dynamics term (V1-V6 of the exactness memo) would perturb nevresim and the
torch twin differently. A shared-convention seam term produces exactly the
observed all-temporal agreement.

## 5. Tier-0 inertness (why the bug was invisible)

The unification below changes behavior only where at least one of:
(a) a value-op gathers a source with κ_buf ≠ 1;
(b) a value-op's entry divisor changes 1 → κ_fold;
(c) σ ≠ ∅ with κ_fold ≠ 1.
All three require a plain, non-homogeneous host ComputeOp fed by a neural
producer — the offload/ViT/mixer topology. Subsume boundaries are
wrapper-mapper outputs: the wrapped divisor θ_w is read identically by both
propagation functions (activation_scales.py:33-46 ↔ segment_boundary.py:73-75)
so RATE ≡ VALUE/θ coincide; and post-ReLU outputs make σ = ∅. The homogeneous
host ops tier-0 does route (MaxPool, relays) commute with the gauge, so the
`wire_transparent` classification keeps today's exact code path — inertness by
PATH identity, not merely by value equality. (Converted from argument to
measurement by probe P0-3.)

## 6. The unification (what the code must reflect)

One classifier assigns each host op a gauge (the mixed-gauge WireCurrency):
`owns_domain` (ScaleNormalizingWrapper — exists, untouched);
`wire_transparent` (positively-homogeneous allowlist, membership proven by an
f(αx)=α·f(x) property test; κ_buf = agg(src); no transcode — today's path);
`value_op` (LayerNorm, GELU/softmax/attention, biased Linear, patch-embed
conv, wrapped encoders; κ_buf = 1; gather-lift by κ_buf(src); entry divisor
κ_fold; σ as σ/κ_fold). κ_fold has ONE propagation (the IR table generalized
vector-preserving; the NF twin under a permanent nodewise-equality contract
test); `boundary_normalization_scales` becomes the derived view
divisor = κ_fold/κ_buf. σ is calibrated once at the AQ install seam (frozen,
verified at SCM) so the trained entry op is the exact deployed composition
(I1). TTFS is the all-wire-gauge instance of the same algebra (its
`apply_ttfs` scalar transcode ≡ gather-lift with κ_buf ≡ κ_fold): the LIF
walk converges to the convention `TtfsSegmentPolicy` already implements —
mode-generic by construction.

## 7. Probe results (2026-07-17, `currency_unification_artifacts/probe_scales.py`)

**P0-1 — V-B measured.** All 12 ViT segment entries carry κ_fold ≠ 1:
`input_activation_scale` = 2.640, 1.449, 1.099, 1.028, 1.126, 1.281, 1.469,
1.591, 1.860, 2.225, 3.250, 4.144 — while the temporal divisor for their
plain-LayerNorm producers is 1 (code-verified hole at
segment_boundary.py:77-84). The temporal walks therefore feed the on-chip
entry a value mis-scaled by up to 2.6-4.1× versus what the weight fold and
the trained quantizer expect. The NF-side κ_fold table
(`read_boundary_out_scales`) resolves the same seams consistently with the
entry scales — confirming the two-table split is IR/temporal-side only.

**P0-2 — σ is real, and its harm is CAPACITY, not mismatch.** Direct LN-module
hooks measure ~50% negative mass at every seam (ranges to ±23 at the widest).
The value observed AT the entry perceptron is exactly [0, κ] — the
`ChipInputQuantizer` amputates the negative half identically in training and
in the plain walk, so the QAT ADAPTED to the amputated function (the analytic
0.79 includes this cost). Consequence for the design: σ (V-D) recovers
capacity; the chance-collapse discriminator between analytic and temporal is
V-A (rate-vs-value into non-homogeneous host ops) + V-B (the κ divisor hole).
This ordering is testable in the P1 fixture and falsifiable in P4: fixing
V-A/V-B alone should already lift the temporal walks to the analytic band,
with σ then closing the remaining amputation gap.

**P0-3 — tier-0 inertness measured (t0_03).** `node_output_shifts` = 0 entries
(σ tables empty); IR = 1292 NeuralCores + 6 ComputeOps. The remaining
inventory item (per-cell op types for the `wire_transparent` allowlist across
all tier-0 IRs) is folded into P1's fixture work.

**Open cell for P1 (honest).** One instrumentation anomaly remains unexplained
(a perceptron-level pre-hook read [0,κ] where the mapper chain hands over a
signed tensor); it does not affect the conclusions above (both readings are
consistent with the quantizer clamp semantics), but the P1 micro-fixture must
make every call site loggable so the exact plain-walk seam composition is
pinned rather than inferred.
