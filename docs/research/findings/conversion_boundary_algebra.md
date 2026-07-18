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

## 8. P1 results (2026-07-17, `tests/unit/spiking/test_wire_currency_contract.py`)

The micro-fixture (on-chip LIF θ=1.7 → plain signed LayerNorm → on-chip LIF,
offload placement) reproduces the production signature at unit scale and
sharpens three claims:

1. **The temporal family is internally consistent — T2 is a GREEN lock, not
   RED.** NF walk and the HCM twin agree to 0.000000 on the signed seam
   (mirroring the 32/32 argmax agreement on the large-backbone cell). The
   plan's T2-RED prediction is refuted at fixture level: the two sides do not
   disagree with each other, they share the same wrong convention. T2 is now
   the joint-movement lock — it must hold before AND after the unification.
2. **The V-B hole manifests as ABSENCE.** `boundary_normalization_scales` on
   the fixture returns an EMPTY table (no entry at all for the plain LN
   producer); the divisor defaults to the identity downstream. The seam
   machinery even self-reports the consequence ("neural stage receives
   negative boundary values … the [0,1] spike-encode clamp drops the
   residual").
3. **Deterministic bias vs grid noise, measured.** At T=32 the temporal walk
   deviates from the value-domain reference by 0.225 max (batch-mean far
   above κ/(2T)=0.027) while NF↔HCM sit at 0.0 — exactly the §3 composition
   law (mean-shifting convention bias vs zero-mean grid noise). The strict
   xfails (T1, T3, T4-today, T5) encode V-A/V-B/V-C/V-D; the GREEN locks
   (T2, T4-required, T6 homogeneity, T7 subsume, NF↔IR κ_fold nodewise) pin
   what the fix must preserve.
4. **The P0-2 anomaly is explained by V-C.** The walk's own boundary
   re-encode clamps the raw value to [0,1] and multiplies θ back BEFORE the
   consumer perceptron sees it (`train_of` clamp-then-scale) — a pre-hook on
   the entry perceptron therefore reads [0,κ] even though the mapper chain's
   host output is signed. The [0,κ] observation was the V-C composition
   itself, measured.
5. **I2 kernel twin locked.** `TestLifAnalyticTemporalTwins`
   (test_wire_semantics.py): the deployed `lif_fire_and_reset` cycle loop
   under constant drive equals `lif_count_staircase` bit-for-bit in float64,
   both compare modes, off exact-integer ties of T·z/θ; at an exact tie the
   accumulated membrane sits 1 ulp off the one-shot product and may legally
   flip one count (measured: |Δcount| ≤ 1, tie set only).
6. **Tier-0 `wire_transparent` inventory enumerated (31 cached IRs).** Host
   op module types: MaxPool2d(34), AdaptiveAvgPool2d(8), ComputeAdapter(4),
   Linear(22), Perceptron(12), Conv2DPerceptronMapper(15),
   ScaleNormalizingWrapper(5). Neural-FED plain ops are exclusively
   homogeneous (MaxPool2d 26, AdaptiveAvgPool2d 8, ComputeAdapter 4) plus 10
   bare biased Linears that are ALL TERMINAL heads (post-exit-decode,
   value-domain in both representations, never re-encoded). **No tier-0 cell
   contains a plain host op that is both neural-fed and re-encoded** — the §5
   inertness argument is now a per-cell measurement.

## 9. P3 realization (2026-07-17): arming, not lifting — and two measured refinements

The unification landed through the `owns_domain` channel rather than
hand-lifting in every runner: `mark_wire_value_ops` classifies each host
ComputeOp (non-homogeneous AND re-encoded into on-chip segments), and the
ComputeOp wrap policy arms the ScaleNormalizingWrapper for marked ops even at
uniform scalar source scales — the single-source uniform-θ skip was V-A/V-B's
mechanical root (`apply_compute_op_scale_policy` assumed scale transparency
that only holds for positively homogeneous modules). One emission gives every
representation the value-domain composition for free: the NF twin
(`forward_scale_normalized`), the HCM torch flow, nevresim, SANA-FE, and Lava
all execute the same emitted module; the neural entry's clamp becomes
divide-first at the trained κ because the wrapper's `output_scale` IS the
consumer's fold currency (same `mean_source` propagation as
`input_activation_scale`). Terminal heads, host-consumed ops, and encoder-fed
ops keep the trained-through convention; tier-0 is inert by the §8.6
measurement.

Two design cells were REFUTED by measurement during P3 and corrected:

1. **The bake needs no κ-conversion.** The consumer's effective weight
   already folds `per_input_scales` (= κ_fold), so the historical bake
   `B − W_eff·σ` with buffer-unit σ moves the charge by exactly
   `W·(κ·σ_wire)` — the value-preservation identity — for armed (wire-buffer)
   and plain (value-buffer) producers alike. An explicit `σ·κ` conversion
   double-counts (measured: bias delta / W@(κσ) ≡ 1.70 = κ with the
   conversion, 1.00 without). The gauge conversion lives in the FOLD, once.
2. **The entry quantizer is σ-free.** σ is producer-walk-applied
   (`_negative_shift` lifts the buffer before the entry sees it) and
   consumer-bias-baked; a quantizer-side σ would double-apply. The P2 σ
   parameter was removed; V-D-training closes by ORDERING instead:
   `ensure_negative_boundary_policy` runs at the AQ install seam (offload
   only), so the exact-QAT trains through the shifted boundary and the SCM
   invocation degrades to the drift verifier (idempotent: re-calibrating the
   shifted walk stamps nothing new).

Post-P3 contract-suite state: T1 (armed wrapper owns the seam), T2 (NF==HCM
held through the flip — the joint-movement proof), T3 (temporal within the
grid envelope of the value composition: pre-fix 1.6× over the bound, post-fix
3× under), T4 (σ policy preserves the signed band through the real
calibrate→stamp→bake machinery), T5 (trained entry == deployed seam), T6/T7
(homogeneity + subsume byte path) — all plain green asserts, no xfails.

## 10. P4 cells (2026-07-17): what the torch-mixer repro flushed out

The failure-mode cells (t0_30 lif offload, t0_32 sync offload; torch-converted
non-core mixer, patch 4x4 c32 fc64) turned three latent defects into measured
fixes before any accuracy read:

1. **The subsume minimal pair is structurally impossible** for this vehicle:
   under `subsume` every torch-mixer perceptron becomes a host ComputeOp
   (measured 0% on-chip; the on-chip-majority validity gate fires). t0_31 is
   retired; the pre/post-fix A/B on t0_30 itself carries the isolation.
2. **Mixed-axis sigma is not bias-compensable.** A rank-3 seam value
   (token x channel) yields per-position minima; a per-output bias cannot
   compensate a shift that varies along the consumer's batched axis — and
   mixer blocks mix BOTH axes. Fix: per-channel sigma for rank<=2 boundary
   values (the exactly-compensable case), SCALAR sigma for rank>=3
   (axis-invariant). The PRE-fix pipeline hard-crashes on exactly this
   (measured at 812820f0: LIF Adaptation cliff to 0.46 — the crater
   signature — then the 512-vs-16 bake shape error): the baseline verdict
   for t0_30 is STRUCTURAL FAILURE, not a low number.
3. **Host chains needed no fail-loud.** The old walk raised on a
   ComputeOp->ComputeOp seam ("no consuming perceptron bias to compensate");
   under the producer-side lift the host consumer reads the lifted value in
   every representation, so the walk now skips it — only the chain's last op
   (the one a perceptron re-encodes) stamps + bakes, and a host-only-consumed
   op is never a lossy boundary. The ON mechanism now handles the
   LN->transpose->fc chains the OFF mechanism previously had to subsume.

### 10b. The two cell verdicts (2026-07-18)

4. **The I1 capacity condition is load-bearing.** With sigma stamped at the
   AQ install, the shifted wire is (v - min v)/kappa: kappa must cover the
   seam's value-range WIDTH. Theta-pass-through kappa (~2.5) under a range
   of ~7 saturated every armed seam to a constant — AQ flatlined at EXACTLY
   0.1135 at S=8 AND S=32 (deterministic, S-independent, gradient-free
   through the saturated clamp, unrecoverable by training). Fix: the AQ seam
   calibrates each armed op's observed range and lifts
   ``boundary_traffic_scale`` (the existing one-scale-to-both-walks seam) to
   the range width BEFORE the sigma policy; the wrapper s_out, the weight
   fold, and the entry quantizer then agree on the covering currency.
5. **The sync/torch-mixer gap is the TTFS analog, and pre-existing.** With
   arming mode-gated OFF for TTFS wires (they own their transcode via
   apply_ttfs), torch<->deployed parity still reads 0.0000 on this vehicle:
   the per-op scalar apply_ttfs lifts meet the same mixed-axis/per-instance
   host-Linear semantics the rate path just fixed. Recorded as an open
   program item; the t0_32 cell is retired from tier-0 (pass-all gate) with
   the repro one git-show away.

### 10c. AQ-seam install bisection (2026-07-18) — the opt-in decision

Component-wise replay on the cached pre-AQ mixer model (val acc per step):
post-shift baseline 0.9648 → theta promotion 0.9648 → half-step fold 0.9644
→ **entry quantizers 0.1704** → +cover 0.1846 → +sigma policy 0.1079. The
wreck is the ENTRY QUANTIZER INSTALL itself (hard [0,kappa] clamp on signed
multi-host-chain inputs), and sigma+cover do NOT restore the composition —
the value-preservation chain has a residual defect on this topology
(suspects: the bake/lift interplay across residual adds; per-instance _col
splits vs the scalar sigma; open). The AQ flatline (exactly 0.1135 at any S)
is the QAT's inability to climb this install.

Consequence: `lif_aq_negative_boundary` (NEW, default OFF) gates the AQ-seam
sigma+cover — the historical SCM-time sigma stays; the exact-QAT climbs the
quantizer install alone (the offloaded-ViT-proven path: 0.33→0.79). Arming
V-D-training-closure stays available for A/B once the install composition is
debugged on the repro cell.

### 10d. Parity-gate replay (2026-07-18) — sigma is CONSUMER-side; the producer lift is refuted

Instrumented replay of the SCM torch<->deployed-sim gate on the cached mixer
artifacts:

- WITHOUT the sigma policy: **agreement 1.0000** (torch 0.9219 == sim 0.9219,
  identical argmax distributions) — the armed composition (V-A/V-B/V-C fixes,
  terminal arming included) is CORRECT end-to-end.
- WITH the sigma policy: agreement 0.0000; BOTH sides collapse to different
  constant argmaxes (torch acc 0.156 @ all-1s, sim 0.016 @ all-8s).

Derivation of the wreck: a stamped producer's sigma lift flows through the
residual adds into the TERMINAL host chain (mean -> classifier) where no bake
exists to compensate — the P3c "skip host consumers" walk change silently
traded the (correct) fail-loud for a broken value chain, and the P3c
PRODUCER-side mapper lift contaminates every host reader by construction.

**The corrected architecture (next session's implementation):** sigma applies
exactly where value ENTERS an encode domain — consumer-side:
1. deployed: stage-input assembly shifts (exists) + host-gather lifts ONLY
   for sigma-baked consumer modules (subsumed perceptrons), never plain hosts;
2. training: the sigma-aware entry quantizer, NO-TAIL form
   ``kappa*snap(clamp((v+sigma)/kappa))`` paired with the consumer bake
   (Perceptron.forward applies input_activation, so the walk and the plain
   forward get it for free);
3. the producer-side mapper lift (P3c) is REMOVED; host consumers and
   terminal paths read RAW values in every representation;
4. calibration then records RAW minima again — the effective-minima /
   delta-bake semantics revert with it (the once-flag idempotency contract
   returns).

Blast radius meanwhile: tier-0 is sigma-free (measured); the offloaded ViT's
LN seams feed entries DIRECTLY (the T4-covered single-seam case — no host
chains between the sigma op and its consumers), so BA-P5 on t2_04 does not
wait on this; the mixer cell fails LOUD at the parity gate (no silent
corruption path exists).

### 10e. The sigma-scope law LANDED + the cell verdict (2026-07-18)

The law (plan sec.1): sigma is a property of the (producer -> neural-entry)
EDGE. Landed as (a) `trained_entry_boundary` — a boundary whose non-host
consumers all carry a trained entry quantizer is the QAT's own clamp, sigma
skips it; (b) the ONE scope filter in `apply_negative_boundary_policy` also
drops never-encoded (host-only-consumed) boundaries — the ON path
historically over-stamped them, which is the only reason host chains had to
fail loud; (c) the producer-side lift reverted to the proven consumer/walk
semantics (raw minima, once-flag bake + drift belt, pre-scan fail-loud
narrowed to a sigma-op feeding BOTH an unquantized entry and a host op).

Empirical proof: the parity replay WITH the sigma policy active reads
**agreement 1.0000** (torch 0.9219 == deployed 0.9219). The t0_30 cell then
ran the ENTIRE pipeline: AQ retained 0.9559, LIF 0.9143, WQ 0.9238,
**deployed target metric 0.923**, SCM parity green, Loihi spike parity 1.0
over 12 cores, SANA-FE completed — zero tracebacks. The crater->fixed
ledger closes: pre-fix (812820f0) = structural crash after a 0.46
conversion cliff; post-fix = a green deployment within 2*SE of its WQ read.

The agreement-triage law (plan sec.2) is hereby the debugging SSOT for
deployed divergences: exactly-0.0 => deterministic convention/offset defect;
~1/K => decoupled garbage; 0.9x => noise family.

### 10f. The signed-seam capacity ledger (2026-07-19) — the next phase's design

Measured on the t2_04 offloaded-ViT AQ install (S=32, offload, armed seams):

| Install variant | AQ entry read | Verdict |
|---|---|---|
| pre-arming (inconsistent NF) | 0.33 | trains the WRONG function (deploys at chance) |
| armed, sigma-free trained clamp | 0.06 | EXACT train==deploy; ~50% of LN mass amputated |
| armed + full-width cover | 0.0146 | REFUTED: kappa=range-max destroys grid resolution (kappa/T ~ 1.2 per step) |

Two laws extracted:
1. **Capacity vs resolution**: the entry kappa must cover the encodable band
   AND keep kappa/T at the signal scale — kappa is a QUANTILE of the
   (shifted) range, never the max (house convention:
   activation_scale_quantile / FANIN_TRAFFIC_QUANTILE).
2. **The signed-seam completion**: the sigma-free trained clamp is exact but
   capacity-lossy where the seam has large negative mass (ViT LN ~50%). The
   complete fix folds sigma INTO the armed op's own function, installed
   BEFORE training: ``SNW'(x) = (f(x*s_in) + sigma)/s_out`` with the value
   twin in the plain forward and quantile-kappa. Every consumer (terminal
   heads included) then sees sigma uniformly in every representation and the
   QAT adapts — pre-training producer-side sigma is sound precisely because
   training absorbs it; POST-training sigma stays banned (the sigma-scope
   skip). This supersedes both the deleted AQ sigma half and the full-width
   cover; it is the designed next phase, to land tests-first with the T4
   family extended to the SNW-offset composition.

Meanwhile the honest t2_04 measurement runs sigma-free (the exact-but-
amputated baseline): its AQ ceiling is the quantitative cost of the missing
signed-seam capacity, the A/B target for the next phase.

### 10g. Signed-seam capacity CONFIRMED (2026-07-18) — sigma-in-the-op measured

The R1 install landed (armed-only stamping; quantile sigma + quantile kappa;
consumer bakes by shift response) and the ViT A/B read the decisive number:

| t2_04 AQ entry (full acc) | config |
|---|---|
| 0.0146 | full-width cover (kappa=max) — REFUTED, destroys resolution |
| 0.06 | sigma-free trained clamp — exact but capacity-amputated |
| **0.5956** | **sigma-in-the-op** (entry post-recovery 0.6046, retention armed) |

A 10x capacity recovery, inside the predicted 0.3-0.6+ band: the seam loss
was signed-band amputation, not resolution or training capacity.

Three implementation laws proven on the way (each fail-loud first):
1. **Armed-only stamping.** Only ops with wrap slots (per_source_scales) may
   carry output_value_offset — the wrapper is what transports the offset into
   every deployed representation; an unarmed stamp (patch_embed) is a
   train/deploy split by construction.
2. **Shift-response classification of host consumers.** Scalar-shift-
   equivariant ops (pools, mean/select/flatten family, residual ``add``,
   ``getitem``) pass sigma through; LayerNorm absorbs it; bias carriers bake
   it: Linear via ``bias -= sigma.W.sum(dim=1)`` and packed self-attention via
   ``in_proj_bias -= sigma.in_proj_weight.sum(dim=1)`` (q=k=v arrive from the
   same seam, deduped to one edge — ONE packed bake). ``cat`` stays fail-loud:
   a partial-slice shift is not bias-compensable downstream.
3. **The bake law as a test.** ``f_baked(v + sigma) == f(v)`` locked for every
   carrier (TestHostBiasCarrierBake); non-carriers refuse loudly.

### 10h. ViT e2e (Q4): downstream fixes + the honest infra wall (2026-07-18)

Past the AQ σ-in-the-op recovery (§10g, 0.596), the σ-armed ViT ladder
surfaced two latent gaps and one hard infra wall. The gaps (both fixed
tests-first, gate 8250 green):

1. **`ScaleNormalizingWrapper` was not transparent to the wrapped module's
   calling convention.** An armed MultiheadAttention op crashed the wire twin
   (`need_weights` kwarg dropped; the `(attn, weights)` tuple return applied
   to scale/offset unselected). SNW now carries `module_kwargs` + a scalar
   `output_index`, selects the tuple element BEFORE offset/scale, and
   `forward_scale_normalized` delegates the selection to the wrapper (no
   double-index). The value twin already did both; the twins are now aligned.
2. **MHA joined `_bake_shift_into_host_bias`** as a σ bias carrier (§10g).

**The wall is VRAM/wall-time scaling of the offloaded whole-backbone LIF
genuine forward at S=32 — NOT the algebra.** Ledger:

| symptom | cause | lever |
|---|---|---|
| OOM, GPU 0 at 1.1 GiB free | other tenants saturating the shared GPU | pin to a free GPU |
| OOM 16.30 GiB, process holds 93 GiB | segment_forward `values` dict = every layer's [S=32, batch, seq, hidden] at eval-batch 512 | `deployment_parameters.batch_size` (nested block; the EVAL batch, NOT `tuning_batch_size`) → 64 clears it (0 OOMs) |
| every attempt dies at ~917s, no traceback, no LIF cache | the box's ~952s session reaper kills the process mid-step; each attempt restarts LIF from the AQ cache | run under a USER-started tmux (immune) — `scratchpad/t2_04_tmux_run.sh` |

The bottleneck within a window is the **genuine spike EVAL** (eval_n_batches=39
× S=32 over the whole backbone), not training — reducing `tuning_budget_scale`
did not help. Every downstream step (TTFS/Noise/WQ/mapping/SANA-FE) has the
same slow genuine eval, so the ladder cannot chain under the reaper; the
immune-tmux run is the path to the true e2e number.

**Not a collapse:** the LIF entry `best_full_acc=0.011` (vs `post_acc≈0.774`
analytic) is the genuine spike forward at scale=1.0, which reads chance BY
DESIGN — LIF adaptation is what recovers it (fully lossless on MNIST). Whether
it recovers for the offloaded ViT at S=32 is UNMEASURED: the adaptation
training never completed a full rung under the reaper. That measurement is the
one open Q4 item, gated on the tmux run.
