# The absolute host prefix has no entry gauge: why a streamed leaf-ViT deploys a constant-input core

**Question.** The mixed wire/absolute seam repair (`establish_wire_gauge` /
`establish_gauge_for_mixed_domain_seams`) unblocked LIF Adaptation on a
transformer residual, and a tier-0 cell (`t0_52`, lif/streamed, leaf ViT,
offload, wq5, S=4) was authored to cover it end to end. That cell reaches Soft
Core Mapping and then fails the FATAL streamed NF↔SCM exactness certificate on
**94% of neuron-windows**. This memo answers what actually diverges, proves it
is not the seam and not a currency-coherence break, and states the one change
that would fix it.

**Headline.** The seam is exact. The divergence is at the **entry to the first
neural core**, which sits UPSTREAM of the seam, and it is not a timing or
tie-breaking effect: the deployed core is **silent** (mean window count 0.136 of
T=4) where the twin **saturates** (3.343 of 4). The cause is arithmetic and
exact: the perceptron's bias, after the SCM-time negative-boundary shift bake,
requires **up to 230.3 weight-grid units** on a 5-bit grid whose bias register
holds **±15**, and `chip_quantize._scale_hardware_bias` **saturates it
silently** on 87 of 96 neurons — a mean of 51.7 grid units of charge deleted
against a threshold of 33.0. The bias is that large because the negative shift
(+3.76 value units) is applied to a boundary whose wire gauge is **1.0** — the
default `input_data_scale`, inherited unchanged through a ten-node ABSOLUTE
host prefix that no mechanism ever measures. The same unit mismatch means
**99.5% of the entry rates clamp to exactly 1.0**, so even with a perfectly
represented bias this cell deploys a core whose input barely depends on the
sample. One missing quantity — a measured entry gauge for an absolute host
prefix — produces both symptoms.

**Status.** The `t0_52` row is WITHDRAWN from `templates/generate.py` (the tier
matrix is green at 37 tier-0 cells, as before). The seam repair it was meant to
cover is KEPT and is pinned by mutation-verified unit coverage instead. The
framework gap below is open.

---

## 1. The vehicle and where the seam is

`t0_52` = `cifar_vit_leaf`, `patch_size 7`, `embed_dim 24`, `num_heads 2`,
`depth 1`, `mlp_ratio 4.0`, MNIST 28px → 17 tokens (16 patches + cls), platform
A (`has_bias: true`), `weight_bits 5` (q_max 15), `simulation_steps 4`,
`spiking_family lif` / `spiking_variant streamed`, `encoding_layer_placement
offload`, `weight_quantization true`, `negative_value_shift` at its default
`true`.

Measured mapper graph at the Soft Core Mapping gate (19 nodes; `kappa` is
`read_boundary_out_scales`, `ABS` is the `value_domain_map` flag):

| # | node | ABS | kappa | armed wrap slots |
|---|------|-----|-------|------------------|
| 0 | Input | yes | 1.0 | — |
| 1 | `patch_embed` Conv2d | yes | 1.0 | — |
| 2–3 | Reshape, Permute | yes | 1.0 | — |
| 4 | `cat` (cls token) | yes | 1.0 | — |
| 5 | `add` (pos embed) | yes | 1.0 | — |
| 6 | `blocks_0_norm1` LayerNorm | yes | 1.0 | — |
| 7 | `blocks_0_attn` MultiheadAttention | yes | 1.0 | — |
| 8 | `add_1` (attention residual) | yes | 1.0 | — |
| 9 | `blocks_0_norm2` LayerNorm | yes | 1.0 | — |
| 10 | Ensure2D | yes | 1.0 | — |
| **11** | **PERCEPTRON `blocks_0_fc1` 24→96** | no | **2.5431** | θ = 2.5431, `input_activation_scale` = **1.0**, `per_input_scales` = **1.0** |
| 12 | Ensure2D | no | 2.5431 | — |
| 13 | `blocks_0_fc2` Linear | no | 2.5431 | in [2.5431] out 2.5431 |
| **14** | **`add_2` — THE MIXED SEAM** | no | 1.7716 | in **[1.0, 2.5431]** out **1.7716** |
| 15–18 | `norm`, `getitem_3`, Ensure2D, `head` | no | 1.7716 | in [1.7716] out 1.7716 |

Two facts follow immediately, and both are load-bearing:

- **The graph has exactly ONE mixed seam** (node 14) and **exactly ONE on-chip
  perceptron** (node 11), which sits *upstream* of it. 17 neural cores (one per
  token) × 96 neurons = 1632 neurons; the failure reports `3264 = 2 samples ×
  1632`, all on `perceptron 0`, because perceptron 0 is the only perceptron
  there is. "All mismatches on perceptron 0" is therefore not evidence about
  the seam either way — **the seam's value never enters a window count in this
  vehicle at all.**
- **The repair did its job.** At the gate `heterogeneous_domain_joins` returns
  0, the emitted IR op for the seam is a `ScaleNormalizingWrapper` carrying
  exactly the propagated currencies, and `verify_boundary_currency_coherence`
  passes.

## 2. What actually diverges

Failure, `soft_core_mapping_step.py:559` → `nf_scm_parity.py:258`:

```
streamed NF<->SCM exactness violated: 3061/3264 neuron-window count mismatches
over 2 samples (atol=0; worst=(4.0, 0, 0, 459, 4.0, 0.0); per-perceptron={0: 3061})
```

Window-count distributions over counts 0…4 (T = 4):

| side | mean | max | histogram |
|------|------|-----|-----------|
| NF (torch twin) | 3.3434 | 4 | `[203, 138, 339, 239, 2345]` |
| SCM (identity-mapped executor) | 0.1363 | 1 | `[2819, 445, 0, 0, 0]` |

The twin saturates; the chip is nearly silent. That is a factor of ~25 in
effective drive, not a tie.

**The entry agrees.** Executor stage `neural_segment_until:blocks_0_fc2_col0`:
`seg_input_rates` mean **0.99839** (min 0.562, max 1.0), `seg_input_spike_count`
mean **3.99265** of 4. Twin entry value (post-shift) mean 3.77886, min 0.53953,
max 7.19786; its clamped rate mean **0.99863**, with **99.51%** of entries at
exactly 1.0. Both sides drive the core with an essentially all-ones train, and
they agree on it. **The divergence is inside the core.**

**The weights agree exactly; the bias does not.** Weight-grid scale =
`threshold / θ` = `33.0 / 2.5431447` = **12.976061**.

| neuron | twin `rowsum × scale` | core `colsum` | twin `bias × scale` | core `hardware_bias` |
|--------|----------------------|---------------|---------------------|----------------------|
| 0 | −21.000 | −21.000 | 83.95 | **15** |
| 1 | −46.000 | −46.000 | 179.94 | **15** |
| 2 | −14.000 | −14.000 | 57.64 | **15** |
| 3 | −13.000 | −13.000 | 55.88 | **15** |
| 4 | −20.000 | −20.000 | 81.19 | **15** |
| 5 | −4.000 | −4.000 | 23.04 | **15** |
| 6 | −3.000 | −3.000 | 18.28 | **15** |
| 7 | −4.000 | −4.000 | 22.04 | **15** |

Every weight matches to the last integer. **87 of 96** biases exceed the
representable ±15 and saturate; the mean deleted charge is **51.70 grid units**
against `threshold = 33.0`. Per-cycle drive at the (shared) all-ones input:
twin **+49.88** mean, 71.88% of neurons at or above θ; chip **−1.79** mean,
**0.00%** at or above threshold. That reproduces the observed 3.34-vs-0.14
counts exactly.

## 3. The chain that produces an unrepresentable bias

1. `blocks_0_norm2` is a LayerNorm feeding the only core, so its calibrated
   minimum is deeply negative: **−3.7596** (scalar — a rank-≥3 token×channel
   seam takes a scalar minimum by design).
2. `SoftCoreMappingStep._apply_negative_boundary_policy` →
   `apply_negative_value_shifts` stamps `_negative_shift = +3.759628` and
   `apply_negative_shift_bias` bakes `B' = B − W·s` into the consuming
   perceptron. Verified against the artifacts: predicted `[6.46977471,
   13.86729725, 4.44162479, 4.30601907]`, actual `[6.4697752, 13.86729813,
   4.44162416, 4.30601883]`.
3. **Before** the bake the bias is exactly on the integer grid and in range
   (max **10.00** of 15). **After** it is off-grid and up to **230.34**.
4. `chip_quantize._scale_hardware_bias` clips to `±q_max·r` (r = 1: no
   two-scale bias grid) and casts. Silently.

`_scale_hardware_bias`'s own docstring states the contract it relies on — it
applies "the same saturation NAPQ applies", i.e. the torch bias was already
saturated identically during training, making the clip a no-op. **That contract
holds only while nothing writes the bias after NAPQ.** The SCM-time shift bake
does, so the twin trains and is measured against a bias the chip cannot hold.
(The bake also takes the bias off the integer grid, which breaks the atol=0
lattice by rounding even when the magnitude fits — a smaller, independent
defect of the same seam.)

## 4. Why the bias is that big: the entry gauge is never measured

`perceptron.input_activation_scale` = **1.0**, because the entire ten-node host
prefix stays ABSOLUTE at unit gauge:
`ComputeOpMapper.propagate_boundary_scale` falls through to
`mean_source_scale(deps, out_scales, default)` with `default = input_data_scale
= 1.0`, and `boundary_traffic_scale` — the observed-traffic lift that exists for
exactly this purpose — is `None`, because its only installer
(`signed_seam_install.install_signed_seam_offsets`) is gated on
`lif_exact_qat_active(...)` **and** the opt-in `lif_aq_negative_boundary`, and
the streamed discipline does not run the exact-QAT install.

So the wire band `[0, 1]` and the shifted host signal `[0.54, 7.20]` are in
different units, and two things follow from that one fact:

- **Accuracy:** 99.5% of entry rates clamp to 1.0. The deployed core sees a
  near-constant input. Both twins do this identically, so the *exactness* gate
  cannot see it — this failure mode is invisible to every certificate we have.
- **Representability:** the compensating bias is `W·s` with `s ≈ 3.76` in value
  units, i.e. ≈ 65 grid units mean where 15 is representable.

Had the entry gauge tracked the traffic (κ ≈ 7.2), the entry rate would span
`[0, 1]` instead of saturating, `per_input_scales` would fold 7.2 into the
effective weights, the weight-grid scale would drop by the same factor
(12.976 → ≈1.80), and the bias would land at ≈9 grid units — inside ±15. **One
missing quantity causes both symptoms, and one change fixes both.**

## 5. Hypotheses ruled out

- **κ_T ≠ κ_S / a second κ writer** (calculus §11.2; the leading hypothesis
  going in). REFUTED by measurement. Arming is STICKY:
  `apply_compute_op_scale_policy`'s already-armed branch preserves
  `output_scale` and only refreshes per-source scales from the walk, so the
  later `compute_per_source_scales` calls at `weight_quantization_step.py:50`
  and `soft_core_mapping_step.py:228` re-derive the SAME κ that LIF Adaptation
  armed. At the gate: 0 unclassified joins, the seam wrapper carrying
  `(1.0, 2.5431) → 1.7716`, coherence certificate passing. One writer holds.
- **The seam itself.** The only on-chip perceptron is upstream of it; the seam's
  value never reaches a window count in this vehicle.
- **Entry-boundary encode drift between twin and executor.** Measured equal
  (rate means 0.99863 vs 0.99839, spike counts 3.99/4).
- **Weight quantization drift.** Measured exactly equal, per neuron.
- **Depth-balancing relays / latency +1.** 17 cores, every `latency = 0`, no
  relays inserted.

## 6. Two guards that are missing

- **G1 — the negative-boundary policy proves only half its post-condition.**
  `apply_negative_boundary_policy` re-checks that no boundary is left negative,
  and `warn_once_lossy_negative_clamp` watches the same (lower) side. Nothing
  checks the UPPER side: that `shifted_value / κ` still fits `[0, 1]`. Here it
  fits it 0.5% of the time.
- **G2 — `_scale_hardware_bias` saturates silently.** Nothing verifies that the
  torch twin's bias is representable on the deployed grid, so a silently
  corrupted chip is authorable. Making it fail loud restores the fail-loud law.

Both are one-line-ish guards with **tier-wide blast radius**: any currently
"green" cell that saturates today would turn red the moment either lands. They
must be landed together with a full tier-0 re-validation, which is why this unit
documents them rather than shipping them.

## 7. Narrowest next step

Give an ABSOLUTE host prefix a measured entry gauge before the negative-shift
bake — i.e. make the boundary κ cover the shifted range, which is precisely the
`boundary_traffic_scale` quantile mechanism `install_signed_seam_offsets`
already computes but only under `lif_exact_qat`. Concretely, in order:

1. Reproduce standalone (cheap): the cached run in
   `generated/t0_52_lifs_vitleaf_wq_s4_offload_phased_deployment_run` resumes at
   `start_step = "Soft Core Mapping"` and re-fails in **50.9 s**; everything in
   §2–§4 was measured that way.
2. Land G1 + G2 as loud refusals and re-validate tier-0 — this converts every
   instance of this class from silent corruption to a named failure, and tells
   you how many cells are already affected.
3. Extend the traffic-gauge lift to the streamed discipline (mode-independent),
   then re-add the leaf-ViT row and measure.

## 8. If the row is re-added

- **Vehicle sizing (measured), so it need not be re-derived:** 28px / patch 7 →
  17 tokens, `embed_dim 24`, `depth 1` gives 19 mapper nodes, 1 perceptron
  (24→96), 1 mixed seam — where `mmix` and `stream_cnn` produce ZERO mixed seams
  at any placement. `mlp_ratio` must be the canonical **4.0**: the block MLP's
  first Linear is the only on-chip tensor (attention, patch-embed and readout
  are all host), so at ratio 1.0 the offload majority gate refuses the mapping
  at 10.66% of 5626 params, while 4.0 puts it at 26.22% of 9154 (floor 20%).
- **The wall budget was also wrong.** The row was budgeted 10 min (900 s at the
  default `--budget-scale 1.5`). Measured on the failing run: **868.3 s to
  reach** Soft Core Mapping (of which Weight Quantization alone is **505.7 s**,
  58% of the run; Pretraining 113.6 s, LIF Adaptation 87.4 s), leaving 31.7 s
  for Soft Core Mapping (**50.9 s** measured alone) plus Core Quantization
  Verification, Hard Core Mapping, Simulation, Loihi, SANA-FE and Deployment
  Record. The row would have timed out with **no verdict** even had it been
  exact. A re-added leaf-ViT row needs its own measurement on an unloaded
  machine (this one carried a load average of ~165 from sibling agents) and a
  budget of at least 20 min.

## 9. What was kept

The seam repair is real, proven, and unchanged: `establish_wire_gauge` +
`establish_gauge_for_mixed_domain_seams`, scoped so a graph that already
classifies is left byte-identical. Its coverage is now unit-level and
mutation-verified rather than tier-level:

- `tests/unit/spiking/test_wire_gauge_establishment.py` — the unarmed seam still
  fails loud; establishment arms it at the two producers' true gauges; the armed
  seam value recovers the ABSOLUTE sum; the coherence certificate passes; the
  repair is scoped; and the unity fallback is reachable ONLY when every producer
  gauge is already 1.0 (pinned by refusing it on a non-unit graph).
- `tests/unit/pipelining/test_streamed_mixed_seam_exactness.py` — NF↔SCM window
  counts at atol=0 across the seam **with a consuming neural core downstream of
  it**, so the seam value reaches the comparison, plus three SSOT breaks that
  must each fire the gate: the twin drops the emitted wrapper, IR emission drops
  it, and the deployed wrapper decodes every source at unity instead of at its
  producer's gauge (the §11.2 physics claim, measured: 56% of windows move).

One negative result worth recording: `compute_op_owns_scale_domain` is NOT a
lever on the streamed-lif path and cannot be covered by a streamed cell.
`resolve_stage_compute_scales` is called with `apply_ttfs=False` on every
rate/LIF call site and takes its `(1, 1)` early return before consulting the
predicate; only the TTFS-family runners reach it. Forcing it to `False` in
source leaves all streamed counts bit-identical (verified). The deployed seam's
actual SSOT is the wrapper baked into `op.params["module"]` at IR emission plus
`boundary_normalization_scales`, and those are what the three breaks above
exercise.
