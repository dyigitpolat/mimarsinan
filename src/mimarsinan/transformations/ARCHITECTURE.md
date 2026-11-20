# transformations/ — weight/activation transforms for quantization, fusion, and pruning

Pure model-transformation utilities applied between training and hardware mapping in the
deployment pipeline. The central abstraction is the "effective parameter" view provided by
`PerceptronTransformer` — weights/biases with normalization, per-input scales, and activation
scales folded in — on top of which quantization, normalization fusion, and pruning operate.
Everything here mutates or inspects torch perceptrons and numpy core matrices; the `mapping`
and `pipelining` modules consume the results when emitting IR and hard cores.

## Key files
| File | Purpose |
|---|---|
| `activation_scale_policy.py` | Selectable per-layer ANN-to-SNN activation-scale calibration policies (`count_quantile` default, `percentile_norm`, `max_norm`, `min_distortion`) behind `make_activation_scale_policy`; carries `deployed_distortion`/`optimal_theta` — theta as the argmin of the deployed staircase's two error terms (saturation above theta + the theta/L grid below) [calculus §17.13]. |
| `channel_scale_equalization.py` | Exact cross-layer per-channel scale migration (M4): finds feature-adjacent affine pairs A→ReLU→B whose channel axis is consumed only on linear feature axes (via the shared `mapping.channel_axis_walk` traversal through permute/leading-dim/mean-over-non-channel ops), then migrates `s_c = q99_c/geomean` clipped to `[1/r, r]` (`DEFAULT_CLIP_RATIO=4`, the measured WQ-grid guard) as `W_A ← S⁻¹W_A, b_A ← S⁻¹b_A` (norm-affine realization for BN-attached hops), `W_B ← W_B·S`. Function-preserving by ReLU positive homogeneity; weight-shared axes (token-mixer fc2) are honestly left alone — their exact escape is a per-channel theta, owned by scale propagation. |
| `normalization_aware_perceptron_quantization.py` | `NormalizationAwarePerceptronQuantization`: per-perceptron quantization of effective weights/biases; `rate` interpolates between FP (`0`) and fully quantized (`1`), and `parameter_scale` is always the full-range scale. `two_scale=True` derives the weight grid `s_w` from max-abs weights alone, and the bias lands where the platform delivers it, exactly integer on the weight-grid lattice the chip export emits either way: on bias rows (a param-encoded bias, bias-row splitting) on its own ±q_max range of an integer-ratio-snapped grid (`bias_scale = parameter_scale/r`, `r = ceil(max|b| s_w / q_max)`, each of the `r` rows an equal share); in an on-chip bias register (`register_bias=True`, the route the WQ step names from `resolve_wq_two_scale_projection`) on the weight lattice itself, `round(b s_w)` within the span `[-(q_max+1) r, q_max r]` the export allocates, `bias_scale == parameter_scale` (U-WQB: the coarser grid bought no register and cost up to `r/2` weight steps per neuron). That `r` IS the bias-row count a param-encoded platform maps and the register's span, so the formula body lives in the bias-row SSOT (`mapping/support/bias_rows.py`, imported inside the projection because mapping imports transformations); `bias_rows_floor` is bias-row splitting's declared k, a FLOOR under the computed bound rather than a cap, so a bias the declared k outgrows buys rows instead of saturating at the register edge. Every route re-derives a recorded input shift's compensation on the weights it installs (`perceptron/input_shift.py`) before the bias is coded. |
| `normalization_fusion.py` | `fuse_into_perceptron`: folds `perceptron.normalization` into `perceptron.layer` (fused bias = `effective_preactivation_bias`), sets normalization to Identity, and refreshes TTFS bias references. |
| `quantization_bounds.py` | `quantization_bounds(bits)` -> `(q_min, q_max)` for signed symmetric quantization; shared by SCM mapping, chip quantize, and tuners. |
| `quantization_verify.py` | `assert_integer_scaled_matrix`: shared integer-quantization checks (returns failure messages) for perceptron and IR verification paths. |
| `transformation_utils.py` | `transform_np_array`: applies a torch-tensor transform to a numpy array (numpy/torch bridge helper). |
| `weight_quantization.py` | `TensorQuantization`: symmetric N-bit quantization (`quantize`, `scaled_quantize`) for torch tensors and numpy arrays. |
| `parameter_transforms/` | `SequentialTransform`: composable chains of parameter transforms applied in sequence. |
| `perceptron/` | `PerceptronTransformer` plus the bias canonicalization suite (`bias_saturation.py`: provable constant-OFF saturation clip + empirical shift math feeding the saturation-aware NAPQ scale; `bias_canonicalization.py`: the guarded, decision-VERIFIED QAT-entry pass — empirical saturation shifts + nuisance-channel removal, bit-exact restore on any calibration decision flip) and the input-shift record (`input_shift.py`, U-WQB: the shift a negative-boundary bake `B - W s` compensated, a non-persistent `input_shift` buffer beside the bake's `_neg_shift_baked`, written by `mapping.support.negative_shift.apply_negative_shift_bias` for every caller; `compensate_projected_weights` moves the bias by `-(W_after - W_before) s` at every weight projection, so the bias compensates the weights the chip holds and not the float ones; Weight Quantization refuses by name a bake without its record, `UnrecordedShiftBakeError`). `PerceptronTransformer`: computes effective weights/biases by fusing normalization, `per_input_scales`, and (possibly per-channel) `activation_scale`; applies transforms in effective-parameter space — every effective→raw write re-commits the layer's prune masks and fails loud on non-finite results. |
| `pruning/` | Pruning suite: mask computation from weight-L1 or activation importance with cross-layer propagation and IR-derived I/O exemptions (`masks.py`, `activation.py` — the importance probe DECIDES which channels are deleted permanently and a NaN importance vector still sorts into a finite boolean mask, so its reads go through `data_handling.batch_integrity` like every other decided number), rate-adaptive mask application (`apply.py`), structured magnitude channel pruning that shrinks layer shapes and core counts (`magnitude.py`, default-off deployment knob; [P] `prune_perceptron_chain_by_counts` is the counts-driven variant the candidate pruning twin drives with the mask floor rule, and `masks.mask_prune_count` is THE `floor(rate*f*dim)` formula both the committed masks and that twin read), the committed-raw-parameter pruning contract (`committed_masks.py`: per-layer and module-tree commit/verify helpers backing the enforcement hooks, transformer writes, DFQ entry, the soft-core-mapping gate, and the pipeline-cache store/load boundary), and the criterion-agnostic seed generators (`seed_generators.py`: the `row_col_l1`/`activation`/`partial_column_group` registry behind `prune_criterion` — each criterion emits `LayerSeedMasks` in the ONE committed-mask buffer format (`install_seed_masks`, also the SSOT behind the pruning tuner's buffer registration) that seeds the IR pruning cascade; `partial_column_group` kills Meng-style contiguous row-groups per column, whose completed rows/cols the cascade harvests). |

## Declared grids (Wave 2 WI-5b, WI-5c)

`NormalizationAwarePerceptronQuantization(fractional_scale=, float_bias=)`: the scale `q_max / max` instead of its
integer floor (`_grid_scale`), and a float bias (the weight grid from `max|w|` alone, the bias as trained, no
saturation clip since no bias grid exists); both default off, the projection then byte for byte the base's.

## Per-layer weight widths (Wave 2 WI-5g)

`quantization_bounds.py` names the width a layer or a core declares (`LAYER_WEIGHT_BITS` on a perceptron,
`CORE_WEIGHT_BITS` on an IR core, bank, soft or hard core) and reads it (`layer_weight_bits`, `core_weight_bits`, the
platform width when absent); `NormalizationAwarePerceptronQuantization.transform` projects each perceptron on its own
width's grid, and the bias canonicalization predicts its weight zeros at that width.

## Dependencies
- `models` — `normalization_fusion.py` uses `models.nn.activations.ttfs_spiking.refresh_perceptron_bias_references` and `models.perceptron_mixer.perceptron.effective_preactivation_bias`; `perceptron/perceptron_transformer.py` lazily imports `models.nn.layers.norm_affine_params` to read normalization affine parameters.
- `mapping` — `pruning/masks.py` lazily imports `mapping.pruning.boundary_policy.compute_perceptron_io_exemption_indices` to exempt model-I/O layers when computing masks from an IR graph; `channel_scale_equalization.py` imports `channel_axis_walk.channel_aligned_consumer_targets` (the shared mapper-DAG traversal) and `PerceptronMapper` to find exactly-migratable pairs.

## Constant-OFF bias clip on the proven input range (D19)

`perceptron/bias_saturation.py`: `clip_off_saturated_effective_bias` clips a channel to `off_saturation_bias_bound`,
`-sum_i max(w_i lo_i, w_i hi_i)` over the `InputRange` that `constant_off_input_range` proves, and only there: a node
that declares a constant-OFF floor (`models.nn.off_floor.has_off_floor` on `base_activation`: ReLU, LIF and TTFS
nodes and their blends; an affine package's output is linear in its bias, so it is never clipped), not an encoding
layer, and a proven range from `declared_input_range`: the value range of the boundary grid the input wire ends on
(`models.nn.activations.value_quantizer.wire_output_grid`), hulled with 0 for zero padding and divided by
`per_input_scales` (`PerceptronTransformer.input_scale`); the event wire's `UNIT_INPUT_RANGE` [0, 1] (bitwise the
legacy formula) when the wire holds no value-domain grid and the perceptron is not stamped a value-domain reader
(`models.perceptron_mixer.perceptron.reads_value_domain`); none (no clip) for a value-domain reader with no armed grid
at its wire's end. On a declared range the bound is taken in float64 and lowered by the float32 error bound of the
forward's sum, so a clipped channel stays OFF in the layer's own forward. `perceptron/bias_canonicalization.py` runs
its guarded empirical shift only on a node with the floor (its OFF test `v_max <= 0` is a floor's state); the
verified channel removal is unchanged.

## Dependents
- `mapping` — `PerceptronTransformer` (perceptron/conv1d/conv2d mappers, bias compensation), `TensorQuantization` (`mapping_utils`), `quantization_bounds` and `assert_integer_scaled_matrix` (`export.chip_quantize`, `pruning.boundary_policy`).
- `pipelining` — `fuse_into_perceptron` (normalization-fusion step), `PerceptronTransformer` (quantization-verification step), `quantization_bounds` and `pruning.committed_masks` commit/verify (soft-core mapping step and the cache store/load boundary), `prune_perceptron_chain` (soft-core structured-pruning step), `equalize_channel_scales` + `DEFAULT_CLIP_RATIO` (the config-gated Scale Migration step and `DeploymentPlan`/registry defaults).
- `spiking` — `pruning.committed_masks.commit_perceptron_pruning` (DFQ starts from the committed-pruning state).
- `tuning` — `NormalizationAwarePerceptronQuantization` (quantization tuner), `apply_pruning_masks` and `collect_activation_stats` (pruning tuner), `pruning.committed_masks` (the pruning enforcement hooks), `pruning.seed_generators.install_seed_masks`/`structured_seed_masks_from_keep` (the tuner's prune-buffer registration delegates to the shared seed-mask SSOT), `PerceptronTransformer` (activation-shift tuner), `quantization_bounds` (TTFS cycle-adaptation tuner).

## Exported API
`__init__.py` re-exports:
- `PerceptronTransformer` — effective weight/bias view of a perceptron.
- `TensorQuantization` — symmetric N-bit tensor quantization.
- `NormalizationAwarePerceptronQuantization` — rate-interpolated effective-weight quantization.
- `transform_np_array` — numpy/torch transform bridge.
- `compute_pruning_masks`, `apply_pruning_masks` — pruning mask computation and application.
- `prune_perceptron_chain`, `kept_output_channels`, `ChannelPruningResult` — structured magnitude channel pruning.
- `generate_seed_masks`, `install_seed_masks`, `structured_seed_masks_from_keep`, `LayerSeedMasks`, `SeedContext`, `PRUNE_CRITERIA`, `DEFAULT_PRUNE_CRITERION` — criterion-agnostic seed generators for the IR pruning cascade.
- `ActivationScalePolicy`, `make_activation_scale_policy`, `DEFAULT_ACTIVATION_SCALE_POLICY` — activation-scale calibration policies.

## The Fold per Core Position

[U-PFOLD] `perceptron/position_fold.py` is the one reader of a perceptron's fold in its two forms: per input
(`(in_features,)`, today's form and bytes) or, where one fold per input cannot hold every element at its producer's
gauge, one row per core position (`(positions, in_features)`: a linear's leading positions, a convolution's output
positions, `fold_slot_source` naming each slot's element). `PerceptronTransformer()` is the per-input reader and
refuses a per-position perceptron by name (`PositionFoldError`); `PerceptronTransformer(positions=True)` is the reader
in both forms (each weight row's effective weight, each position's effective bias, a projection per weight row), the
form Weight Quantization's projection, Quantization Verification and the row-drive reader take. The module also holds
the walks' per-element fold (`input_fold_view`), a shift's slots (`shift_rows`) and the refusal kept where positions
would share one core (`assert_one_fold_per_core`). A projection holds each weight row's projected raw weight
(`position_weight`: a linear one per position, a convolution one per distinct patch fold) beside the raw weight, the
latent every re-projection starts from (`apply_weight_projection`; the transform trainer carries the rows,
`copy_position_state`; Normalization Fusion scales them, `fuse_output_scale`). `perceptron/position_bias.py` writes a
bias corrected from the effective weight per position (the shift bake, its compensation, the affine fold: the shared
bias takes row 0's, `position_bias` each position's offset, none where all are equal) and runs a bias transform on
every position's total; `perceptron/position_forward.py` is the network function over the projected rows and the
offsets (forward the projected row, backward the latent, the transform trainer's surrogate). The bias saturation
bound is the minimum over positions, the canonicalization's scans the maximum.

[U-PFOLD r2] The reader declares, for every view it hands out, which axis holds the output neuron (`OutputAxes`:
`PER_INPUT`, a weight `(out, in, ...)` and a bias `(out,)`; `PER_POSITION`, weight rows `(rows, out, in)` and bias
rows `(positions, out)`; a perceptron's views by `output_axes(perceptron, positions)`), and a bias transform of the
reader in both forms receives the bias view it declares. Every reader that works per output neuron (the bounds and
scans, the bias writers, the network function, the mappers' banks, the twins, the advisories) takes the axis from it
(`along_outputs`, `output_rows`, `outputs_first`, `position_bias.weight_charges`), never from a literal index.

[U-PFOLD r3] Outside the one reader, a reader of the rows of a fold per position or of their count takes them by
`position_folds(perceptron)` (each core position's fold row, `rows[index]` of `position_rows`, on the weight's device):
the event-serial twin's position count, the mappers' bank key and assertion, the saturation bound's input divisor. Every
other site that names the stamp is the one writer, the per-input reader, or a reviewed presence, save or restore site
or a read of values that are the same in either form, each listed with its read in the closure test
(`tests/unit/transformations/test_fold_stamp_closure.py`, a scan of the source's syntax trees).

[U-RSHIFT] `perceptron/input_shift.py` holds the record of a received shift (`input_shift`: one value, one per input, or
one per element of the input; `_neg_shift_baked` marks only that a record exists) and the refusal of a bake with no
record; the correction and its compensation at a projection are `mapping/support/received_shift.py`'s (Weight
Quantization calls `compensate_projected_weights`). `perceptron/position_bias.py`: `held_per_position` is the stamp
writer's rule for a perceptron whose positions hold their own bias; the affine fold's per-position form is the one
writer's.
