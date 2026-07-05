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
| `activation_scale_policy.py` | Selectable per-layer ANN-to-SNN activation-scale calibration policies (`count_quantile` default, `percentile_norm`, `max_norm`) behind `make_activation_scale_policy`. |
| `chip_quantization.py` | Legacy chip-level core-matrix quantization/verification (`ChipQuantization`); the production IR path uses `mapping.export.chip_quantize` instead. |
| `normalization_aware_perceptron_quantization.py` | `NormalizationAwarePerceptronQuantization`: per-perceptron quantization of effective weights/biases; `rate` interpolates between FP (`0`) and fully quantized (`1`), and `parameter_scale` is always the full-range scale. |
| `normalization_fusion.py` | `fuse_into_perceptron`: folds `perceptron.normalization` into `perceptron.layer` (fused bias = `effective_preactivation_bias`), sets normalization to Identity, and refreshes TTFS bias references. |
| `quantization_bounds.py` | `quantization_bounds(bits)` -> `(q_min, q_max)` for signed symmetric quantization; shared by SCM mapping, chip quantize, and tuners. |
| `quantization_verify.py` | `assert_integer_scaled_matrix`: shared integer-quantization checks (returns failure messages) for perceptron and IR verification paths. |
| `transformation_utils.py` | `transform_np_array`: applies a torch-tensor transform to a numpy array (numpy/torch bridge helper). |
| `weight_clipping.py` | `SoftTensorClipping` / `clip_core_weights` / `get_clipped_w_b`: soft weight clipping (clamp to mean of top/bottom fraction) for training stability. |
| `weight_quantization.py` | `TensorQuantization`: symmetric N-bit quantization (`quantize`, `scaled_quantize`) for torch tensors and numpy arrays. |
| `parameter_transforms/` | `SequentialTransform`: composable chains of parameter transforms applied in sequence. |
| `perceptron/` | `PerceptronTransformer`: computes effective weights/biases by fusing normalization, `per_input_scales`, and (possibly per-channel) `activation_scale`; applies transforms in effective-parameter space — every effective→raw write re-commits the layer's prune masks and fails loud on non-finite results. |
| `pruning/` | Pruning suite: mask computation from weight-L1 or activation importance with cross-layer propagation and IR-derived I/O exemptions (`masks.py`, `activation.py`), rate-adaptive mask application (`apply.py`), structured magnitude channel pruning that shrinks layer shapes and core counts (`magnitude.py`, default-off deployment knob), and the committed-raw-parameter pruning contract (`committed_masks.py`: per-layer and module-tree commit/verify helpers backing the enforcement hooks, transformer writes, DFQ entry, the soft-core-mapping gate, and the pipeline-cache store/load boundary). |

## Dependencies
- `models` — `normalization_fusion.py` uses `models.nn.activations.ttfs_spiking.refresh_perceptron_bias_references` and `models.perceptron_mixer.perceptron.effective_preactivation_bias`; `perceptron/perceptron_transformer.py` lazily imports `models.nn.layers.norm_affine_params` to read normalization affine parameters.
- `mapping` — `pruning/masks.py` lazily imports `mapping.pruning.boundary_policy.compute_perceptron_io_exemption_indices` to exempt model-I/O layers when computing masks from an IR graph.

## Dependents
- `mapping` — `PerceptronTransformer` (perceptron/conv1d/conv2d mappers, bias compensation), `TensorQuantization` (`mapping_utils`), `quantization_bounds` and `assert_integer_scaled_matrix` (`export.chip_quantize`, `pruning.boundary_policy`).
- `pipelining` — `fuse_into_perceptron` (normalization-fusion step), `PerceptronTransformer` (quantization-verification step), `quantization_bounds` and `pruning.committed_masks` commit/verify (soft-core mapping step and the cache store/load boundary), `prune_perceptron_chain` (soft-core structured-pruning step).
- `spiking` — `pruning.committed_masks.commit_perceptron_pruning` (DFQ starts from the committed-pruning state).
- `tuning` — `NormalizationAwarePerceptronQuantization` (quantization tuner), `apply_pruning_masks` and `collect_activation_stats` (pruning tuner), `pruning.committed_masks` (the pruning enforcement hooks), `PerceptronTransformer` (activation-shift tuner), `quantization_bounds` (TTFS cycle-adaptation tuner).

## Exported API
`__init__.py` re-exports:
- `PerceptronTransformer` — effective weight/bias view of a perceptron.
- `TensorQuantization` — symmetric N-bit tensor quantization.
- `NormalizationAwarePerceptronQuantization` — rate-interpolated effective-weight quantization.
- `transform_np_array` — numpy/torch transform bridge.
- `compute_pruning_masks`, `apply_pruning_masks` — pruning mask computation and application.
- `prune_perceptron_chain`, `kept_output_channels`, `ChannelPruningResult` — structured magnitude channel pruning.
- `ActivationScalePolicy`, `make_activation_scale_policy`, `DEFAULT_ACTIVATION_SCALE_POLICY` — activation-scale calibration policies.
