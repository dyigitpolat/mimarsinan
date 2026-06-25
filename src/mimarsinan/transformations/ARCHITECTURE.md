# transformations/ -- Model Transformations

Provides weight and activation transformation utilities used during quantization
pipeline steps and hardware mapping.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron_transformer.py` | `PerceptronTransformer` | Computes effective weights/biases by fusing normalization; uses `per_input_scales` for per-channel input scaling. `apply_effective_bias_transform` and `apply_effective_bias_transform_to_norm` are no-ops when `layer.bias is None` (bias-less perceptrons). `_get_u_beta_mean` delegates to `models.nn.layers.norm_affine_params` (detached). |
| `weight_quantization.py` | `TensorQuantization` | Symmetric tensor quantization to N-bit integers |
| `normalization_aware_perceptron_quantization.py` | `NormalizationAwarePerceptronQuantization` | Weight quantization that accounts for normalization parameters. Accepts a `rate` argument and linearly interpolates in weight-value space between the FP and fully-quantized effective weights (`rate == 0` is identity, `rate == 1` matches the legacy full-quantization output). `parameter_scale` is always set to the full-range scale so downstream IR mapping is unaffected by rate |
| `chip_quantization.py` | `ChipQuantization` | Legacy chip-level quantization utilities (mostly tests); production IR path uses `mapping.chip_quantize`. |
| `quantization_bounds.py` | `quantization_bounds` | `(q_min, q_max)` from `weight_bits`; shared by SCM and `chip_quantize`. |
| `activation_scale_policy.py` | `ActivationScalePolicy`, `CountQuantilePolicy`, `PercentileNormPolicy`, `MaxNormPolicy`, `make_activation_scale_policy` | Selectable per-layer activation-scale (ANN->SNN) calibration policies. DEFAULT `count_quantile` is byte-identical to the legacy `scale_from_activations` (count quantile over positive activations). `percentile_norm` = Rueckauer et al. (2017) robust-norm (p-th percentile of the FULL distribution; `p=100` == max-norm), a default-OFF baseline for head-to-head conversion comparison. |
| `quantization_verify.py` | `assert_integer_scaled_matrix` | Shared integer-quantization checks for IR and perceptron verification. |
| `normalization_fusion.py` | `fuse_into_perceptron` | Folds `perceptron.normalization` into `perceptron.layer` (fused bias = `effective_preactivation_bias`, the same SSOT the TTFS segment policy charges — fusion is therefore behavior-preserving under the cascade forward); used by `NormalizationFusionStep`. |
| `weight_clipping.py` | `SoftTensorClipping`, `clip_core_weights` | Soft weight clipping for training stability |
| `transformation_utils.py` | `transform_np_array` | Low-level numpy array quantization helper |
| `pruning.py` | `compute_masks_from_importance`, `compute_all_pruning_masks`, `compute_pruning_masks`, `apply_pruning_masks`, `_collect_activation_stats` | Unified mask computation: `compute_masks_from_importance` (importance + exempt layers + cross-layer propagation) used by `PruningTuner._get_masks` and by `compute_all_pruning_masks`. Activation-based or weight-L1 importance; 1D masks stored as `prune_row_mask`/`prune_col_mask` on layers for lossless IR extraction. |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `parameter_transforms/` | Composable parameter transform chains |

## Dependencies

- **Internal**: None (leaf module for core transform logic).
- **External**: `torch`, `numpy`.

## Dependents

- `mapping.mapping_utils` imports `PerceptronTransformer` and `TensorQuantization`
- `tuning.tuners` imports `NormalizationAwarePerceptronQuantization` and pruning (`compute_masks_from_importance`, `apply_pruning_masks`, `_collect_activation_stats`)
- `pipelining.pipeline_steps` imports `PerceptronTransformer` for fusion/verification
- `model_training` imports `PerceptronTransformer` for training utilities

## Exported API (\_\_init\_\_.py)

`PerceptronTransformer`, `TensorQuantization`, `NormalizationAwarePerceptronQuantization`,
`transform_np_array`, `compute_pruning_masks`, `apply_pruning_masks`,
`ActivationScalePolicy`, `make_activation_scale_policy`, `DEFAULT_ACTIVATION_SCALE_POLICY`.
