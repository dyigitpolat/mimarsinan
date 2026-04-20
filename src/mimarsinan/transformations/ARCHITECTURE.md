# transformations/ -- Model Transformations

Provides weight and activation transformation utilities used during quantization
pipeline steps and hardware mapping.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron_transformer.py` | `PerceptronTransformer` | Computes effective weights/biases by fusing normalization; uses `per_input_scales` for per-channel input scaling. `apply_effective_bias_transform` and `apply_effective_bias_transform_to_norm` are no-ops when `layer.bias is None` (bias-less perceptrons). |
| `weight_quantization.py` | `TensorQuantization` | Symmetric tensor quantization to N-bit integers (legacy closed-form path retained for non-QAT call sites) |
| `lsq_quantization.py` | `LSQQuantizer`, `ste_round` | LSQ (Learnable Step-Size Quantization) `nn.Module` with a log-space learnable step and a straight-through estimator for the rounding op.  Forward = clamp+STE-round; backward gives gradients to both the input weights and `log_scale`.  Phase C1 replacement for the rate-mixed FP/Q interpolation that used to happen inside `NormalizationAwarePerceptronQuantization`. |
| `normalization_aware_perceptron_quantization.py` | `NormalizationAwarePerceptronQuantization` | Weight quantization that accounts for normalization parameters.  **Phase C1**: lazily attaches an `LSQQuantizer` to the perceptron (`perceptron.weight_quantizer`), seeds its step from the effective weight max-abs the first time, then uses it to bake hard-quantized weights into `layer.weight.data` while also publishing the legacy `parameter_scale = q_max / max(|w|)` for downstream code that still reads it. |
| `chip_quantization.py` | `ChipQuantization` | Chip-level quantization utilities |
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
`transform_np_array`, `compute_pruning_masks`, `apply_pruning_masks`.
