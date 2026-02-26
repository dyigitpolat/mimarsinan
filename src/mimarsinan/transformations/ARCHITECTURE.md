# transformations/ -- Model Transformations

Provides weight and activation transformation utilities used during quantization
pipeline steps and hardware mapping.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `perceptron_transformer.py` | `PerceptronTransformer` | Computes effective weights/biases by fusing normalization; applies bias transforms |
| `weight_quantization.py` | `TensorQuantization` | Symmetric tensor quantization to N-bit integers |
| `normalization_aware_perceptron_quantization.py` | `NormalizationAwarePerceptronQuantization` | Weight quantization that accounts for normalization parameters |
| `chip_quantization.py` | `ChipQuantization` | Chip-level quantization utilities |
| `weight_clipping.py` | `SoftTensorClipping`, `clip_core_weights` | Soft weight clipping for training stability |
| `transformation_utils.py` | `transform_np_array` | Low-level numpy array quantization helper |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `parameter_transforms/` | Composable parameter transform chains |

## Dependencies

- **Internal**: None (leaf module for core transform logic).
- **External**: `torch`, `numpy`.

## Dependents

- `mapping.mapping_utils` imports `PerceptronTransformer` and `TensorQuantization`
- `tuning.tuners` imports `NormalizationAwarePerceptronQuantization`
- `pipelining.pipeline_steps` imports `PerceptronTransformer` for fusion/verification
- `model_training` imports `PerceptronTransformer` for training utilities

## Exported API (\_\_init\_\_.py)

`PerceptronTransformer`, `TensorQuantization`, `NormalizationAwarePerceptronQuantization`,
`transform_np_array`.
