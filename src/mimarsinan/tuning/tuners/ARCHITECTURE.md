# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_tuner.py` | `BasicTuner` | Base tuner using `WeightTransformTrainer` and `SmartSmoothAdaptation` |
| `perceptron_tuner.py` | `PerceptronTuner` | Base for per-perceptron tuners using `BasicTrainer` |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Base for tuners using `PerceptronTransformTrainer` |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness |
| `core_flow_tuner.py` | `CoreFlowTuner` | Adjusts spiking thresholds on IR graph (rate-coded mode only) |
| `noise_tuner.py` | `NoiseTuner` | Introduces training noise |
| `scale_tuner.py` | `ScaleTuner` | Adjusts activation scaling factors |

## Tuner Hierarchy

```
BasicTuner (WeightTransformTrainer)
PerceptronTuner (BasicTrainer, per-perceptron)
├── ClampTuner
├── ActivationQuantizationTuner
├── NoiseTuner
└── ScaleTuner
PerceptronTransformTuner (PerceptronTransformTrainer)
└── NormalizationAwarePerceptronQuantizationTuner
CoreFlowTuner (operates on IRGraph, not model)
```

## Dependencies

- **Internal**: `tuning` (adaptation framework), `model_training` (trainers), `data_handling` (`DataLoaderFactory`), `models` (layers, unified_core_flow), `mapping.ir` (`NeuralCore`), `transformations`.
- **External**: `torch`, `numpy`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports specific tuners for each tuning step.

## Exported API (\_\_init\_\_.py)

All tuner classes.
