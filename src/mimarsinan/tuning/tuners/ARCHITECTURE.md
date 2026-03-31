# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_tuner.py` | `BasicTuner` | Base tuner using `WeightTransformTrainer` and `SmartSmoothAdaptation`; optional `initial_tolerance_fn` from `tolerance_calibration.initial_tolerance_fn_for_pipeline_if_enabled` when `tuner_calibrate_smooth_tolerance` is true |
| `perceptron_tuner.py` | `PerceptronTuner` | Base for per-perceptron tuners using `BasicTrainer`; same optional smooth-tolerance calibration wiring |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Base for tuners using `PerceptronTransformTrainer`; same optional calibration wiring |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU via `ActivationReplacementDecorator`. `run()` iterates `model.get_perceptrons()` — which already excludes Identity (host-side) perceptrons via the mapper eligibility contract — and commits all non-ReLU-compatible bases to `LeakyGradReLU` (ReLU). Resets `activation_adaptation_rate=0`, measures accuracy **once** via `trainer.test()` on the full test set, and caches it in `_committed_metric`. Callers must read `_committed_metric` instead of calling `validate()` again to avoid advancing the validation iterator. |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness |
| `core_flow_tuner.py` | `CoreFlowTuner` | Adjusts spiking thresholds on IR graph (rate-coded mode only) |
| `noise_tuner.py` | `NoiseTuner` | Introduces training noise |
| `pruning_tuner.py` | `PruningTuner` | Gradually zeros least-significant rows/columns; recomputes importance at the start of each adaptation cycle; static `adapter.tolerance = 0.05` when calibration is off |

## Tuner Hierarchy

```
BasicTuner (WeightTransformTrainer)
PerceptronTuner (BasicTrainer, per-perceptron)
├── ActivationAdaptationTuner
├── ClampTuner
├── ActivationQuantizationTuner
├── NoiseTuner
└── PruningTuner (per-cycle importance refresh via before_cycle)
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
