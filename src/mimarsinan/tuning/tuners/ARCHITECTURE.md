# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `../unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder). `SmoothAdaptationTuner`: the single orchestration loop; subclasses implement `_update_and_evaluate(rate)` |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Extends `SmoothAdaptationTuner`; uses `PerceptronTransformTrainer`; stochastic mixing of previous/new perceptron transforms |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU; `_after_run()` commits to LeakyGradReLU and caches metric via `trainer.test()`; includes commit guard: if post-commit accuracy falls below `target_adjuster.floor`, restores pre-commit state; `validate()` returns cached metric |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively; validates `activation_scales`, logs diagnostics, probes saturation; caches final `trainer.test()` metric |
| `activation_shift_tuner.py` | `ActivationShiftTuner` | Extends `TunerBase` (not smooth adaptation); applies shift once, recovers with LR-search + step-training; caches final `trainer.test()` metric |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness; extends `PerceptronTransformTuner` |
| `core_flow_tuner.py` | `CoreFlowTuner` | Adjusts spiking thresholds on IR graph (standalone, not in tuner hierarchy) |
| `noise_tuner.py` | `NoiseTuner` | Introduces training noise |
| `pruning_tuner.py` | `PruningTuner` | Gradually zeros least-significant rows/columns; recomputes importance at each cycle; overrides `_adaptation` and `_before_cycle` |

## Tuner Hierarchy

```
TunerBase
├── SmoothAdaptationTuner
│   ├── ActivationAdaptationTuner
│   ├── ClampTuner
│   ├── ActivationQuantizationTuner
│   ├── NoiseTuner
│   ├── PruningTuner (overrides _adaptation, _before_cycle)
│   └── PerceptronTransformTuner (PerceptronTransformTrainer)
│       └── NormalizationAwarePerceptronQuantizationTuner
└── ActivationShiftTuner (one-shot, not smooth adaptation)
CoreFlowTuner (standalone, operates on IRGraph)
```

## Dependencies

- **Internal**: `tuning` (adaptation framework), `model_training` (trainers), `data_handling` (`DataLoaderFactory`), `models` (layers, unified_core_flow), `mapping.ir` (`NeuralCore`), `transformations`.
- **External**: `torch`, `numpy`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports specific tuners for each tuning step.

## Exported API (\_\_init\_\_.py)

`TunerBase`, `SmoothAdaptationTuner`, `ClampTuner`, `ActivationAdaptationTuner`,
`ActivationQuantizationTuner`, `ActivationShiftTuner`,
`NormalizationAwarePerceptronQuantizationTuner`, `CoreFlowTuner`, `NoiseTuner`,
`PerceptronTransformTuner`, `PruningTuner`.
