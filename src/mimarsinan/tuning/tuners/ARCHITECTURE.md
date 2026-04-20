# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `../unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `_RECOVERY_PATIENCE` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr`).  `SmoothAdaptationTuner`: baseline calibration from `validate_n_batches` at rate 0.0, one-shot with **validation** gate (never `test()`), rollback tolerance `pipeline_dt + 3*se`, all `min_improvement` from `accuracy_se()` |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Extends `SmoothAdaptationTuner`; uses `PerceptronTransformTrainer`; stochastic mixing of previous/new perceptron transforms; delegates `_adaptation()` to base class (validation gate, min_improvement, hooks); `_after_run()` forces rate=1.0 transform, recovery training, calls `_ensure_validation_threshold()` and `_flush_enforcement_hooks()`; only sets `_committed_rate = 1.0` when validation-based recovery reaches the floor |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU; `_after_run()` commits to LeakyGradReLU and caches `_committed_metric` from `validate_n_batches` (not `test()`); includes commit guard: if post-commit accuracy falls below `target_adjuster.floor`, restores pre-commit state; `validate()` returns cached metric |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively; validates `activation_scales`, logs diagnostics, probes saturation; caches final validation metric (never `test()`).  **Learnable ceiling (Phase B2)**: flips `Perceptron.log_clamp_ceiling.requires_grad` on for the duration of `run()` so the optimizer (built from `self.model.parameters()`) can refine each perceptron's clamp ceiling in log-space; `_after_run` consolidates the learnt ceiling back into `activation_scale` via `set_activation_scale(exp(log_clamp_ceiling))`, rebuilds decorators, then disables `requires_grad` again so downstream steps see a frozen scalar just like before. |
| `activation_shift_tuner.py` | `ActivationShiftTuner` | Extends `TunerBase` (not smooth adaptation); applies shift once, recovers with LR-search + step-training using `min_improvement=accuracy_se()` and `eval_n_batches`; caches final validation metric |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness; extends `PerceptronTransformTuner` |
| `core_flow_tuner.py` | `CoreFlowTuner` | Adjusts spiking thresholds on IR graph (standalone, not in tuner hierarchy) |
| `noise_tuner.py` | `NoiseTuner` | Introduces training noise |
| `pruning_tuner.py` | `PruningTuner` | Gradually zeros least-significant rows/columns; recomputes importance at each cycle; overrides `_before_cycle`, `_recovery_training_hooks`, `_after_run`, `_update_and_evaluate`; uses base-class `_find_lr` (anchored LR search); `_force_to_full_rate` drives pruning from committed rate to 1.0 in gradual increments with `min_improvement=accuracy_se()/2`; uses base-class `_adaptation` with LR search |

## Tuner Hierarchy

```
TunerBase
├── SmoothAdaptationTuner
│   ├── ActivationAdaptationTuner
│   ├── ClampTuner
│   ├── ActivationQuantizationTuner
│   ├── NoiseTuner
│   ├── PruningTuner (overrides _before_cycle, _recovery_training_hooks, _after_run)
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
