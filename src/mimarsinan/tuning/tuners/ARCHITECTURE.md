# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `../unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `_RECOVERY_PATIENCE` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr`). `SmoothAdaptationTuner`: baseline calibration from `validate_n_batches` at rate 0.0, one-shot with test gate, rollback tolerance `pipeline_dt + 3*se`, all `min_improvement` from `accuracy_se()` |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Extends `SmoothAdaptationTuner`; uses `PerceptronTransformTrainer`; stochastic mixing of previous/new perceptron transforms with a **per-cycle frozen mask** (a fresh Bernoulli mask at probability `rate` is drawn lazily per `(perceptron_id, param_name)` inside each `_mixed_transform(rate)` closure, then reused across every training step and validation within that probe -- see "Per-cycle frozen mask" below); delegates `_adaptation()` to base class (test gate, min_improvement, hooks); `_after_run()` forces rate=1.0 transform, recovery training, `_ensure_pipeline_threshold()` |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU; `_after_run()` commits to LeakyGradReLU and caches metric via `trainer.test()`; includes commit guard: if post-commit accuracy falls below `target_adjuster.floor`, restores pre-commit state; `validate()` returns cached metric |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively; validates `activation_scales`, logs diagnostics, probes saturation; caches final `trainer.test()` metric |
| `activation_shift_tuner.py` | `ActivationShiftTuner` | Extends `TunerBase` (not smooth adaptation); applies shift once, recovers with LR-search + step-training using `min_improvement=accuracy_se()` and `eval_n_batches`; caches final `trainer.test()` metric |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness; extends `PerceptronTransformTuner` |
| `lif_adaptation_tuner.py` | `LIFAdaptationTuner` | Knowledge-distillation recovery after swapping Perceptron `base_activation` to `LIFActivation`; teacher = frozen pre-LIF snapshot, student = LIF-ified model; loss = α·CE + (1−α)·T²·KL (T=3, α=0.3, matching `spikingjelly-example/train.py`) |
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
LIFAdaptationTuner (standalone; knowledge-distillation recovery after LIF swap)
```

## Dependencies

- **Internal**: `tuning` (adaptation framework), `model_training` (trainers), `data_handling` (`DataLoaderFactory`), `models` (layers, unified_core_flow), `mapping.ir` (`NeuralCore`), `transformations`.
- **External**: `torch`, `numpy`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports specific tuners for each tuning step.

## Exported API (\_\_init\_\_.py)

`TunerBase`, `SmoothAdaptationTuner`, `ClampTuner`, `ActivationAdaptationTuner`,
`ActivationQuantizationTuner`, `ActivationShiftTuner`,
`NormalizationAwarePerceptronQuantizationTuner`, `LIFAdaptationTuner`, `NoiseTuner`,
`PerceptronTransformTuner`, `PruningTuner`.

## Per-cycle frozen mask (PerceptronTransformTuner)

`PerceptronTransformTuner._mixed_transform(rate)` now returns a closure
that captures a private `mask_cache` dictionary. The first time the
closure is invoked for a given perceptron slot the Bernoulli mask at
probability `rate` is drawn for each parameter (keyed by
`(id(perceptron), param_name)`) and stored in the cache. Subsequent
invocations of the same closure reuse the cached mask.

Why this matters:

- `PerceptronTransformTrainer._perceptron_slots` keeps a persistent
  `temp_p` object per perceptron and refreshes its parameters from
  `aux_model` before every training step. Both training (per step) and
  validation (inside `_update_and_evaluate`) dispatch through the
  *same* `trainer.perceptron_transformation` callable, i.e. the *same*
  `_mixed_transform(rate)` closure. With the cache, all of those calls
  see the *same* stochastic realisation of the prev/new mix.
- Without the cache (legacy behaviour) the random mask was redrawn on
  every call. Combined with the now rate-aware
  `NormalizationAwarePerceptronQuantization`, this meant training
  chased a moving-target loss surface and validation measured a
  different realisation than training had just optimised -- cycles
  rolled back even at tiny rates and the committed rate could not
  progress past ~0.
- `_update_and_evaluate(rate)` creates a fresh closure (and therefore a
  fresh cache) per probe, so the mask is still stochastic across
  probes -- preserving the intended regularisation flavour while
  eliminating the within-cycle instability.

Endpoint behaviour is unchanged: `rate == 0.0` always produces an
all-False mask (identity), `rate == 1.0` always produces an all-True
mask (fully-transformed output).
