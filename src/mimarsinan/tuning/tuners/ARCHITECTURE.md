# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `../unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `_RECOVERY_PATIENCE` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr`); wires the trainer's Phase-D2 two-tier validation (`validate_fast` / `validate_full`) from the budget's `progress_eval_batches` / `eval_n_batches`.  `SmoothAdaptationTuner`: baseline calibration from `trainer.validate_full()` at rate 0.0, one-shot with **validation** gate (never `test()`), rollback tolerance `pipeline_dt + 3*se`, all `min_improvement` from `accuracy_se()`; per-cycle `_update_and_evaluate` uses `validate_fast`, pre-/post-cycle + safety-net use `validate_full`. |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Extends `SmoothAdaptationTuner`; uses `PerceptronTransformTrainer`. **Phase C1**: no more random-mask stochastic mixing -- LSQ + STE makes the quantizer differentiable, so `_mixed_perceptron_transform` deterministically applies the new transform when `rate > 0` and is a no-op otherwise.  `_adaptation()` delegated to the base class (validation gate, min_improvement, hooks); `_after_run()` forces rate=1.0 transform, recovery training, calls `_ensure_validation_threshold()` and `_flush_enforcement_hooks()`; only sets `_committed_rate = 1.0` when validation-based recovery reaches the floor. |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU; `_after_run()` commits to LeakyGradReLU and caches `_committed_metric` from `validate_full` (not `test()`); includes commit guard: if post-commit accuracy falls below `target_adjuster.floor`, restores pre-commit state; `validate()` returns cached metric or `validate_full` |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively; validates `activation_scales`, logs diagnostics, probes saturation; caches final validation metric (never `test()`).  **Learnable ceiling (Phase B2)**: flips `Perceptron.log_clamp_ceiling.requires_grad` on for the duration of `run()` so the optimizer (built from `self.model.parameters()`) can refine each perceptron's clamp ceiling in log-space; `_after_run` consolidates the learnt ceiling back into `activation_scale` via `set_activation_scale(exp(log_clamp_ceiling))`, rebuilds decorators, then disables `requires_grad` again so downstream steps see a frozen scalar just like before. |
| `activation_shift_tuner.py` | `ActivationShiftTuner` | Extends `TunerBase` (not smooth adaptation); applies shift once, recovers with LR-search + step-training using `min_improvement=accuracy_se()`; caches final validation metric via `validate_full` (Phase D2) |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels using the hard `QuantizeDecorator` (`StaircaseFunction`, STE-style backward).  **Per-layer binary-discrete rollout**: at tuner start `_measure_layer_sensitivities` probes every perceptron once (install the hard quantiser alone on that perceptron, record the `validate_fast` drop, restore) and caches an ascending-sensitivity order (least-sensitive first).  Each cycle's `rate` is mapped to `k = round(rate * N)` and the first `k` perceptrons in the cached order receive the hard quantiser via `AdaptationManager.set_per_perceptron_rate("quantization_rate", name, 1.0)`; the rest stay un-quantised.  The scalar `AdaptationManager.quantization_rate` is left at 0.0 throughout -- per-perceptron overrides carry the full schedule.  `_get_extra_state` / `_set_extra_state` snapshot the per-perceptron override dict so cycle rollback returns to the previous `k`-perceptron state.  `_after_run` forces `k = N` before calling `_attempt_recovery_if_below_floor` exactly once.  The annealed DSQ path (soft quantiser with β schedule) was removed because its low-β output collapsed to ~0.5 regardless of input, destroying all signal and producing catastrophic cycles at every rate. |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness; extends `PerceptronTransformTuner`.  **Phase C1**: eagerly installs an `LSQQuantizer` on every perceptron of the model (seeded from the max-abs of the effective weight) *before* `PerceptronTransformTrainer` deepcopies the model, so the aux/main pair both carry the quantizer and the optimiser updates to `log_scale` survive the per-step refresh. |
| `core_flow_tuner.py` | `CoreFlowTuner` | Adjusts spiking thresholds on IR graph (standalone, not in tuner hierarchy) |
| `pruning_tuner.py` | `PruningTuner` | Gradually zeros least-significant rows/columns; recomputes importance at each cycle; overrides `_before_cycle`, `_recovery_training_hooks`, `_after_run`, `_update_and_evaluate`; uses base-class `_find_lr` (anchored LR search); `_force_to_full_rate` drives pruning from committed rate to 1.0 in gradual increments with `min_improvement=accuracy_se()/2`; uses base-class `_adaptation` with LR search |

## Tuner Hierarchy

```
TunerBase
├── SmoothAdaptationTuner
│   ├── ActivationAdaptationTuner
│   ├── ClampTuner
│   ├── ActivationQuantizationTuner
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
`NormalizationAwarePerceptronQuantizationTuner`, `CoreFlowTuner`,
`PerceptronTransformTuner`, `PruningTuner`.
