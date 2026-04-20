# tuning/ -- Training-Aware Tuning Subsystem

Manages the progressive application of activation and weight transformations
while maintaining model accuracy through smooth adaptation.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `adaptation_manager.py` | `AdaptationManager` | Manages decorator rates (activation_adaptation, clamp, shift, quantization); clamp uses deterministic `MixAdjustmentStrategy`, quantization keeps nested random-mask path; for TTFS omits standalone shift decorator but nests shift inside QuantizeDecorator.  **Per-perceptron overrides (Phase B1)**: scalar fields (e.g. `clamp_rate`) remain the global default, but `set_per_perceptron_rate(rate_name, perceptron_name, value)` and `set_per_perceptron_schedule(rate_name, t, sensitivities)` register per-perceptron rates that win inside `update_activation()` via `get_rate()`.  Sensitivity-weighted schedule: given a global progress `t ∈ [0,1]` and per-perceptron sensitivities, rate for each perceptron is `clip(t * s_max / s_i, 0, 1)` -- least-sensitive perceptrons saturate first. |
| `smart_smooth_adaptation.py` | `SmartSmoothAdaptation` | Standalone greedy bisection loop; `get_target` callable (no internal target adjuster); step-size formula `(1.0 - t) * 2` (first probe at rate 1.0); `_adjust_minimum_step` doubles `min_step` but does not escalate `tolerance`; tolerance set at construction |
| `tolerance_calibration.py` | `ToleranceCalibrationConfig`, `estimate_tolerable_instant_drop`, `effective_probe_lr`, `LrProbeSpec`, `initial_tolerance_fn_for_pipeline_if_enabled` | Optional probe ladder; probes use `train_n_steps`; default `residual_threshold` is 0.02 (2%) |
| `basic_interpolation.py` | `BasicInterpolation` | Linear interpolation utilities for adaptation schedules |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster`, `target_decay_from_validation_samples`, `from_pipeline` | Proportional target decay; `floor_ratio` derived as `1 - degradation_tolerance`; growth capped at `original_metric`; `update_target(post_acc)` called only when the tuner is stuck (3+ consecutive committed steps < 1%), not on every cycle |
| `learning_rate_explorer.py` | `LRRangeFinder`, `find_lr_range_for_trainer`, `clone_state_for_trainer`, `restore_state_for_trainer` | Multi-step exponential LR sweep with "largest non-destructive" selection heuristic (picks highest LR that does not degrade accuracy beyond `margin`); `anchor_lr` parameter centres range on `pipeline_lr` (±1 order of magnitude); `margin` derived from `budget.accuracy_se()`; restores state after each probe |
| `tuning_budget.py` | `TuningBudget`, `tuning_budget_from_pipeline`, `max_total_training_steps`, `min_step_for_smooth_adaptation` | All fields derived from `check_interval = sqrt(SPE)`; `max_training_steps = 3 * SPE * budget_scale`; `lr_steps_per_probe = lr_num_probes = tolerance_probe_steps = check_interval`; `eval_n_batches` from validation set size; `eval_sample_count` tracks number of evaluation samples; `accuracy_se()` returns `0.5 / sqrt(eval_sample_count)` — the Bernoulli worst-case standard error used to derive all tuning thresholds |
| `unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `CATASTROPHIC_DROP_FACTOR`, `_RECOVERY_PATIENCE`, `_SMALL_STEP_THRESHOLD`, `_STUCK_STREAK_REQUIRED` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr=pipeline_lr`, validate).  `SmoothAdaptationTuner`: the single orchestration loop (baseline calibration -> one-shot -> SmartSmoothAdaptation -> _continue_to_full_rate -> _after_run); `run()` calibrates tuner target from `validate_n_batches` at rate 0.0 before adaptation starts; max_cycles capped to `min(30, ...)`.  **All decision probes use validation only** — `trainer.test()` is never called from tuner code.  Subclasses implement `_update_and_evaluate(rate)` and optionally `_recovery_training_hooks(rate)`, `_before_cycle()`, `_after_run()`.  `_adaptation()`: skips LR discovery when instant_acc is near target (uses `pipeline_lr`); scales recovery budget proportionally to accuracy gap; tracks committed step sizes via `_small_step_streak` — target relaxation only triggers after `_STUCK_STREAK_REQUIRED` (3) consecutive committed steps smaller than `_SMALL_STEP_THRESHOLD` (1%); rollbacks never trigger target decay; one-shot validation gate at rate 1.0 compares `validate_n_batches` against the validation baseline minus measurement noise.  Two tolerance levels: `_rollback_tolerance` (`3 * accuracy_se` — noise-only guard) and `_pipeline_tolerance` (strict pipeline dt).  `_ensure_validation_threshold()` (aliased to `_ensure_pipeline_threshold` for back-compat) runs a validation-only last-resort retry using the pipeline hard floor when available.  `_flush_enforcement_hooks()` runs exactly one eval-mode forward pass at the end of `_after_run` so any pre-hook-driven state (pruning masks, clamp limits) is consistent before the pipeline measures the final metric.  `_RECOVERY_PATIENCE` (5): default patience for recovery training |
| `shift_calculation.py` | `calculate_activation_shift` | Computes activation shift amounts for quantization alignment |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `tuners/` | Concrete tuner implementations for specific transformations |

## Dependencies

- **Internal**: `models.layers` (all decorator types), `model_training` (trainers).
- **External**: `torch`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports `AdaptationManager` (model building),
  `calculate_activation_shift` (activation shift step), and tuners.

## Exported API (\_\_init\_\_.py)

`AdaptationManager`, `SmartSmoothAdaptation`, `LRRangeFinder`,
`calculate_activation_shift`.
