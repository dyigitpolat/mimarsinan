# tuning/ -- Training-Aware Tuning Subsystem

Manages the progressive application of activation and weight transformations
while maintaining model accuracy through smooth adaptation.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `adaptation_manager.py` | `AdaptationManager` | Manages decorator rates (activation_adaptation, clamp, shift, quantization); clamp uses deterministic `MixAdjustmentStrategy`, quantization keeps nested random-mask path; for TTFS omits standalone shift decorator but nests shift inside QuantizeDecorator |
| `smart_smooth_adaptation.py` | `SmartSmoothAdaptation` | Standalone greedy bisection loop; `get_target` callable (no internal target adjuster); step-size formula `(1.0 - t) * 2` (first probe at rate 1.0); `_adjust_minimum_step` doubles `min_step` but does not escalate `tolerance`; tolerance set at construction |
| `tolerance_calibration.py` | `ToleranceCalibrationConfig`, `estimate_tolerable_instant_drop`, `effective_probe_lr`, `LrProbeSpec`, `initial_tolerance_fn_for_pipeline_if_enabled` | Optional probe ladder; probes use `train_n_steps`; default `residual_threshold` is 0.02 (2%) |
| `basic_interpolation.py` | `BasicInterpolation` | Linear interpolation utilities for adaptation schedules |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster`, `target_decay_from_validation_samples`, `from_pipeline` | Proportional target decay with midpoint pull for large misses; `floor_ratio` (default 0.90); growth capped at original; pipeline config key `tuner_target_floor_ratio` |
| `learning_rate_explorer.py` | `LRRangeFinder`, `find_lr_range_for_trainer`, `clone_state_for_trainer`, `restore_state_for_trainer` | Multi-step exponential LR sweep; `num_probes` and `steps_per_probe` from `TuningBudget`; restores state after each probe |
| `tuning_budget.py` | `TuningBudget`, `tuning_budget_from_pipeline`, `max_total_training_steps`, `min_step_for_smooth_adaptation` | All fields derived from `check_interval = sqrt(SPE)`; `max_training_steps = 3 * SPE * budget_scale`; `lr_steps_per_probe = lr_num_probes = tolerance_probe_steps = check_interval`; `eval_n_batches` from validation set size |
| `unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `CATASTROPHIC_DROP_FACTOR`, `_RECOVERY_PATIENCE` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder, validate). `SmoothAdaptationTuner`: the single orchestration loop (one-shot -> SmartSmoothAdaptation -> _after_run); subclasses implement `_update_and_evaluate(rate)` and optionally `_recovery_training_hooks(rate)` (returns forward-pre-hooks active during recovery training), `_before_cycle()`, `_after_run()`. `_adaptation()` includes LR search, hook-wrapped recovery, rollback. Two tolerance levels: `_rollback_tolerance` (max of pipeline dt and 2% — permissive, for adaptation progress) and `_pipeline_tolerance` (strict pipeline dt). `_ensure_pipeline_threshold()` uses `_pipeline_tolerance` for last-resort retry with `min_improvement=1e-4` so sub-0.1% gains accumulate. `_RECOVERY_PATIENCE` (5): default patience for recovery training |
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
