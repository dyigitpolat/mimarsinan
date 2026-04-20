# tuning/ -- Training-Aware Tuning Subsystem

Manages the progressive application of activation and weight transformations
while maintaining model accuracy through smooth adaptation.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `adaptation_manager.py` | `AdaptationManager` | Manages decorator rates (activation_adaptation, clamp, shift, quantization); clamp uses deterministic `MixAdjustmentStrategy`, quantization keeps nested random-mask path; for TTFS omits standalone shift decorator but nests shift inside QuantizeDecorator |
| `smart_smooth_adaptation.py` | `SmartSmoothAdaptation` | Standalone greedy bisection loop; `get_target` callable (no internal target adjuster); step-size formula `(1.0 - t) * 2` (first probe at rate 1.0); `_adjust_minimum_step` doubles `min_step` but does not escalate `tolerance`; tolerance set at construction |
| `basic_interpolation.py` | `BasicInterpolation` | Linear interpolation utilities for adaptation schedules |
| `per_layer_schedule.py` | `uniform_rate_fn`, `LinearPerLayerSchedule`, `build_per_layer_schedule` | Optional per-perceptron rate schedule (opt-in via `config["per_layer_rate_schedule"]`); scalar-rate default preserved; endpoint invariants (rate 0.0 and rate 1.0 stay uniform across layers) |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster`, `target_decay_from_validation_samples`, `from_pipeline` | Proportional target decay; `floor_ratio` derived as `1 - degradation_tolerance`; growth capped at `original_metric`; `update_target(post_acc)` called only when the tuner is stuck (3+ consecutive committed steps < 1%), not on every cycle |
| `learning_rate_explorer.py` | `LRRangeFinder`, `find_lr_range_for_trainer`, `clone_state_for_trainer`, `restore_state_for_trainer` | Multi-step exponential LR sweep with "largest non-destructive" selection heuristic (picks highest LR that does not degrade accuracy beyond `margin`); `anchor_lr` parameter centres range on `pipeline_lr` (±1 order of magnitude); `margin` derived from `budget.accuracy_se()`; restores state after each probe |
| `tuning_budget.py` | `TuningBudget`, `tuning_budget_from_pipeline`, `max_total_training_steps`, `min_step_for_smooth_adaptation` | All fields derived from `check_interval = sqrt(SPE)`; `max_training_steps = 3 * SPE * budget_scale`; `lr_steps_per_probe = lr_num_probes = tolerance_probe_steps = check_interval`; `eval_n_batches` from validation set size; `eval_sample_count` tracks number of evaluation samples; `accuracy_se()` returns `0.5 / sqrt(eval_sample_count)` — the Bernoulli worst-case standard error used to derive all tuning thresholds |
| `unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `CATASTROPHIC_DROP_FACTOR`, `_RECOVERY_PATIENCE`, `_SMALL_STEP_THRESHOLD`, `_STUCK_STREAK_REQUIRED` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr=pipeline_lr`, validate). `SmoothAdaptationTuner`: the single orchestration loop (baseline calibration -> one-shot -> SmartSmoothAdaptation -> _continue_to_full_rate -> _after_run -> _stabilize_at_full_rate); `run()` calibrates tuner target from `validate_n_batches` at rate 0.0 before adaptation starts; max_cycles capped to `min(30, ...)`. Subclasses implement `_update_and_evaluate(rate)` and optionally `_recovery_training_hooks(rate)`, `_before_cycle()`, `_after_run()`, `_stabilization_budget()`. `_adaptation()`: skips LR discovery when instant_acc is near target (uses `pipeline_lr`); scales recovery budget proportionally to accuracy gap; tracks committed step sizes via `_small_step_streak` — target relaxation only triggers after `_STUCK_STREAK_REQUIRED` (3) consecutive committed steps smaller than `_SMALL_STEP_THRESHOLD` (1%); rollbacks never trigger target decay; the rate=1.0 internal gate uses `_validation_baseline` (captured once at run start) — `trainer.test()` is never called in tuner internals (see "Test-set isolation" below). Two tolerance levels: `_rollback_tolerance` (`3 * accuracy_se` — noise-only guard) and `_pipeline_tolerance` (strict pipeline dt). `_attempt_recovery_if_below_floor()` (aliased as `_ensure_pipeline_threshold`) uses validation-only recovery. `_RECOVERY_PATIENCE` (5): default patience for recovery training. See "Baseline-anchored absolute floor" and "Post-step stabilization" below |
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

## SmoothAdaptationTuner — invariants

### Baseline-anchored absolute floor

The per-cycle rollback gate in `_adaptation()` enforces two thresholds, and
takes the stricter:

1. **Relative (noise) gate.** `post_acc >= pre_cycle_acc - _rollback_tolerance`.
   Catches individual-cycle regressions outside measurement noise.
2. **Absolute (baseline-anchored) gate.** `post_acc >= max(_validation_baseline
   * (1 - _pipeline_tolerance), _pipeline_hard_floor)`. This prevents
   cumulative drift across many cycles: without it, every cycle can drop
   by up to `_rollback_tolerance` relative to its predecessor, and over
   ~N cycles the committed model could silently degrade by ~`N *
   rollback_tolerance` without any single cycle looking regressive.

`_validation_baseline` is captured exactly once at run start (the average
of two consecutive `validate_n_batches` calls at rate 0.0). When no
baseline has been seeded (e.g. unit tests that invoke `_adaptation()`
without calling `run()`), the gate falls back to the relative threshold.

The `AdaptationTargetAdjuster`'s target decay is also clamped to the
baseline-anchored floor, so a stuck-streak relaxation cannot take the
tuner below the cumulative-drift guard.

### Post-step stabilization

After `_after_run()` has forced the final `rate == 1.0` state and the
validation-only safety net (`_attempt_recovery_if_below_floor`) has run,
`run()` calls `_stabilize_at_full_rate()`. This runs one extra
`train_steps_until_target` pass with:

- LR: the cached pipeline LR (no new LR search).
- Budget: `_stabilization_budget()` steps — default `2 *
  _budget.max_training_steps`. Subclasses can return `None` / `0` to
  disable (or a larger value for tuners that benefit more from extra
  stabilization).
- Hooks: `_recovery_training_hooks(1.0)` installed/removed via
  `try/finally` so pruning and decorator invariants are enforced
  throughout and released on exception.
- Rollback: pre-stabilization state is restored if validation drops by
  more than `_rollback_tolerance`, so this pass can never make the model
  worse than the point it was invoked at.

This gives the model additional training time at the committed rate=1.0
configuration — the regime where we historically saw slow but steady
accuracy recovery that ran out of budget inside the per-cycle `_adaptation`
loop.

### Rate-aware weight quantization

`transformations/normalization_aware_perceptron_quantization.py` accepts
a `rate` argument that linearly interpolates in weight-value space between
the FP and fully-quantized effective weights: `rate * q(w) + (1 - rate) *
w`. At `rate == 0` it is identity; at `rate == 1` it matches the legacy
full-quantization output bit-exactly. Partial rates produce a *coherent*
small perturbation of the FP model rather than a random mix of FP and
integer weights — which is what the legacy `_mix_params` random-mask mix
produced on top of a rate-ignorant transformation, and the reason
`NormalizationAwarePerceptronQuantizationTuner` used to fast-fail on every
partial-rate cycle. `parameter_scale` is always set to the full-range
scale (so downstream IR mapping is unaffected by the rate).
