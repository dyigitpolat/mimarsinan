# tuning/orchestration/ -- Smooth-adaptation orchestration loop + services

The control core of the tuning subsystem: the rate-search loop, the per-cycle
predictor→corrector→commit/rollback machinery, the adaptation-manager rate host,
and the decision services the refactor is extracting from the loop.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `tuner_base.py` | `TunerBase`, `CATASTROPHIC_DROP_FACTOR`, `_RECOVERY_PATIENCE`, `_STUCK_STREAK_REQUIRED` | Shared infrastructure: trainer, budget, target adjuster, LR finder (`_find_lr`/`_get_cached_lr` with `anchor_lr=pipeline_lr`), `validate`. Tuning constants. |
| `smooth_adaptation_cycle.py` | `SmoothAdaptationCycleMixin` | One adaptation cycle: clone pre-state → pre-cycle eval → `_update_and_evaluate(rate)` probe → catastrophic gate → LR → recovery train → dual rollback gate → commit + stuck-streak target relaxation. Records one `DecisionRecord` per exit into `self._cycle_log`. Accept/reject decisions delegate to `AcceptanceSensor` (static methods); `_absolute_post_acc_floor` delegates too. |
| `smooth_adaptation_run.py` | `SmoothAdaptationRunMixin` | `run()`: baseline calibration via `AcceptanceSensor.calibrate_baseline` → one-shot `_adaptation(1.0)` → `SmartSmoothAdaptation` ramp → `_continue_to_full_rate` → `_after_run` → `_stabilize_at_full_rate`. `_log_cycle_summary` prints the `DecisionTrace`. |
| `smooth_adaptation_tuner.py` | `SmoothAdaptationTuner` (= Cycle+Run+Base mixins), `UnifiedPerceptronTuner` (alias) | The composed orchestration tuner all rate tuners extend. |
| `acceptance_sensor.py` | `AcceptanceSensor`, `BaselineRef` | The accept/reject/recovered decision service. Pure, bit-exact extraction of the loop's statistical math: `calibrate_baseline` (SE, empirical noise, clamped rollback tolerance, fixed baseline), `absolute_floor` (cumulative-drift guard), `is_catastrophic`/`rollback_threshold`/`is_rollback`/`reached_target`. **Paired McNemar gate (P2b)**: `paired_drop_se` / `paired_is_rollback` test the candidate vs the fixed-baseline correctness vector on a shared subsample (`(b10-b01)/N`, SE `sqrt(b10+b01)/N` — several-fold tighter than the marginal `0.5/sqrt(n)`). Wired into the rollback gate behind `tuning_use_paired_sensor` (off → marginal path bit-exact); Monte-Carlo-calibrated in `test_paired_sensor_calibration`. |
| `rate_scheduler.py` | `RateScheduler` | One rate-search policy (spec §5.2 greedy-to-1.0 then bisect-the-gap) that replaces the one-shot + `SmartSmoothAdaptation` + `_continue_to_full_rate` loops. Policies: `greedy_to_one` (default), `uniform_ladder` (KD-blend), `one_shot_only`. Wired behind `tuning_use_driver` via `SmoothAdaptationRunMixin._run_with_scheduler` (the driver path); reaches the same final committed rate as the legacy loops (outcome equivalence), with `_after_run`/`_stabilize_at_full_rate` reused unchanged. |
| `recovery_engine.py` | `RecoveryEngine` | The corrector. `train_to_target` centralizes the `train_steps_until_target` + recovery-hook install/remove-in-`finally` pattern shared by the per-cycle recovery, the below-floor safety net, and the full-rate stabilization pass (pure, bit-exact). LR discovery / persistent-optimizer / tunable-param folding are the flagged behavior-changing extensions layered on later. |
| `checkpoint_guard.py` | `CheckpointGuard`, `Handle` | Scoped, location-aware rollback snapshots. `scope=full`/`location=device` (default) delegates verbatim to `clone/restore_state_for_trainer` (byte-identical, golden-safe); `scope=tunable` clones only `requires_grad` params + buffers (skip frozen backbone); `location=cpu_pinned` offloads to CPU (frees ~1× model VRAM, syncs before restore). Wired into `_clone_state`/`_restore_state` behind `tuning_use_checkpoint_guard`; aux-model trainers fall back to full/device. |
| `adaptation_manager.py` | `AdaptationManager` | Rate-field host (`clamp_rate`/`quantization_rate`/`shift_rate`/`noise_rate`/`activation_adaptation_rate`/`scale_rate`); `update_activation` rebuilds a perceptron's decorator stack from current rates. |
| `adaptation_manager_factory.py` | `create_adaptation_manager_for_model` | Build a manager + run initial `update_activation` for all perceptrons. |
| `tuning_budget.py` | `TuningBudget`, `tuning_budget_from_pipeline`, `max_total_training_steps`, `min_step_for_smooth_adaptation` | Step-based budget from dataset size; `accuracy_se()` = `0.5/sqrt(eval_sample_count)`. |
| `kd_blend_adaptation_tuner.py` | `KDBlendAdaptationTuner`, `BlendActivation`, `_InstalledForward` | ANN→SNN blend ramp (LIF/TTFS base): live `BlendActivation.rate` via `set_blend_rate`, KD recovery against a frozen teacher, parity-critical `_finalize` forward-install. |

## Dependencies

- **Internal**: `tuning.trace` (decision artifact), `tuning.learning_rate_explorer`, `tuning.smart_smooth_adaptation`, `tuning.adaptation_target_adjuster`, `tuning.perceptron_rate`, `tuning.forward_install`, `tuning.teacher`, `model_training` (trainers).
- **External**: `torch`.

## Dependents

- `tuning.tuners.*` (concrete tuners extend `SmoothAdaptationTuner`/`KDBlendAdaptationTuner`).
- `tuning.adaptation_rate_tuner.AdaptationRateTuner`.
- `pipelining.pipeline_steps` (model building imports `AdaptationManager`).

## Exported API

The package `__init__.py` is empty; re-exports live in `tuning/__init__.py`.
Internal modules are imported by full path.
