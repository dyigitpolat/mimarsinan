# tuning/orchestration/ -- Smooth-adaptation orchestration loop + services

The control core of the tuning subsystem: the rate-search loop, the per-cycle
predictor→corrector→commit/rollback machinery, the adaptation-manager rate host,
and the decision services the refactor is extracting from the loop.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `tuner_base.py` | `TunerBase`, `CATASTROPHIC_DROP_FACTOR`, `_RECOVERY_PATIENCE`, `_STUCK_STREAK_REQUIRED` | Shared infrastructure: trainer, budget, target adjuster, LR finder (`_find_lr`/`_get_cached_lr` with `anchor_lr=pipeline_lr`), `validate`. Tuning constants. |
| `smooth_adaptation_cycle.py` | `SmoothAdaptationCycleMixin` | One adaptation cycle: clone pre-state → pre-cycle eval → `_update_and_evaluate(rate)` probe → catastrophic gate → LR → recovery train → dual rollback gate → commit + stuck-streak target relaxation. Records one `DecisionRecord` per exit into `self._cycle_log`. Accept/reject decisions delegate to `AcceptanceSensor` (static methods); `_absolute_post_acc_floor` delegates too. **Diagnostic** (`tuning_full_transform_probe`, default off): after each commit, `_probe_full_transform` measures the value-domain rate-1.0 accuracy from the committed state and logs `drop = committed_acc - full_acc`; `_log_full_transform_trend` reports CONVERGING vs FLAT/DIVERGING — i.e. whether the gradual ramp is pulling the model toward 1.0-viability (it is, for LIF/pruning: full_acc climbs ~0.57→0.94 as α→1; the residual gap is the separate `finalize_cliff`). |
| `smooth_adaptation_run.py` | `SmoothAdaptationRunMixin` | `run()`: baseline calibration via `AcceptanceSensor.calibrate_baseline` → `_run_with_scheduler` (the single `RateScheduler` loop, attempt = `_adaptation`) → `_finalize_run` (`_after_run` → `_stabilize_at_full_rate`). `_continue_to_full_rate` is retained as an `_after_run` helper for the rate-tuner families. `_log_cycle_summary` prints the `DecisionTrace`. |
| `smooth_adaptation_tuner.py` | `SmoothAdaptationTuner` (= Cycle+Run+Base mixins), `UnifiedPerceptronTuner` (alias) | The composed orchestration tuner all rate tuners extend. |
| `acceptance_sensor.py` | `AcceptanceSensor`, `BaselineRef` | The accept/reject/recovered decision service. Pure, bit-exact extraction of the loop's statistical math: `calibrate_baseline` (SE, empirical noise, clamped rollback tolerance, fixed baseline), `absolute_floor` (cumulative-drift guard), `is_catastrophic`/`rollback_threshold`/`is_rollback`/`reached_target`. **Paired McNemar gate (P2b)**: `paired_drop_se` / `paired_is_rollback` test the candidate vs the fixed-baseline correctness vector on a shared subsample (`(b10-b01)/N`, SE `sqrt(b10+b01)/N` — several-fold tighter than the marginal `0.5/sqrt(n)`). Wired into the rollback gate behind `tuning_use_paired_sensor` (off → marginal path bit-exact); Monte-Carlo-calibrated in `test_paired_sensor_calibration`. |
| `rate_scheduler.py` | `RateScheduler` | The single rate-search policy (spec §5.2 greedy-to-1.0 then bisect-the-gap) that replaced the legacy one-shot + grow/halve ramp + `_continue_to_full_rate` loops. The first (greedy/ladder) jump always runs; `epsilon` bounds only the bisection. Policies: `greedy_to_one` (default), `uniform_ladder` (KD-blend), `one_shot_only`. |
| `adaptation_driver.py` | `AdaptationDriver` | Thin orchestrator: `run()` drives the scheduler from `committed` toward 1.0 over a per-cycle `attempt` (predictor→corrector→commit/rollback), then `finalize`. `build_scheduler` selects the policy. The sole run loop, via `SmoothAdaptationRunMixin._run_with_scheduler` (attempt = the tuner's `_adaptation`, finalize = `_finalize_run`). The fully standalone form (built from model+axis+services, dissolving `_adaptation` into `_cycle`) is the remaining V6 step. |
| `characterization.py` | `characterize`, `Profile` | Pre-search profiling (spec §10 / V9): sweeps the paired drop on an α grid → `Profile(monotonic, max_slope, epsilon_hint, feasible_max, drops)`. A non-monotonicity beyond the noise budget (A1) is the trigger for dense-grid safe mode; the steepest slope sets `epsilon_hint` (A3) so bisection never steps over a cliff. |
| `recovery_engine.py` | `RecoveryEngine`, `PersistentOptimizerOwner`, `RESET_PER_CYCLE`, `PERSIST_WITHIN_CYCLE` | The corrector. `train_to_target` centralizes the `train_steps_until_target` + recovery-hook remove-in-`finally` pattern (per-cycle recovery, below-floor safety net, stabilization) and threads an optional owned `optimizer`. **Optimizer policy**: the cycle's `_optimizer_policy()`/`_recovery_optimizer(lr)` pass `optimizer=None` by default (fresh build + del each call — bit-exact), or, under the `tuning_persist_optimizer` opt-in, a `PersistentOptimizerOwner`-built optimizer reused across recovery calls so Adam moments survive (owner reset at `run()` start, getattr-guarded). Tuner families whose recovery replaces the parameter set each cycle set `_supports_persistent_optimizer = False` and always reset. LR discovery / tunable-param folding remain later extensions. |
| `checkpoint_guard.py` | `CheckpointGuard`, `Handle` | Scoped, location-aware rollback snapshots. `scope=full`/`location=device` (default) delegates verbatim to `clone/restore_state_for_trainer` (byte-identical, golden-safe); `scope=tunable` clones only `requires_grad` params + buffers (skip frozen backbone); `location=cpu_pinned` offloads to CPU (frees ~1× model VRAM, syncs before restore). The sole snapshot path in `_clone_state`/`_restore_state`; `scope`/`location` are config opt-ins defaulting to full/device. |
| `adaptation_manager.py` | `AdaptationManager` | Rate-field host (`clamp_rate`/`quantization_rate`/`shift_rate`/`noise_rate`/`activation_adaptation_rate`/`scale_rate`); `update_activation` rebuilds a perceptron's decorator stack from current rates. |
| `adaptation_manager_factory.py` | `create_adaptation_manager_for_model` | Build a manager + run initial `update_activation` for all perceptrons. |
| `tuning_budget.py` | `TuningBudget`, `tuning_budget_from_pipeline`, `max_total_training_steps`, `min_step_for_smooth_adaptation` | Step-based budget from dataset size; `accuracy_se()` = `0.5/sqrt(eval_sample_count)`. |
| `kd_blend_adaptation_tuner.py` | `KDBlendAdaptationTuner`, `BlendActivation`, `_InstalledForward` | ANN→SNN blend ramp (LIF/TTFS base): live `BlendActivation.rate` via `set_blend_rate`, KD recovery against a frozen teacher, parity-critical `_finalize` forward-install. |

## Dependencies

- **Internal**: `tuning.trace` (decision artifact), `tuning.axes` (rate application), `tuning.learning_rate_explorer`, `tuning.adaptation_target_adjuster`, `tuning.perceptron_rate`, `tuning.forward_install`, `tuning.teacher`, `model_training` (trainers).
- **External**: `torch`.

## Dependents

- `tuning.tuners.*` (concrete tuners extend `SmoothAdaptationTuner`/`KDBlendAdaptationTuner`).
- `tuning.adaptation_rate_tuner.AdaptationRateTuner`.
- `pipelining.pipeline_steps` (model building imports `AdaptationManager`).

## Exported API

The package `__init__.py` is empty; re-exports live in `tuning/__init__.py`.
Internal modules are imported by full path.
