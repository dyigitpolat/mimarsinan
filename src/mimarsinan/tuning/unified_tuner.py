"""TunerBase and SmoothAdaptationTuner — shared tuning orchestration."""

from __future__ import annotations

import time

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    find_lr_range_for_trainer,
    restore_state_for_trainer,
)
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.tuning_budget import (
    min_step_for_smooth_adaptation,
    resolve_tuning_batch_size,
    tuning_budget_from_pipeline,
)


CATASTROPHIC_DROP_FACTOR = 0.8
"""Fast-fail threshold as a fraction of the adaptation target."""

_RECOVERY_PATIENCE = 5
"""Default patience for recovery training."""

_STUCK_STREAK_REQUIRED = 3
"""Consecutive cycles missing the target before it is relaxed."""


class TunerBase:
    """Shared infrastructure for all tuners."""

    _budget_multiplier = 1.0

    _skip_one_shot = False

    def __init__(self, pipeline, model, target_accuracy, lr):
        self.pipeline = pipeline
        self.model = model
        self.pipeline_lr = lr
        self.lr = lr
        self.name = "Tuning Rate"

        self._budget = tuning_budget_from_pipeline(pipeline)
        if self._budget_multiplier != 1.0:
            self._budget.max_training_steps = int(
                self._budget.max_training_steps * self._budget_multiplier
            )
        self.target_adjuster = AdaptationTargetAdjuster.from_pipeline(
            target_accuracy, pipeline
        )

        self.trainer = self._create_trainer()
        self.trainer.report_function = pipeline.reporter.report

    def _tuning_recipe(self):
        """Recipe for tuning-phase trainers (explicit opt-in via tuning_recipe)."""
        return build_recipe(self.pipeline.config, key="tuning_recipe")

    def _create_trainer(self):
        num_workers = self.pipeline.config.get("num_workers", 4)
        trainer = BasicTrainer(
            self.model,
            self.pipeline.config["device"],
            DataLoaderFactory(self.pipeline.data_provider_factory,
                              num_workers=num_workers),
            self.pipeline.loss,
            recipe=self._tuning_recipe(),
        )
        tuning_bs = resolve_tuning_batch_size(self.pipeline, trainer.training_batch_size)
        if tuning_bs != trainer.training_batch_size:
            trainer.set_training_batch_size(tuning_bs)
        return trainer

    def close(self):
        """Shut down DataLoader workers owned by this tuner."""
        if hasattr(self, "trainer") and self.trainer is not None:
            self.trainer.close()

    def _find_lr(self):
        with self.trainer.validation_context("probe"):
            return find_lr_range_for_trainer(
                self.trainer,
                self.pipeline,
                self._budget,
                validate_fn=lambda: self.trainer.validate_n_batches(
                    self._budget.progress_eval_batches
                ),
                anchor_lr=self.pipeline_lr,
            )

    def _get_cached_lr(self):
        if getattr(self, "_cached_lr", None) is None:
            self._cached_lr = self._find_lr()
        return self._cached_lr

    def _invalidate_lr_cache(self):
        self._cached_lr = None

    def _get_target(self):
        return self.target_adjuster.get_target()

    def validate(self):
        return self.trainer.validate()

    @property
    def final_metric(self):
        """Cached final test-consistent metric set by ``_after_run``."""
        return getattr(self, "_final_metric", None)

    def run(self):
        raise NotImplementedError


class SmoothAdaptationTuner(TunerBase):
    """Orchestration loop for smooth rate-based adaptation."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._natural_rate = 0.0
        self._pipeline_tolerance = float(
            pipeline.config.get("degradation_tolerance", 0.05)
        )
        se = self._budget.accuracy_se()
        self._rollback_tolerance = 3 * se
        self._missed_target_streak = 0
        self._pipeline_hard_floor = None
        self._pre_relaxation_target = None
        self._validation_baseline = None
        self._cycle_log = []
        self._cached_lr = None

    def _update_and_evaluate(self, rate):
        """Apply transformation T at *rate* and return a validation metric."""
        raise NotImplementedError

    def _before_cycle(self):
        """Called before each adaptation cycle (e.g. refresh pruning importance)."""

    def _after_run(self):
        """Called after adaptation completes. Returns the final metric."""
        return self.trainer.validate()

    def _recovery_training_hooks(self, rate):
        """Return a list of hooks to keep active during recovery training.

        Subclasses (e.g. PruningTuner) override this to register forward
        pre-hooks that enforce invariants (e.g. pruning masks) throughout
        recovery. Hooks are removed after recovery finishes.
        """
        return []

    def _get_extra_state(self):
        """Override to return tuner-specific state to save alongside model params."""
        return None

    def _set_extra_state(self, extra):
        """Override to restore tuner-specific state."""

    def _clone_state(self):
        model_state = clone_state_for_trainer(self.trainer)
        return (model_state, self._get_extra_state())

    def _restore_state(self, state):
        model_state, extra = state
        restore_state_for_trainer(self.trainer, model_state)
        if extra is not None:
            self._set_extra_state(extra)

    def _absolute_post_acc_floor(self):
        """Baseline-anchored absolute floor for the per-cycle rollback gate."""
        baseline = getattr(self, "_validation_baseline", None)
        pipeline_floor = getattr(self, "_pipeline_hard_floor", None)
        tolerance = getattr(self, "_pipeline_tolerance", None)

        floors = []
        if baseline is not None and tolerance is not None:
            floors.append(float(baseline) * (1.0 - float(tolerance)))
        if pipeline_floor is not None:
            floors.append(float(pipeline_floor))
        if not floors:
            return None
        return max(floors)

    def _adaptation(self, rate):
        """Recovery training at a given rate with rollback on regression."""
        t_cycle_start = time.time()
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        pre_state = self._clone_state()
        pre_cycle_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        with self.trainer.validation_context("probe"):
            instant_acc = self._update_and_evaluate(rate)

        catastrophic_floor = self._get_target() * CATASTROPHIC_DROP_FACTOR
        if instant_acc is not None and float(instant_acc) < catastrophic_floor:
            self._restore_state(pre_state)
            if hasattr(self, "_cycle_log"): self._cycle_log.append({
                "rate": rate, "committed": self._committed_rate,
                "instant_acc": float(instant_acc), "outcome": "catastrophic",
                "elapsed_sec": time.time() - t_cycle_start,
            })
            return self._committed_rate

        t0 = time.time()
        lr = self._get_cached_lr()
        t_lr = time.time() - t0
        self.pipeline.reporter.report("LR_found", lr)
        self.pipeline.reporter.report("T_find_lr_sec", t_lr)

        hooks = self._recovery_training_hooks(rate)
        try:
            self.trainer.train_steps_until_target(
                lr,
                self._budget.max_training_steps,
                self._get_target(),
                0,
                validation_n_batches=self._budget.progress_eval_batches,
                check_interval=self._budget.check_interval,
                patience=_RECOVERY_PATIENCE,
                min_steps=self._budget.check_interval * 3,
                min_improvement=self._budget.accuracy_se(),
            )
        finally:
            for h in hooks:
                h.remove()

        post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        noise_margin = self._rollback_tolerance
        relative_threshold = pre_cycle_acc - noise_margin
        absolute_floor = self._absolute_post_acc_floor()
        if absolute_floor is not None:
            rollback_threshold = max(relative_threshold, absolute_floor)
        else:
            rollback_threshold = relative_threshold
        if post_acc < rollback_threshold:
            self._restore_state(pre_state)
            self._invalidate_lr_cache()
            if hasattr(self, "_cycle_log"): self._cycle_log.append({
                "rate": rate, "committed": self._committed_rate,
                "instant_acc": float(instant_acc) if instant_acc is not None else None,
                "pre_cycle_acc": float(pre_cycle_acc),
                "post_acc": float(post_acc), "lr": lr,
                "outcome": "rollback", "elapsed_sec": time.time() - t_cycle_start,
            })
            return self._committed_rate

        self._committed_rate = rate
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)

        reached_target = post_acc >= self._get_target() - noise_margin
        if reached_target:
            self._missed_target_streak = 0
            pre_relax = getattr(self, "_pre_relaxation_target", None)
            if pre_relax is not None:
                self.target_adjuster.target_metric = pre_relax
                self._pre_relaxation_target = None
        else:
            self._missed_target_streak += 1

        if self._missed_target_streak >= _STUCK_STREAK_REQUIRED:
            self._pre_relaxation_target = self._get_target()
            self.target_adjuster.update_target(post_acc)
            abs_floor = self._absolute_post_acc_floor()
            if abs_floor is not None:
                self.target_adjuster.target_metric = max(
                    self.target_adjuster.target_metric, abs_floor
                )
            self._missed_target_streak = 0
            self._invalidate_lr_cache()

        if hasattr(self, "_cycle_log"): self._cycle_log.append({
            "rate": rate, "committed": self._committed_rate,
            "pre_cycle_acc": float(pre_cycle_acc),
            "post_acc": float(post_acc), "lr": lr,
            "reached_target": bool(reached_target),
            "outcome": "commit", "elapsed_sec": time.time() - t_cycle_start,
        })
        return rate

    def _attempt_recovery_if_below_floor(self):
        """Last-resort recovery when validation is below the pipeline floor."""
        hard_floor = getattr(self, "_pipeline_hard_floor", None)

        n_eval = self._budget.eval_n_batches
        best_val = float(self.trainer.validate_n_batches(n_eval))
        if hard_floor is None:
            return best_val
        if best_val >= hard_floor:
            return best_val

        best_state = self._clone_state()

        def _attempt_lrs():
            yield self._get_cached_lr()
            yield self.pipeline_lr

        for attempt, lr_to_use in enumerate(_attempt_lrs()):
            hooks = self._recovery_training_hooks(1.0)
            try:
                self.trainer.train_steps_until_target(
                    lr_to_use,
                    self._budget.max_training_steps,
                    self._get_target(),
                    0,
                    validation_n_batches=self._budget.progress_eval_batches,
                    check_interval=self._budget.check_interval,
                    patience=_RECOVERY_PATIENCE,
                    min_steps=self._budget.check_interval * 3,
                    min_improvement=self._budget.accuracy_se() / 2,
                )
            finally:
                for h in hooks:
                    h.remove()
            val_acc = float(self.trainer.validate_n_batches(n_eval))
            if val_acc > best_val:
                best_val = val_acc
                best_state = self._clone_state()
            if best_val >= hard_floor:
                self._restore_state(best_state)
                return best_val

        self._restore_state(best_state)
        if best_val < hard_floor:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: could not recover validation "
                f"above pipeline floor after retries "
                f"(best_validation={best_val:.4f}, floor={hard_floor:.4f}). "
                "The pipeline's step-level test assertion will determine "
                "whether this step fails overall.",
                stacklevel=2,
            )
        return best_val

    def _ensure_pipeline_threshold(self):
        return self._attempt_recovery_if_below_floor()

    def _stabilization_budget(self):
        """Number of gradient steps for the final rate=1.0 stabilization pass."""
        return 2 * int(self._budget.max_training_steps)

    def _stabilize_at_full_rate(self):
        """Extra training at rate=1.0 after ``_after_run`` has committed."""
        if self._committed_rate < 1.0 - 1e-6:
            return
        budget = self._stabilization_budget()
        if budget is None or int(budget) <= 0:
            return
        budget = int(budget)

        lr = min(float(self._get_cached_lr()), float(self.pipeline_lr))

        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        try:
            pre_val = float(self.trainer.validate_n_batches(n_eval))
        except Exception:
            pre_val = None

        progress_n = max(
            self._budget.progress_eval_batches,
            self._budget.eval_n_batches,
        )
        max_patience = max(_RECOVERY_PATIENCE, budget // max(1, self._budget.check_interval))
        hooks = self._recovery_training_hooks(1.0)
        try:
            self.trainer.train_steps_until_target(
                lr,
                budget,
                self._get_target(),
                0,
                validation_n_batches=progress_n,
                check_interval=self._budget.check_interval,
                patience=max_patience,
                min_steps=budget,
                min_improvement=self._budget.accuracy_se() / 2,
            )
        finally:
            for h in hooks:
                h.remove()

        if pre_val is not None:
            try:
                post_val = float(self.trainer.validate_n_batches(n_eval))
            except Exception:
                post_val = None
            if post_val is not None and post_val < pre_val - self._rollback_tolerance:
                self._restore_state(pre_state)
                self._invalidate_lr_cache()
                self.pipeline.reporter.report(
                    f"{self.name} stabilization rollback",
                    {"pre": pre_val, "post": post_val},
                )

    def _continue_to_full_rate(self):
        """Continue gradual adaptation from committed rate toward 1.0."""
        current = self._committed_rate
        if current >= 1.0 - 1e-6:
            return

        remaining = 1.0 - current
        step = remaining / 4.0
        max_attempts = min(20, max(4, int(1.0 / max(step, 1e-6))))
        attempts = 0

        while current < 1.0 - 1e-6 and attempts < max_attempts:
            target = min(current + step, 1.0)
            result = self._adaptation(target)

            if result is not None and float(result) < target - 1e-9:
                step /= 2.0
                if step < 1e-4:
                    break
            else:
                current = target
                remaining = 1.0 - current
                step = max(remaining / 4.0, step)
            attempts += 1

    def run(self):
        self._committed_rate = 0.0
        self._natural_rate = 0.0
        self._small_step_streak = 0
        self._pre_relaxation_target = None
        self._cycle_log = []
        self._cached_lr = None

        self._pipeline_tolerance = float(
            self.pipeline.config.get("degradation_tolerance", 0.05)
        )

        pipeline_prev = getattr(self.pipeline, "get_target_metric", lambda: None)()
        budget = getattr(self.pipeline, "accuracy_budget", None)
        if (
            pipeline_prev is not None
            and float(pipeline_prev) > 0
            and budget is not None
        ):
            self._pipeline_hard_floor = budget.step_floor(
                previous_metric=float(pipeline_prev),
                per_step_tolerance=self._pipeline_tolerance,
            )
            if self._pipeline_hard_floor <= 0.0:
                self._pipeline_hard_floor = float(pipeline_prev) * (
                    1.0 - self._pipeline_tolerance
                )
        elif pipeline_prev is not None and float(pipeline_prev) > 0:
            self._pipeline_hard_floor = float(pipeline_prev) * (1.0 - self._pipeline_tolerance)
        else:
            self._pipeline_hard_floor = None

        se = self._budget.accuracy_se()
        val_a = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        val_b = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        empirical_noise = abs(val_a - val_b)
        self._rollback_tolerance = max(
            min(max(3 * se, 3 * empirical_noise), 0.05),
            0.005,
        )

        baseline_val = (val_a + val_b) / 2.0
        self.target_adjuster.target_metric = baseline_val
        self.target_adjuster.original_metric = baseline_val
        self.target_adjuster.floor = baseline_val * (1.0 - self._pipeline_tolerance)

        self._validation_baseline = baseline_val

        self.pipeline.reporter.report("BUDGET", {
            "max_training_steps": self._budget.max_training_steps,
            "check_interval": self._budget.check_interval,
            "eval_n_batches": self._budget.eval_n_batches,
            "progress_eval_batches": self._budget.progress_eval_batches,
            "lr_num_probes": self._budget.lr_num_probes,
            "accuracy_se": se,
            "rollback_tolerance": self._rollback_tolerance,
            "pipeline_hard_floor": self._pipeline_hard_floor,
        })

        if not self._skip_one_shot:
            self._before_cycle()
            self._adaptation(1.0)
            if self._committed_rate >= 1.0 - 1e-6:
                self._natural_rate = self._committed_rate
                result = self._after_run()
                assert self._committed_rate >= 1.0 - 1e-6, (
                    f"Tuning rate must reach 1.0 by the end of the step, "
                    f"but _committed_rate is {self._committed_rate:.6f}"
                )
                self._stabilize_at_full_rate()
                self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
                self._log_cycle_summary()
                return result

        ms = min_step_for_smooth_adaptation(self.pipeline, self._budget)
        max_cycles = min(30, max(
            10,
            self._budget.max_training_steps // max(1, self._budget.check_interval),
        ))

        adapter = SmartSmoothAdaptation(
            self._adaptation,
            interpolators=[BasicInterpolation(0.0, 1.0)],
            get_target=self._get_target,
            min_step=ms,
            before_cycle=self._before_cycle,
        )
        adapter.adapt_smoothly(max_cycles=max_cycles)

        if self._committed_rate < 1.0 - 1e-6:
            self._continue_to_full_rate()

        self._natural_rate = self._committed_rate

        if self._natural_rate < 1.0 - 1e-6:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: natural adaptation reached only "
                f"{self._natural_rate:.4f}; _after_run will force to 1.0",
                stacklevel=2,
            )

        result = self._after_run()
        assert self._committed_rate >= 1.0 - 1e-6, (
            f"Tuning rate must reach 1.0 by the end of the step, "
            f"but _committed_rate is {self._committed_rate:.6f}"
        )
        self._stabilize_at_full_rate()
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self._log_cycle_summary()
        return result

    def _log_cycle_summary(self):
        """Print the full adaptation cycle log for debugging."""
        if not self._cycle_log:
            return
        print(f"[{self.__class__.__name__}] Cycle summary ({len(self._cycle_log)} cycles):")
        for i, entry in enumerate(self._cycle_log):
            parts = [f"  [{i}] rate={entry.get('rate', '?'):.4f}"]
            parts.append(f"committed={entry.get('committed', '?'):.4f}")
            if "post_acc" in entry:
                parts.append(f"post_acc={entry['post_acc']:.4f}")
            if "lr" in entry:
                parts.append(f"lr={entry['lr']:.2e}")
            parts.append(f"outcome={entry.get('outcome', '?')}")
            parts.append(f"elapsed={entry.get('elapsed_sec', 0):.1f}s")
            print(" ".join(parts))


UnifiedPerceptronTuner = SmoothAdaptationTuner
