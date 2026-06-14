"""Smooth adaptation cycle, recovery, and rollback logic."""

from __future__ import annotations

import time
import warnings

from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    restore_state_for_trainer,
)
from mimarsinan.tuning.trace import DecisionRecord, DecisionTrace
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuner_base import (
    TunerBase,
    _RECOVERY_PATIENCE,
    _STUCK_STREAK_REQUIRED,
)


class SmoothAdaptationCycleMixin(TunerBase):
    """Adaptation cycle, rollback, and recovery training."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._checkpoint_guard = None
        if pipeline.config.get("tuning_use_checkpoint_guard", False):
            self._checkpoint_guard = CheckpointGuard(
                self.trainer,
                scope=pipeline.config.get("checkpoint_scope", "full"),
                location=pipeline.config.get("checkpoint_location", "device"),
            )
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
        self._cycle_log = DecisionTrace.new()
        self._cached_lr = None
        # P2b paired-McNemar rollback gate (vs the fixed baseline correctness
        # vector on a shared confirm subsample); off unless flagged.
        self._paired_gate = bool(pipeline.config.get("tuning_use_paired_sensor", False))
        self._k_commit = float(pipeline.config.get("k_commit", 2.0))
        self._confirm_indices = None
        self._ref_correct = None

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
        guard = getattr(self, "_checkpoint_guard", None)
        if guard is not None:
            return (guard.snapshot(), self._get_extra_state())
        model_state = clone_state_for_trainer(self.trainer)
        return (model_state, self._get_extra_state())

    def _restore_state(self, state):
        model_state, extra = state
        guard = getattr(self, "_checkpoint_guard", None)
        if guard is not None:
            guard.restore(model_state)
        else:
            restore_state_for_trainer(self.trainer, model_state)
        if extra is not None:
            self._set_extra_state(extra)

    def _baseline_or_none(self):
        baseline = getattr(self, "_validation_baseline", None)
        return float(baseline) if baseline is not None else None

    def _absolute_post_acc_floor(self):
        """Baseline-anchored absolute floor for the per-cycle rollback gate."""
        return AcceptanceSensor.absolute_floor(
            getattr(self, "_validation_baseline", None),
            getattr(self, "_pipeline_tolerance", None),
            getattr(self, "_pipeline_hard_floor", None),
        )

    def _adaptation(self, rate):
        """Recovery training at a given rate with rollback on regression."""
        t_cycle_start = time.time()
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        pre_state = self._clone_state()
        last = getattr(self, "_last_post_acc", None)
        pre_cycle_acc = float(last) if last is not None else \
            self.trainer.validate_n_batches(self._budget.eval_n_batches)

        with self.trainer.validation_context("probe"):
            instant_acc = self._update_and_evaluate(rate)

        if AcceptanceSensor.is_catastrophic(instant_acc, self._get_target()):
            self._restore_state(pre_state)
            if hasattr(self, "_cycle_log"): self._cycle_log.record(DecisionRecord(
                cycle_index=len(self._cycle_log),
                outcome="catastrophic",
                rate=float(rate),
                committed=float(self._committed_rate),
                elapsed_sec=time.time() - t_cycle_start,
                instant_acc=float(instant_acc),
                pre_cycle_acc=float(pre_cycle_acc),
                target=float(self._get_target()),
                validation_baseline=self._baseline_or_none(),
                rollback_tolerance=float(self._rollback_tolerance),
            ))
            return self._committed_rate

        t0 = time.time()
        lr = self._get_cached_lr()
        t_lr = time.time() - t0
        self.pipeline.reporter.report("LR_found", lr)
        self.pipeline.reporter.report("T_find_lr_sec", t_lr)

        hooks = self._recovery_training_hooks(rate)
        RecoveryEngine.train_to_target(
            self.trainer,
            lr,
            self._get_target(),
            max_steps=self._budget.max_training_steps,
            validation_n_batches=self._budget.progress_eval_batches,
            check_interval=self._budget.check_interval,
            patience=_RECOVERY_PATIENCE,
            min_steps=self._budget.check_interval * 3,
            min_improvement=self._budget.accuracy_se(),
            hooks=hooks,
        )

        post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        noise_margin = self._rollback_tolerance
        absolute_floor = self._absolute_post_acc_floor()
        rollback_threshold = AcceptanceSensor.rollback_threshold(
            pre_cycle_acc, noise_margin, absolute_floor
        )
        if getattr(self, "_paired_gate", False) and self._ref_correct is not None:
            cand_correct = self.trainer.validate_correctness_on_indices(self._confirm_indices)
            rolled_back = AcceptanceSensor.paired_is_rollback(
                self._ref_correct, cand_correct, self._k_commit
            )
        else:
            rolled_back = AcceptanceSensor.is_rollback(post_acc, rollback_threshold)
        if rolled_back:
            self._restore_state(pre_state)
            if hasattr(self, "_cycle_log"): self._cycle_log.record(DecisionRecord(
                cycle_index=len(self._cycle_log),
                outcome="rollback",
                rate=float(rate),
                committed=float(self._committed_rate),
                elapsed_sec=time.time() - t_cycle_start,
                instant_acc=float(instant_acc) if instant_acc is not None else None,
                pre_cycle_acc=float(pre_cycle_acc),
                post_acc=float(post_acc),
                lr=float(lr),
                target=float(self._get_target()),
                validation_baseline=self._baseline_or_none(),
                rollback_threshold=float(rollback_threshold),
                absolute_floor=(float(absolute_floor) if absolute_floor is not None else None),
                rollback_tolerance=float(noise_margin),
            ))
            return self._committed_rate

        self._committed_rate = rate
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)

        reached_target = AcceptanceSensor.reached_target(
            post_acc, self._get_target(), noise_margin
        )
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

        if hasattr(self, "_cycle_log"): self._cycle_log.record(DecisionRecord(
            cycle_index=len(self._cycle_log),
            outcome="commit",
            rate=float(rate),
            committed=float(self._committed_rate),
            elapsed_sec=time.time() - t_cycle_start,
            pre_cycle_acc=float(pre_cycle_acc),
            post_acc=float(post_acc),
            lr=float(lr),
            reached_target=bool(reached_target),
            target=float(self._get_target()),
            validation_baseline=self._baseline_or_none(),
            rollback_threshold=float(rollback_threshold),
            absolute_floor=(float(absolute_floor) if absolute_floor is not None else None),
            rollback_tolerance=float(noise_margin),
        ))
        self._last_post_acc = float(post_acc)
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
            RecoveryEngine.train_to_target(
                self.trainer,
                lr_to_use,
                self._get_target(),
                max_steps=self._budget.max_training_steps,
                validation_n_batches=self._budget.progress_eval_batches,
                check_interval=self._budget.check_interval,
                patience=_RECOVERY_PATIENCE,
                min_steps=self._budget.check_interval * 3,
                min_improvement=self._budget.accuracy_se() / 2,
                hooks=hooks,
            )
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

