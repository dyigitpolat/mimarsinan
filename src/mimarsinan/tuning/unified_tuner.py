"""TunerBase and SmoothAdaptationTuner -- the single orchestration hierarchy.

TunerBase provides shared infrastructure (pipeline, model, trainer, budget,
target adjuster, LR finder). SmoothAdaptationTuner adds the greedy bisection
loop for smooth rate-based adaptation.

Concrete tuners override _update_and_evaluate(rate) and optionally
_before_cycle() / _after_run().
"""

from __future__ import annotations

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
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
    tuning_budget_from_pipeline,
)


CATASTROPHIC_DROP_FACTOR = 0.8
"""Fast-fail threshold as a fraction of the adaptation target.

If the instant accuracy after applying a transformation drops below
``target * CATASTROPHIC_DROP_FACTOR``, the cycle is abandoned immediately
without wasting compute on LR exploration and recovery training.
A value of 0.8 means any >20% instant drop is treated as unrecoverable.
"""


class TunerBase:
    """Shared infrastructure for all tuners (except CoreFlowTuner).

    Provides pipeline access, model, trainer, budget, target adjuster,
    LR finder, and validate().  Subclasses implement run().
    """

    def __init__(self, pipeline, model, target_accuracy, lr):
        self.pipeline = pipeline
        self.model = model
        self.pipeline_lr = lr
        self.lr = lr
        self.name = "Tuning Rate"

        self._budget = tuning_budget_from_pipeline(pipeline)
        self.target_adjuster = AdaptationTargetAdjuster.from_pipeline(
            target_accuracy, pipeline
        )

        self.trainer = self._create_trainer()
        self.trainer.report_function = pipeline.reporter.report

    def _create_trainer(self):
        return BasicTrainer(
            self.model,
            self.pipeline.config["device"],
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss,
        )

    def _find_lr(self):
        return find_lr_range_for_trainer(
            self.trainer,
            self.pipeline,
            self._budget,
            validate_fn=lambda: self.trainer.validate_n_batches(
                self._budget.eval_n_batches
            ),
        )

    def _get_target(self):
        return self.target_adjuster.get_target()

    def validate(self):
        return self.trainer.validate()

    def run(self):
        raise NotImplementedError


class SmoothAdaptationTuner(TunerBase):
    """The ONE orchestration loop for smooth rate-based adaptation.

    Subclasses must implement ``_update_and_evaluate(rate) -> float``.
    Optional hooks: ``_before_cycle()``, ``_after_run() -> float``.
    """

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._rollback_tolerance = float(
            pipeline.config.get("degradation_tolerance", 0.05)
        )

    def _update_and_evaluate(self, rate):
        """Apply transformation T at *rate* and return a validation metric."""
        raise NotImplementedError

    def _before_cycle(self):
        """Called before each adaptation cycle (e.g. refresh pruning importance)."""

    def _after_run(self):
        """Called after adaptation completes. Returns the final metric."""
        return self.trainer.validate()

    # -- State snapshot protocol ------------------------------------------------
    # _clone_state / _restore_state capture both model parameters AND any
    # tuner-specific "decoration" state (e.g. AdaptationManager rates).
    # Concrete tuners override _get_extra_state / _set_extra_state to include
    # their rate attributes; the base pair handles the model parameters.

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

    def _adaptation(self, rate):
        """Recovery training at a given rate: T -> L -> R, with rollback.

        Clones state before the transformation.  After recovery, if accuracy
        is below ``target * (1 - rollback_tolerance)``, restores the
        checkpoint.  Returns the *committed* rate so that
        ``SmartSmoothAdaptation`` can keep ``t`` in sync with the actual
        model state.
        """
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        pre_state = self._clone_state()

        instant_acc = self._update_and_evaluate(rate)

        # Fast-fail: skip expensive LR exploration + training if model collapsed
        catastrophic_floor = self._get_target() * CATASTROPHIC_DROP_FACTOR
        if instant_acc is not None and float(instant_acc) < catastrophic_floor:
            self._restore_state(pre_state)
            return self._committed_rate

        lr = self._find_lr()
        self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            self._get_target(),
            0,
            validation_n_batches=self._budget.validation_steps,
            check_interval=self._budget.check_interval,
            patience=3,
        )

        post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        threshold = self._get_target() * (1.0 - self._rollback_tolerance)
        if post_acc < threshold:
            self._restore_state(pre_state)
            return self._committed_rate
        else:
            self._committed_rate = rate
            return rate

    def _continue_to_full_rate(self):
        """Continue gradual adaptation from committed rate toward 1.0.

        When the main SmartSmoothAdaptation loop exits before reaching
        rate=1.0, this drives the remaining distance using the same
        _adaptation() mechanism with rollback so that the final commit
        is not a catastrophic one-shot jump.  Capped to avoid runaway cost.
        """
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

        self._rollback_tolerance = float(
            self.pipeline.config.get("degradation_tolerance", 0.05)
        )

        # ------------------------------------------------------------------
        # One-shot: try full transformation in a single cycle.
        # For light transformations (clamp, activation, quantization), this
        # typically succeeds — reducing 5-20 gradual cycles to just one.
        # For heavy transformations (pruning), instant accuracy drops below
        # CATASTROPHIC_DROP_FACTOR, triggering fast-fail with full state
        # restoration and zero wasted compute.
        # ------------------------------------------------------------------
        self._before_cycle()
        self._adaptation(1.0)
        if self._committed_rate >= 1.0 - 1e-6:
            return self._after_run()

        # ------------------------------------------------------------------
        # Gradual fallback: one-shot failed, so use SmartSmoothAdaptation
        # to incrementally drive rate from committed_rate toward 1.0.
        # ------------------------------------------------------------------
        ms = min_step_for_smooth_adaptation(self.pipeline, self._budget)
        max_cycles = max(
            10,
            self._budget.max_training_steps // max(1, self._budget.check_interval),
        )

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

        return self._after_run()


# Backward-compatible aliases used by tuners/__init__.py and existing imports.
UnifiedPerceptronTuner = SmoothAdaptationTuner
