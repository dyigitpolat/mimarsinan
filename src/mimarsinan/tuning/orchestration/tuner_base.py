"""TunerBase and SmoothAdaptationTuner — shared tuning orchestration."""

from __future__ import annotations

import time

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    find_lr_range_for_trainer,
    make_loss_slope_signal,
    restore_state_for_trainer,
)
from mimarsinan.tuning.orchestration.tuning_budget import (
    min_step_for_smooth_adaptation,
    resolve_tuning_batch_size,
    tuning_budget_from_pipeline,
)


CATASTROPHIC_DROP_FACTOR = 0.8
"""Pre-recovery fast-fail margin, as a fraction of the adaptation target.

Deliberately coarse, and NOT a standard-error gate: ``is_catastrophic`` runs on
the *instant* accuracy right after a transformation step is applied but BEFORE
recovery training, where a large drop is expected and routinely reclaimed by
recovery. A statistically tight ``target - k·SE`` threshold would abort almost
every cycle before recovery could run. 0.8 bails only when the raw post-transform
accuracy has collapsed past a fifth of target — beyond any recoverable range. It
is the default ``factor`` of ``AcceptanceSensor.is_catastrophic`` (injectable, so
a caller may tighten it without mutating this module-level default).
"""

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
        # Rank the coarse LR sweep by a cheap training loss-slope signal (reserving
        # full validation for the top few). Trainers without a loss-slope signal
        # (some stubs) fall back to full-validation scoring.
        coarse_signal = make_loss_slope_signal(self.trainer)
        with self.trainer.validation_context("probe"):
            return find_lr_range_for_trainer(
                self.trainer,
                self.pipeline,
                self._budget,
                validate_fn=lambda: self.trainer.validate_n_batches(
                    self._budget.progress_eval_batches
                ),
                anchor_lr=self.pipeline_lr,
                coarse_signal=coarse_signal,
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
