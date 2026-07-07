"""TunerBase and SmoothAdaptationTuner — shared tuning orchestration."""

from __future__ import annotations

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.tuning.learning_rate_explorer import (
    find_lr_range_for_trainer,
    make_loss_slope_signal,
)
from mimarsinan.tuning.orchestration.tuning_budget import (
    resolve_tuning_batch_size,
    tuning_budget_from_pipeline,
)


CATASTROPHIC_DROP_FACTOR = 0.8
"""Pre-recovery fast-fail margin, as a fraction of the adaptation target.

Deliberately coarse, NOT a standard-error gate: the instant pre-recovery drop is
expected to be large and routinely reclaimed by recovery, so this bails only on
collapse beyond any recoverable range (the injectable ``is_catastrophic`` default).
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
        # [P2] frozen-BN-stats contract: tuner-internal training (recovery, LR
        # probes) must not drift BN running statistics — the committed metric
        # would be measured on stats the mapped artifact does not have, and
        # train-mode BN over a poisoned activation was the W2 shift-crater
        # kill condition.
        self.trainer.freeze_bn_stats_in_training = True
        self.trainer.report_function = pipeline.reporter.report

    def _tuning_recipe(self):
        """Recipe for tuning-phase trainers (explicit opt-in via tuning_recipe)."""
        return build_recipe(self.pipeline.config, key="tuning_recipe")

    def _create_trainer(self):
        trainer = BasicTrainer(
            self.model,
            self.pipeline.config["device"],
            DataLoaderFactory.for_pipeline(self.pipeline),
            self.pipeline.loss,
            recipe=self._tuning_recipe(),
            tuning_recipe_recovery=bool(
                self.pipeline.config.get("tuning_recipe_recovery", False)
            ),
        )
        tuning_bs = resolve_tuning_batch_size(self.pipeline, trainer.training_batch_size)
        if tuning_bs != trainer.training_batch_size:
            trainer.set_training_batch_size(tuning_bs)
        return trainer

    def close(self):
        """Shut down DataLoader workers owned by this tuner."""
        if hasattr(self, "trainer") and self.trainer is not None:
            self.trainer.close()

    def _find_lr(self) -> float | None:
        """Explorer LR for recovery; ``None`` = all-destructive refusal (fix C):
        the caller must fall back to the entry state instead of training."""
        coarse_signal = make_loss_slope_signal(self.trainer)
        with self.trainer.validation_context("probe"):
            lr = find_lr_range_for_trainer(
                self.trainer,
                self.pipeline,
                self._budget,
                validate_fn=lambda: self.trainer.validate_n_batches(
                    self._budget.progress_eval_batches
                ),
                anchor_lr=self.pipeline_lr,
                coarse_signal=coarse_signal,
            )
        if lr is None:
            print(
                f"[LR-REFUSE] {type(self).__name__} ({self.name}): the LR "
                "explorer found every candidate destructive; recovery training "
                "is skipped and the entry state is preserved.",
                flush=True,
            )
            self.pipeline.reporter.report(f"{self.name} lr_refusal", 1.0)
            emit_reporter_event(
                self.pipeline.reporter,
                "lr_refusal", {"tuner": type(self).__name__, "name": self.name},
            )
        return lr

    def _get_cached_lr(self) -> float | None:
        if getattr(self, "_lr_search_refused", False):
            return None
        cached = getattr(self, "_cached_lr", None)
        if cached is None:
            cached = self._find_lr()
            if cached is None:
                # Sticky until invalidation: re-probing an unchanged state
                # would re-burn the sweep budget for the same refusal.
                self._lr_search_refused = True
                return None
            self._cached_lr = cached
        return cached

    def _capped_cached_lr(self) -> float | None:
        """Cached explorer LR capped at the pipeline LR; ``None`` on refusal."""
        lr = self._get_cached_lr()
        if lr is None:
            return None
        return min(float(lr), float(self.pipeline_lr))

    def _invalidate_lr_cache(self):
        self._cached_lr = None
        self._lr_search_refused = False

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
