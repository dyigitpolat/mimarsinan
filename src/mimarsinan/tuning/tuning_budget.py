"""Step-based tuning budget derived from dataset size and batch configuration.

All quantities derive from ``check_interval = sqrt(steps_per_epoch)`` -- no
hardcoded constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mimarsinan.data_handling.data_provider import DataProvider


@dataclass
class TuningBudget:
    """Generous max budget with convergence-aware check interval."""

    max_training_steps: int
    check_interval: int
    validation_steps: int
    eval_n_batches: int
    lr_steps_per_probe: int
    lr_num_probes: int
    tolerance_probe_steps: int
    max_lr_exploration_steps: int = 0
    eval_sample_count: int = 0
    progress_eval_batches: int = 16

    def accuracy_se(self) -> float:
        """Bernoulli worst-case standard error for the evaluation metric.

        ``0.5 / sqrt(n)`` where *n* is the number of samples used by
        ``validate_n_batches(eval_n_batches)``.  Every accuracy-comparison
        threshold in the tuning system is a multiple of this quantity.
        """
        return 0.5 / math.sqrt(max(1, self.eval_sample_count))

    @staticmethod
    def from_dataset(
        dataset_size: int,
        batch_size: int,
        budget_scale: float = 1.0,
        *,
        val_set_size: int | None = None,
        val_batch_size: int | None = None,
        degradation_tolerance: float = 0.05,
    ) -> TuningBudget:
        """All fields derived from ``check_interval = sqrt(SPE)``."""
        bs = max(1, int(batch_size))
        steps_per_epoch = max(1, int(dataset_size) // bs)
        # check_interval capped to keep recovery cycles short on large datasets.
        # 100 training steps between validations is enough to detect convergence.
        check_interval = max(1, min(100, int(math.sqrt(float(steps_per_epoch)))))
        # max_training_steps capped at 2000 -- pretrained models recover in
        # far fewer steps; longer runs are diminishing returns.
        spe_budget = int(float(steps_per_epoch) * float(budget_scale))
        max_training_steps = max(1, min(2000, spe_budget))
        validation_steps = max(1, min(32, check_interval))

        # LR probes need only enough steps to detect destructive divergence.
        # 30 steps is sufficient; 8 probes total covers anchor/100 to anchor*10.
        lr_num_probes = 8
        lr_steps_per_probe = min(30, check_interval)
        tolerance_probe_steps = min(50, check_interval)

        if val_set_size is not None and val_batch_size is not None:
            vbs = max(1, int(val_batch_size))
            total_val_batches = max(1, int(val_set_size) // vbs)
            # Cap eval_n_batches so commit/rollback decisions stay <30s.
            # SE = 0.5/sqrt(N) -> 5000 samples gives SE=0.007, plenty for
            # rollback decisions where rollback_tolerance >= 0.005.
            target_eval_samples = 5000
            target_batches = max(1, target_eval_samples // vbs)
            eval_n_batches = max(validation_steps, min(target_batches, total_val_batches))
            eval_sample_count = eval_n_batches * vbs
        else:
            eval_n_batches = validation_steps
            eval_sample_count = eval_n_batches * max(1, bs)

        progress_eval_batches = max(1, min(16, eval_n_batches))

        return TuningBudget(
            max_training_steps=max_training_steps,
            check_interval=check_interval,
            validation_steps=validation_steps,
            eval_n_batches=eval_n_batches,
            lr_steps_per_probe=lr_steps_per_probe,
            lr_num_probes=lr_num_probes,
            tolerance_probe_steps=tolerance_probe_steps,
            max_lr_exploration_steps=lr_steps_per_probe * lr_num_probes,
            eval_sample_count=eval_sample_count,
            progress_eval_batches=progress_eval_batches,
        )

    @staticmethod
    def from_data_provider(
        data_provider: DataProvider,
        budget_scale: float = 1.0,
        degradation_tolerance: float = 0.05,
    ) -> TuningBudget:
        return TuningBudget.from_dataset(
            data_provider.get_training_set_size(),
            data_provider.get_training_batch_size(),
            budget_scale,
            val_set_size=data_provider.get_validation_set_size(),
            val_batch_size=data_provider.get_validation_batch_size(),
            degradation_tolerance=degradation_tolerance,
        )


def tuning_budget_from_pipeline(pipeline) -> TuningBudget:
    """Build a :class:`TuningBudget` from ``pipeline.config`` and data provider."""
    dp = pipeline.data_provider_factory.create()
    return TuningBudget.from_data_provider(
        dp,
        float(pipeline.config.get("tuning_budget_scale", 1.0)),
        degradation_tolerance=float(pipeline.config.get("degradation_tolerance", 0.05)),
    )


def max_total_training_steps(pipeline) -> int:
    """Upper bound on gradient steps for full ``training_epochs`` (for ``SmartSmoothAdaptation`` ``min_step``)."""
    dp = pipeline.data_provider_factory.create()
    bs = max(1, dp.get_training_batch_size())
    spe = max(1, (dp.get_training_set_size() + bs - 1) // bs)
    te = max(1, int(pipeline.config.get("training_epochs", 10)))
    return max(1, te * spe)


def min_step_for_smooth_adaptation(pipeline, budget: TuningBudget) -> float:
    """``max(0.001, budget.check_interval / max_total_training_steps)``."""
    m = max_total_training_steps(pipeline)
    return max(0.001, float(budget.check_interval) / float(m))
