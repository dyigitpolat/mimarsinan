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
        tuning_batch_size: int | None = None,
    ) -> TuningBudget:
        """Budget derived from the *tuning* batch size.

        ``batch_size`` is the training batch size; ``tuning_batch_size``
        overrides it for the tuning phase alone (``None`` keeps the
        training batch size, preserving legacy behavior). Smaller tuning
        batches ⇒ more gradient updates per epoch of data, finer
        recovery, and lower activation memory; the sample-count budget
        (recovery_samples ≈ 1 epoch × budget_scale) is invariant under
        the batch-size change.
        """
        train_bs = max(1, int(batch_size))
        tuning_bs = max(1, int(tuning_batch_size)) if tuning_batch_size is not None else train_bs
        steps_per_epoch = max(1, int(dataset_size) // tuning_bs)
        # check_interval capped to keep recovery cycles short on large datasets.
        # 100 training steps between validations is enough to detect convergence.
        check_interval = max(1, min(100, int(math.sqrt(float(steps_per_epoch)))))
        # Recovery budget is sample-based (≈ budget_scale × 1 epoch of data),
        # so smaller tuning batches give proportionally more gradient steps.
        # Cap raised to 4000 so the common bs//4 default does not get pinned.
        spe_budget = int(float(steps_per_epoch) * float(budget_scale))
        max_training_steps = max(1, min(4000, spe_budget))
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
            min_eval_batches = min(validation_steps, total_val_batches)
            eval_n_batches = max(min_eval_batches, min(target_batches, total_val_batches))
            eval_sample_count = eval_n_batches * vbs
        else:
            eval_n_batches = validation_steps
            eval_sample_count = eval_n_batches * max(1, tuning_bs)

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
        *,
        tuning_batch_size: int | None = None,
    ) -> TuningBudget:
        return TuningBudget.from_dataset(
            data_provider.get_training_set_size(),
            data_provider.get_training_batch_size(),
            budget_scale,
            val_set_size=data_provider.get_validation_set_size(),
            val_batch_size=data_provider.get_validation_batch_size(),
            degradation_tolerance=degradation_tolerance,
            tuning_batch_size=tuning_batch_size,
        )


def resolve_tuning_batch_size(pipeline, training_batch_size: int) -> int:
    """Resolve the tuning-phase batch size from ``pipeline.config``.

    Explicit ``tuning_batch_size`` in config wins. Otherwise defaults to
    ``max(16, training_batch_size // 4)`` — smaller batches keep the
    tuning phase light (less activation memory, finer gradient updates),
    independent of the training-phase batch size. Users can opt out by
    setting ``tuning_batch_size`` equal to ``batch_size``.
    """
    cfg_val = pipeline.config.get("tuning_batch_size")
    if cfg_val is not None:
        return max(1, int(cfg_val))
    return max(16, max(1, int(training_batch_size)) // 4)


def tuning_budget_from_pipeline(pipeline) -> TuningBudget:
    """Build a :class:`TuningBudget` from ``pipeline.config`` and data provider."""
    dp = pipeline.data_provider_factory.create()
    tuning_bs = resolve_tuning_batch_size(pipeline, dp.get_training_batch_size())
    return TuningBudget.from_data_provider(
        dp,
        float(pipeline.config.get("tuning_budget_scale", 1.0)),
        degradation_tolerance=float(pipeline.config.get("degradation_tolerance", 0.05)),
        tuning_batch_size=tuning_bs,
    )


def max_total_training_steps(pipeline) -> int:
    """Upper bound on gradient steps for full ``training_epochs`` (for ``SmartSmoothAdaptation`` ``min_step``).

    Intentionally keyed to the *training* batch size — this is the upper
    bound of a full training run, not the tuning budget, so it must not
    collapse when the tuning batch size is larger than the dataset (as it
    trivially is on unit-test fixtures).
    """
    dp = pipeline.data_provider_factory.create()
    bs = max(1, dp.get_training_batch_size())
    spe = max(1, (dp.get_training_set_size() + bs - 1) // bs)
    te = max(1, int(pipeline.config.get("training_epochs", 10)))
    return max(1, te * spe)


def min_step_for_smooth_adaptation(pipeline, budget: TuningBudget) -> float:
    """``max(0.001, budget.check_interval / max_total_training_steps)``."""
    m = max_total_training_steps(pipeline)
    return max(0.001, float(budget.check_interval) / float(m))
