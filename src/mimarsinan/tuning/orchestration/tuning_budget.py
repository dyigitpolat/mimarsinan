"""Step-based tuning budget derived from dataset size and batch configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.data_handling.data_provider import DataProvider

# Frozen workload-neutral clamps (profile-overridable): the absolute per-tuner
# step cap and the evaluation-subset target, both tier-0-proven defaults.
_GENERIC_STEP_CAP = 4000
_GENERIC_EVAL_SUBSAMPLE_TARGET = 5000


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
        """Bernoulli worst-case standard error ``0.5 / sqrt(n)`` for the evaluation
        metric; every accuracy-comparison threshold is a multiple of this."""
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
        scale_ramp_steps: bool = False,
        eval_subsample_target: int | None = None,
        tuning_step_cap_epochs: float | None = None,
    ) -> TuningBudget:
        """Budget derived from the *tuning* batch size.

        ``tuning_batch_size`` overrides ``batch_size`` for the tuning phase alone
        (``None`` keeps the training batch size). The sample-count budget
        (≈ 1 epoch × budget_scale) is invariant under the batch-size change.
        ``eval_subsample_target`` / ``tuning_step_cap_epochs`` are the
        workload-profile clamps; ``None`` keeps the frozen generic defaults.
        """
        train_bs = max(1, int(batch_size))
        tuning_bs = max(1, int(tuning_batch_size)) if tuning_batch_size is not None else train_bs
        steps_per_epoch = max(1, int(dataset_size) // tuning_bs)
        check_interval = max(1, min(100, int(math.sqrt(float(steps_per_epoch)))))
        spe_budget = int(float(steps_per_epoch) * float(budget_scale))
        base_cap = (
            _GENERIC_STEP_CAP
            if tuning_step_cap_epochs is None
            else max(1, int(float(tuning_step_cap_epochs) * steps_per_epoch))
        )
        cap = max(base_cap, spe_budget) if scale_ramp_steps else base_cap
        max_training_steps = max(1, min(cap, spe_budget))
        validation_steps = max(1, min(32, check_interval))

        lr_num_probes = 8
        lr_steps_per_probe = min(30, check_interval)
        tolerance_probe_steps = min(50, check_interval)

        if val_set_size is not None and val_batch_size is not None:
            vbs = max(1, int(val_batch_size))
            total_val_batches = max(1, int(val_set_size) // vbs)
            target_eval_samples = (
                _GENERIC_EVAL_SUBSAMPLE_TARGET
                if eval_subsample_target is None
                else max(1, int(eval_subsample_target))
            )
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
        scale_ramp_steps: bool = False,
        eval_subsample_target: int | None = None,
        tuning_step_cap_epochs: float | None = None,
    ) -> TuningBudget:
        return TuningBudget.from_dataset(
            data_provider.get_training_set_size(),
            data_provider.get_training_batch_size(),
            budget_scale,
            val_set_size=data_provider.get_validation_set_size(),
            val_batch_size=data_provider.get_validation_batch_size(),
            degradation_tolerance=degradation_tolerance,
            tuning_batch_size=tuning_batch_size,
            scale_ramp_steps=scale_ramp_steps,
            eval_subsample_target=eval_subsample_target,
            tuning_step_cap_epochs=tuning_step_cap_epochs,
        )


def resolve_tuning_batch_size(pipeline, training_batch_size: int) -> int:
    """Resolve the tuning-phase batch size from ``pipeline.config``.

    Explicit ``tuning_batch_size`` wins; otherwise defaults to
    ``max(16, training_batch_size // 4)``.
    """
    cfg_val = pipeline.config.get("tuning_batch_size")
    if cfg_val is not None:
        return max(1, int(cfg_val))
    return max(16, max(1, int(training_batch_size)) // 4)


def tuning_budget_from_pipeline(pipeline) -> TuningBudget:
    """Build a :class:`TuningBudget` from ``pipeline.config`` and data provider."""
    dp = pipeline.data_provider_factory.create()
    tuning_bs = resolve_tuning_batch_size(pipeline, dp.get_training_batch_size())
    workload = ResolvedWorkloadProfile.from_config(pipeline.config)
    return TuningBudget.from_data_provider(
        dp,
        float(pipeline.config.get("tuning_budget_scale", 1.0)),
        degradation_tolerance=float(pipeline.config.get("degradation_tolerance", 0.05)),
        tuning_batch_size=tuning_bs,
        scale_ramp_steps=bool(pipeline.config.get("tuning_budget_scale_ramp_steps", False)),
        eval_subsample_target=workload.eval_subsample_target,
        tuning_step_cap_epochs=workload.tuning_step_cap_epochs,
    )


def max_total_training_steps(pipeline) -> int:
    """Upper bound on gradient steps for full ``training_epochs`` (the scheduler's
    ``epsilon`` min step). Keyed to the training batch size, not the tuning budget,
    so it does not collapse when the tuning batch size exceeds the dataset.
    """
    dp = pipeline.data_provider_factory.create()
    bs = max(1, dp.get_training_batch_size())
    spe = max(1, (dp.get_training_set_size() + bs - 1) // bs)
    te = max(1, int(pipeline.config.get("training_epochs", 10)))
    return max(1, te * spe)


_MAX_MIN_STEP = 0.05


def min_step_for_smooth_adaptation(pipeline, budget: TuningBudget) -> float:
    """``check_interval / max_total_training_steps``, clamped to ``[0.001, 0.05]``.

    The upper clamp matters for tiny models, where the raw ratio can reach ~1/3 —
    far too coarse for the scheduler to bisect into a committable foothold.
    """
    m = max_total_training_steps(pipeline)
    return min(_MAX_MIN_STEP, max(0.001, float(budget.check_interval) / float(m)))
