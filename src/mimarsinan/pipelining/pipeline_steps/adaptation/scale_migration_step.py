"""ScaleMigrationStep: exact cross-layer channel-scale migration before Activation Analysis."""

from typing import Iterable, cast

import torch

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.pipelining.pipeline_steps.activation_utils import analysis_batch_count
from mimarsinan.transformations.channel_scale_equalization import (
    equalize_channel_scales,
)
from mimarsinan.tuning.orchestration.install_resolution import (
    ChannelStatsAccumulator,
    collect_channel_stats,
)

# The migration is exact up to fp rounding (measured max|dlogit| 2.1e-6 on the
# mixer prototype); the postcondition budget leaves ~50x margin above that.
EXACTNESS_ATOL = 1e-4


def _channel_axis_accumulator(perceptron) -> ChannelStatsAccumulator:
    return ChannelStatsAccumulator(
        channel_axis=int(getattr(perceptron, "output_channel_axis", 1))
    )


class ScaleMigrationStep(TrainerPipelineStep):
    """Equalize per-channel activation scales across exactly-migratable pairs.

    Runs after pruning commits its masks and BEFORE Activation Analysis (theta
    is not yet installed, so the weight-space rescale is function-preserving).
    """

    REQUIRES = ("model",)
    PROMISES = ("scale_migration_report",)
    UPDATES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        return plan.scale_migration_enabled

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        self.trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        plan = DeploymentPlan.of(self.pipeline)

        n_batches = analysis_batch_count(self.pipeline)
        # cast: the validation cache yields (input, target) tensor pairs; _gpu_val_cache is untyped upstream.
        validation_batches = cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            self.trainer.iter_validation_batches(n_batches),
        )
        batches = [x for x, _ in validation_batches]
        stats = collect_channel_stats(
            model, batches, device, accumulator_factory=_channel_axis_accumulator,
        )
        reference = self._forward_all(model, batches, device)

        report = equalize_channel_scales(
            model,
            {id(perceptron): acc.per_channel_q99() for perceptron, acc in stats},
            clip_ratio=plan.scale_migration_clip_ratio,
        )
        self._require_function_preserved(model, batches, device, reference)

        print(
            "[ScaleMigrationStep] "
            f"migrated={[hop.name for hop in report.migrated]}, "
            f"skipped={list(report.skipped)}, clip_ratio={report.clip_ratio}"
        )
        self.update_entry("model", model, "torch_model")
        self.add_entry("scale_migration_report", report.as_dict())

    @staticmethod
    def _forward_all(model, batches, device):
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.no_grad():
                return [model(x.to(device)) for x in batches]
        finally:
            if was_training:
                model.train()

    def _require_function_preserved(self, model, batches, device, reference):
        """The section-3.1 identity is this step's postcondition, not a budget."""
        migrated_outputs = self._forward_all(model, batches, device)
        for ref, out in zip(reference, migrated_outputs):
            deviation = float((ref - out).abs().max())
            if deviation > EXACTNESS_ATOL:
                raise RuntimeError(
                    "[ScaleMigrationStep] scale migration failed to preserve "
                    f"the float function: max|dlogit| {deviation:.3e} > "
                    f"{EXACTNESS_ATOL:.0e}. A non-homogeneous op sits inside a "
                    "migrated pair; this is a pair-discovery bug, not a "
                    "tolerable degradation."
                )
