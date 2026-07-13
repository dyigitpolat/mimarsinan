"""ReferenceTeacherSnapshotStep: freeze the post-structural float model as the exact-QAT KD teacher."""

from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_kd_active
from mimarsinan.tuning.teacher import snapshot_frozen_teacher


class ReferenceTeacherSnapshotStep(PipelineStep):
    """[lif_exact_qat_program §8] Snapshot the post-structural float model.

    Runs after Pruning/Scale Migration and before the first function-changing
    activation step, so the frozen teacher computes the campaign's lossless
    reference (post-prune for pruned cells, G8). The AQ-hosted exact-QAT
    distils to it instead of training at plain CE (the measured worst KD arm).
    """

    REQUIRES = ("model",)
    PROMISES = ("reference_teacher_model",)

    @classmethod
    def applies_to(cls, plan):
        return lif_exact_qat_kd_active(plan.config)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        teacher = snapshot_frozen_teacher(model, self.pipeline.config["device"])
        self.add_entry("reference_teacher_model", teacher, "torch_model")

    def validate(self):
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED
