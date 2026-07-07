"""Base pipeline step for steps that own a BasicTrainer."""

from __future__ import annotations

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    METRIC_MEASURED,
    PipelineStep,
)


class TrainerPipelineStep(PipelineStep):
    """Shared validate pattern; cleanup is handled by PipelineStep.cleanup()."""

    def __init__(self, requires, promises, updates, clears, pipeline):
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer: BasicTrainer | None = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_MEASURED if self.trainer is not None else METRIC_CARRIED
