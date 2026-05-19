"""Base pipeline step for steps that own a BasicTrainer."""

from __future__ import annotations

from mimarsinan.pipelining.pipeline_step import PipelineStep


class TrainerPipelineStep(PipelineStep):
    """Shared validate pattern; cleanup is handled by PipelineStep.cleanup()."""

    def __init__(self, requires, promises, updates, clears, pipeline):
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()
