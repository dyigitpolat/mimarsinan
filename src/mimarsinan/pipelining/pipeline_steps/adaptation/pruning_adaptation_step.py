"""PruningAdaptationStep: pipeline step for progressive weight pruning."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.pruning.pruning_tuner import PruningTuner


class PruningAdaptationStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return plan.pruning_enabled

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        self.run_tuner(
            PruningTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
            pruning_fraction=self.pipeline.config.get("pruning_fraction", 0.0),
        )
