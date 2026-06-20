"""PruningAdaptationStep: pipeline step for progressive weight pruning."""

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.pruning.pruning_tuner import PruningTuner


class PruningAdaptationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return plan.pruning_enabled

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        self.run_tuner(
            PruningTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
            pruning_fraction=DeploymentPlan.of(self.pipeline).pruning_fraction,
        )
