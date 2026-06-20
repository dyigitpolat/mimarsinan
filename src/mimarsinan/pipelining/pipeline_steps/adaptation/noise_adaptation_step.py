"""Optional noise-injection adaptation step (gated by ``enable_training_noise``)."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner


class NoiseAdaptationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return plan.is_lif_style and plan.enable_training_noise

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        self.run_tuner(
            NoiseTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
        )
