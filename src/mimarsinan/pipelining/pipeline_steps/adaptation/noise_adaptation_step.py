"""Optional noise-injection adaptation step (gated by ``enable_training_noise``)."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner


class NoiseAdaptationStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return plan.is_lif_style and plan.enable_training_noise

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        self.run_tuner(
            NoiseTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
        )
