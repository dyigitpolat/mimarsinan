from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner


class ActivationQuantizationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return (not plan.is_lif_style) and plan.activation_quantization

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.run_tuner(
            ActivationQuantizationTuner,
            model,
            adaptation_manager,
            target_tq=self.pipeline.config["target_tq"],
        )
