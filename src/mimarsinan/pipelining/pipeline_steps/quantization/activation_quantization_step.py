from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner


class ActivationQuantizationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    # The staged AQ install (sync's conversion endpoint) carries
    # the measured rung-2 training-draw variance.
    DRAW_SELECTED = True

    @classmethod
    def applies_to(cls, plan):
        return plan.requires_activation_quantization_preconditioning

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
