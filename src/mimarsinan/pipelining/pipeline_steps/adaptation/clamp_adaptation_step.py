from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner


class ClampAdaptationStep(TunerPipelineStep):
    """Introduces activation clamping (ClampDecorator) with recovery training."""

    REQUIRES = ("model", "adaptation_manager", "activation_scales", "activation_scale_stats")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return (not plan.is_lif_style) and (
            plan.activation_quantization or plan.requires_ttfs_firing
        )

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        self.run_tuner(
            ClampTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
            activation_scales=self.get_entry("activation_scales"),
            activation_scale_stats=self.get_entry("activation_scale_stats"),
        )
