from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner


class ClampAdaptationStep(TunerPipelineStep):
    """Introduces activation clamping (ClampDecorator) with recovery training."""

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales", "activation_scale_stats"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        self.run_tuner(
            ClampTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
            activation_scales=self.get_entry("activation_scales"),
            activation_scale_stats=self.get_entry("activation_scale_stats"),
        )
