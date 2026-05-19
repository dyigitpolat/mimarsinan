from mimarsinan.pipelining.tuner_pipeline_step import TunerPipelineStep
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
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.tuner = ClampTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
            activation_scales=self.get_entry("activation_scales"),
            activation_scale_stats=self.get_entry("activation_scale_stats"),
        )
        self.tuner.run()
        self._commit_tuner_entries(model, adaptation_manager)
