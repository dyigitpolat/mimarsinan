from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner


class ActivationShiftStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return (not plan.is_lif_style) and plan.activation_quantization

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None

    def cleanup(self):
        if self.trainer is not None:
            self.trainer.close()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.tuner = ActivationShiftTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
        )
        self.trainer = self.tuner.trainer
        self.tuner.run()
        self._commit_tuner_entries(model, adaptation_manager)
