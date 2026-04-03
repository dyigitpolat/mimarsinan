from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.tuner = None
        self.trainer = None

    def validate(self):
        if self.tuner is not None:
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def cleanup(self):
        if self.trainer is not None:
            self.trainer.close()

    def process(self):
        model = self.get_entry("model")

        adaptation_manager = self.get_entry('adaptation_manager')
        self.tuner = ActivationShiftTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
        )
        self.trainer = self.tuner.trainer
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, 'torch_model')
