from mimarsinan.pipelining.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner


class ActivationQuantizationStep(TunerPipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry('model')
        adaptation_manager = self.get_entry('adaptation_manager')
        self.tuner = ActivationQuantizationTuner(
            self.pipeline,
            model=model,
            target_tq=self.pipeline.config['target_tq'],
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config['lr'],
            adaptation_manager=adaptation_manager,
        )
        self.tuner.run()
        self._commit_tuner_entries(model, self.tuner.adaptation_manager)
