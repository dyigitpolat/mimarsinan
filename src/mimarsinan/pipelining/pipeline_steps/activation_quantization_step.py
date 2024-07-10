from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner 

class ActivationQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = ActivationQuantizationTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'] * 0.01, 
            adaptation_manager = self.get_entry('adaptation_manager'))
        self.tuner.run()

        self.update_entry("adaptation_manager", self.tuner.adaptation_manager, "pickle")
        self.update_entry("model", self.tuner.model, 'torch_model')