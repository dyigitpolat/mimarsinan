from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner 

class ActivationQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["shifted_activation_model"]
        promises = ["aq_model"]
        clears = ["shifted_activation_model"]
        super().__init__(requires, promises, clears, pipeline)
        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = ActivationQuantizationTuner(
            self.pipeline,
            model = self.pipeline.cache['shifted_activation_model'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.pipeline.cache.add("aq_model", self.tuner.model, 'torch_model')

        
