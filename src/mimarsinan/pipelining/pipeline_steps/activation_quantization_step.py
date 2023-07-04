from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.pipelining.pipeline_steps.tuners.activation_quantization_tuner import ActivationQuantizationTuner 

class ActivationQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["shifted_activation_model", "as_accuracy"]
        promises = ["aq_model", "aq_accuracy"]
        clears = ["shifted_activation_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = ActivationQuantizationTuner(
            self.pipeline,
            max_epochs = self.pipeline.config['aq_epochs'],
            model = self.pipeline.cache['shifted_activation_model'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.cache['as_accuracy'],
            lr = self.pipeline.config['lr'])
        
        validation_accuracy = tuner.run()

        assert validation_accuracy > self.pipeline.cache['as_accuracy'] * 0.9, \
            "Activation quantization step failed to retain validation accuracy."

        self.pipeline.cache.add("aq_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("aq_accuracy", validation_accuracy)

        self.pipeline.cache.remove("shifted_activation_model")

        
