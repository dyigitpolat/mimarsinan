from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner 

class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["pretrained_model", "pt_accuracy"]
        promises = ["ca_model", "ca_accuracy"]
        clears = ["pretrained_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = ClampTuner(
            self.pipeline,
            model = self.pipeline.cache['pretrained_model'],
            target_accuracy = self.pipeline.cache['pt_accuracy'] * 0.99,
            lr = self.pipeline.config['lr'])
        
        validation_accuracy = tuner.run()

        assert validation_accuracy > self.pipeline.cache['pt_accuracy'] * 0.9, \
            "Clamp adaptation step failed to retain validation accuracy."

        self.pipeline.cache.add("ca_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("ca_accuracy", validation_accuracy)

        self.pipeline.cache.remove("pretrained_model")

        
