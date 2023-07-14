from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["pretrained_model", "pt_accuracy"]
        promises = ["na_model", "na_accuracy"]
        clears = ["pretrained_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = NoiseTuner(
            self.pipeline,
            max_epochs = self.pipeline.config['aq_epochs'],
            model = self.pipeline.cache['pretrained_model'],
            target_accuracy = self.pipeline.cache['pt_accuracy'] * 0.99,
            lr = self.pipeline.config['lr'])
        
        validation_accuracy = tuner.run()

        assert validation_accuracy > self.pipeline.cache['pt_accuracy'] * 0.9, \
            "Noise adaptation step failed to retain validation accuracy."

        self.pipeline.cache.add("na_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("na_accuracy", validation_accuracy)

        self.pipeline.cache.remove("pretrained_model")

        
