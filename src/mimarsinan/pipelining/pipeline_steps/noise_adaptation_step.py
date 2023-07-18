from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["ca_model", "ca_accuracy"]
        promises = ["na_model", "na_accuracy"]
        clears = ["ca_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = NoiseTuner(
            self.pipeline,
            model = self.pipeline.cache['ca_model'],
            target_accuracy = self.pipeline.cache['ca_accuracy'] * 0.99,
            lr = self.pipeline.config['lr'])
        
        accuracy = tuner.run()

        assert accuracy > self.pipeline.cache['ca_accuracy'] * 0.9, \
            "Noise adaptation step failed to retain validation accuracy."

        self.pipeline.cache.add("na_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("na_accuracy", accuracy)

        self.pipeline.cache.remove("ca_model")

        
