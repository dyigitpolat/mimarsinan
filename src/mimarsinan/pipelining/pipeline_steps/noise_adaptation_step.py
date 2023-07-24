from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["sa_model", "sa_accuracy"]
        promises = ["na_model", "na_accuracy"]
        clears = ["sa_model"]
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        tuner = NoiseTuner(
            self.pipeline,
            model = self.pipeline.cache['sa_model'],
            target_accuracy = self.pipeline.cache['sa_accuracy'] * 0.99,
            lr = self.pipeline.config['lr'])
        
        accuracy = tuner.run()

        assert accuracy > self.pipeline.cache['sa_accuracy'] * 0.9, \
            "Noise adaptation step failed to retain validation accuracy."

        self.pipeline.cache.add("na_model", tuner.model, 'torch_model')
        self.pipeline.cache.add("na_accuracy", accuracy)

        self.pipeline.cache.remove("sa_model")