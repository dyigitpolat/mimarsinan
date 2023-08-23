from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["sa_model"]
        promises = ["na_model"]
        clears = ["sa_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = NoiseTuner(
            self.pipeline,
            model = self.pipeline.cache['sa_model'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.pipeline.cache.add("na_model", self.tuner.model, 'torch_model')