from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["ca_model"]
        promises = ["na_model"]
        clears = ["ca_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = NoiseTuner(
            self.pipeline,
            model = self.get_entry('ca_model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.add_entry("na_model", self.tuner.model, 'torch_model')