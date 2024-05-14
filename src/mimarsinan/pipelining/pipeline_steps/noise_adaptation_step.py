from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner 

class NoiseAdaptationStep(PipelineStep):
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
        self.tuner = NoiseTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'] * 1e-3, 
            adaptation_manager = self.get_entry('adaptation_manager'))
        self.tuner.run()

        self.update_entry("adaptation_manager", self.tuner.adaptation_manager, "pickle")
        self.update_entry("model", self.tuner.model, 'torch_model')