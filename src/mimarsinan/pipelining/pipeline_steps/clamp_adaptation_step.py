from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner 

class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["pretrained_model"]
        promises = ["ca_model"]
        clears = ["pretrained_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('pretrained_model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.add_entry("ca_model", self.tuner.model, 'torch_model')

        
