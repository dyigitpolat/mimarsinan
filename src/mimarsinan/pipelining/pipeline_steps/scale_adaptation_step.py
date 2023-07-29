from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.scale_tuner import ScaleTuner 

class ScaleAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["ca_model"]
        promises = ["sa_model"]
        clears = ["ca_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.tuner = None
    
    def validate(self):
        return self.tuner.validate()

    def process(self):
        self.tuner = ScaleTuner(
            self.pipeline,
            model = self.pipeline.cache['ca_model'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        self.pipeline.cache.add("sa_model", self.tuner.model, 'torch_model')
        self.pipeline.cache.remove("ca_model")

        
