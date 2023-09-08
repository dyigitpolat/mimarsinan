from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.scale_tuner import ScaleTuner 

class ScaleAdaptationStep(PipelineStep):
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
        self.tuner = ScaleTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'],
            adaptation_manager = self.get_entry('adaptation_manager'))
        self.tuner.run()

        self.update_entry("adaptation_manager", self.tuner.adaptation_manager, "pickle")
        self.update_entry("model", self.tuner.model, 'torch_model')

        
