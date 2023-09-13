from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner 
from mimarsinan.tuning.adaptation_manager import AdaptationManager

class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["adaptation_manager"]
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        adaptation_manager = AdaptationManager()

        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'],
            adaptation_manager = adaptation_manager)
        self.tuner.run()

        self.add_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", self.tuner.model, 'torch_model')

        
