from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner 

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

class ClampAdaptationStep(PipelineStep):
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
        adaptation_manager = self.get_entry("adaptation_manager")

        validator = BasicTrainer(
            self.get_entry("model"), 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        
        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'],
            adaptation_manager = adaptation_manager)
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", self.tuner.model, 'torch_model')
        