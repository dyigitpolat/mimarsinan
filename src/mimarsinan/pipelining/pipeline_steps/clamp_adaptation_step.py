from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.training_utilities import BasicClassificationLoss
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None
        self.tuner = None

    def validate(self):
        return self.tuner.validate()
        return self.trainer.validate()
    

    def process(self):
        model = self.get_entry('model')
        adaptation_manager = self.get_entry("adaptation_manager")
        
        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'] * 1e-3,
            adaptation_manager = adaptation_manager,
            activation_scales = self.get_entry("activation_scales"))
        self.tuner.run()

        # Trainer
        # self.trainer = BasicTrainer(
        #     model, 
        #     self.pipeline.config['device'], 
        #     DataLoaderFactory(self.pipeline.data_provider_factory),
        #     self.pipeline.loss)
        # self.trainer.report_function = self.pipeline.reporter.report

        # adaptation_manager.clamp_rate = 1.0
        # scales = self.get_entry("activation_scales")
        # for idx, perceptron in enumerate(model.get_perceptrons()):
        #     perceptron.set_activation_scale(scales[idx])
        #     adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')
        