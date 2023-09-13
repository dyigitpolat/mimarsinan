from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        adaptation_manager = self.get_entry('adaptation_manager')
        for perceptron in model.get_perceptrons():
            shift_amount = 0.5 / (self.pipeline.config['target_tq'] * perceptron.base_threshold)

            adaptation_manager.shift_rate = 1.0
            adaptation_manager.update_activation(perceptron)

            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.bias.data += shift_amount
            else:
                perceptron.normalization.bias.data += shift_amount
        
        self.trainer.train_until_target_accuracy(
            self.pipeline.config['lr'] / 20, 
            max_epochs=2, 
            target_accuracy=self.pipeline.get_target_metric())
        
        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, 'torch_model')
