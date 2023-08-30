from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ShiftedActivation

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = []
        promises = []
        updates = ["model"]
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
        
        
        for perceptron in model.get_perceptrons():
            shift_amount = 0.5 / (self.pipeline.config['target_tq'] * perceptron.base_threshold)

            perceptron.set_activation(
                ShiftedActivation(perceptron.activation, shift_amount))

            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.bias.data += shift_amount
            else:
                perceptron.normalization.bias.data += shift_amount
        
        self.trainer.train_until_target_accuracy(
            self.pipeline.config['lr'] / 20, 
            max_epochs=2, 
            target_accuracy=self.pipeline.get_target_metric())
        
        self.update_entry("model", model, 'torch_model')
