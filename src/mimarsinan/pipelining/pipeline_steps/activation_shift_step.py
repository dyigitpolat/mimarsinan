from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ShiftedActivation

import torch.nn as nn

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["na_model"]
        promises = ["shifted_activation_model"]
        clears = ["na_model"]
        super().__init__(requires, promises, clears, pipeline)
        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.pipeline.cache["na_model"]

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider, 
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
        
        self.pipeline.cache.add("shifted_activation_model", model, 'torch_model')
        self.pipeline.cache.remove("na_model")
