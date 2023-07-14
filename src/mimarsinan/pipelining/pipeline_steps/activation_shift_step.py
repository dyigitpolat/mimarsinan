from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ShiftedActivation

import torch.nn as nn

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["na_model", "na_accuracy"]
        promises = ["shifted_activation_model", "as_accuracy"]
        clears = ["na_model"]
        super().__init__(requires, promises, clears, pipeline)


    def process(self):
        model = self.pipeline.cache["na_model"]

        trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider, 
            self.pipeline.loss)
        trainer.report_function = self.pipeline.reporter.report


        shift_amount = 0.5 / self.pipeline.config['target_tq']
        
        model.set_activation(
            ShiftedActivation(model.activation, shift_amount))
        
        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.bias.data += shift_amount
            else:
                perceptron.normalization.bias.data += shift_amount
        
        validation_accuracy = trainer.train_until_target_accuracy(
            self.pipeline.config['lr'] / 20, 
            max_epochs=2, 
            target_accuracy=self.pipeline.cache['na_accuracy'])
        
        assert validation_accuracy > self.pipeline.cache['na_accuracy'] * 0.9, \
            "Activation shift step failed to retain validation accuracy."
        
        self.pipeline.cache.add("shifted_activation_model", model, 'torch_model')
        self.pipeline.cache.add("as_accuracy", validation_accuracy)
        
        self.pipeline.cache.remove("na_model")
