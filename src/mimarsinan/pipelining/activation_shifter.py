from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import ShiftedActivation, ClampedReLU

import torch.nn as nn
class ActivationShifter:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy * 0.99

        # Model
        self.model = pipeline.model

        # Trainer
        self.trainer = BasicTrainer(
            self.model, 
            pipeline.device,
            pipeline.data_provider,
            pipeline.loss)
        self.trainer.report_function = pipeline.reporter.report
        
        # Training
        self.lr = pipeline.lr
        self.cycles = 3

    def shift_activation(self, shift_amount):
        self.model.set_activation(
            ShiftedActivation(ClampedReLU(), shift_amount))
        
        for perceptron in self.model.get_perceptrons():

            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.bias.data += shift_amount
            else:
                perceptron.normalization.bias.data += shift_amount

        
    def run(self):
        shift_amount = 0.5 / self.target_tq
        self.shift_activation(shift_amount)
        self.trainer.train_until_target_accuracy(self.lr / 20, 2, self.target_accuracy)
        return self.trainer.validate()