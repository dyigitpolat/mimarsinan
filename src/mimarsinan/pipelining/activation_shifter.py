from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import ShiftedActivation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

import torch.nn as nn
class ActivationShifter:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq

        # Model
        self.model = pipeline.model

        # Trainer
        self.trainer = BasicTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.wq_loss)
        
    def run(self):
        shift_amount = 0.5 / self.target_tq

        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(
                ShiftedActivation(perceptron.activation, shift_amount))

            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.bias.data += shift_amount
            else:
                perceptron.normalization.bias.data += shift_amount

        return self.trainer.validate_train()