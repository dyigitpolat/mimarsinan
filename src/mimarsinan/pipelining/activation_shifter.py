from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import ShiftedActivation, ClampedReLU
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
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.aq_loss, clip_and_decay_param)
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"Shift accuracy: {value}")
            pipeline.reporter.report(key, value)

        self.report = report
        self.trainer.report_function = self.report
        
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
        for i in range(1, self.cycles + 1):
            print(f"  Activation shift: 0.5 * ({i}/{self.cycles}) / {self.target_tq}")
            shift_amount = 0.5 * (i / self.cycles) / self.target_tq
            self.shift_activation(shift_amount)
            self.trainer.train_n_epochs(self.lr / 20, 2)
        return self.trainer.validate_train()