from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.models.layers import CQ_Activation_Parametric, CQ_Activation, ScaleActivation, ShiftedActivation

import torch

class ActivationQuantizationTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_tq, 
                 target_accuracy, 
                 lr):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.target_tq = target_tq
        self.base_activations = []
        for perceptron in model.get_perceptrons():
            shifted_activation = perceptron.activation

            assert isinstance(shifted_activation, ShiftedActivation)
            base_activation = shifted_activation.activation

            self.base_activations.append(
                ShiftedActivation(
                    base_activation,
                    shifted_activation.shift))

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x

    def _update_and_evaluate(self, rate):
        for perceptron, base_activation in zip(self.model.get_perceptrons(), self.base_activations):
            perceptron.set_activation(
                CQ_Activation_Parametric(
                    self.target_tq, 
                    rate, 
                    base_activation, 
                    perceptron.base_threshold))
        
        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(
                    CQ_Activation(self.target_tq, perceptron.base_threshold))
        
        self.trainer.weight_transformation = self._get_new_parameter_transform()
        self.trainer.train_until_target_accuracy(self._find_lr() / 2, self.epochs, self._get_target())

        return self.trainer.validate()
