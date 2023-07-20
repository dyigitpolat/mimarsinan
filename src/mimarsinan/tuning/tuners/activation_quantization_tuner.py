from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.models.layers import CQ_Activation_Parametric, CQ_Activation

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
        self.base_activation = model.activation

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return SequentialTransform([
            lambda p: torch.clamp(p, -1, 1) ])

    def _update_and_evaluate(self, rate):
        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(CQ_Activation_Parametric(
                self.target_tq, 
                rate, 
                self.base_activation, 
                perceptron.base_threshold))
        
        self.trainer.train_n_epochs(self._find_lr() / 2, 1)
        return self.trainer.validate()

    def run(self):
        super().run()
        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(
                CQ_Activation(self.target_tq, perceptron.base_threshold))
        
        self.trainer.weight_transformation = self._get_new_parameter_transform()
        self.trainer.train_until_target_accuracy(self._find_lr() / 2, self.epochs, self._get_target())

        return self.trainer.validate()
