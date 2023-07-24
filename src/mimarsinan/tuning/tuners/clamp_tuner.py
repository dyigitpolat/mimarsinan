from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.models.layers import ClampedReLU_Parametric, ClampedReLU, ActivationStats

import torch.nn as nn
class ClampTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.base_activations = []

        for perceptron in model.get_perceptrons():
            self.base_activations.append(perceptron.activation)

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x
    
    def _calculate_base_thresholds(self, model):
        for perceptron in model.get_perceptrons():
            perceptron.set_activation(ActivationStats(perceptron.activation))

        self.trainer.validate()

        for perceptron in model.get_perceptrons():
            perceptron.base_threshold = perceptron.activation.max.item()
            print(perceptron.base_threshold)

    def _update_and_evaluate(self, rate):
        for perceptron, activation in zip(self.model.get_perceptrons(), self.base_activations):
            perceptron.set_activation(ClampedReLU_Parametric(rate, activation))

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        self._calculate_base_thresholds(self.model)
        
        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(ClampedReLU(0.0, perceptron.base_threshold))

        self.trainer.train_until_target_accuracy(self._find_lr(), self.epochs, self._get_target())

        return self.trainer.validate()
