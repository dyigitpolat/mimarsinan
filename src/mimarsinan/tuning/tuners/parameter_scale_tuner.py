from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch

import copy 

class ParameterScaleTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.rate = 0.0

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_perceptron_transform(self):
        return lambda perceptron: copy.deepcopy(perceptron)
    
    def _get_new_perceptron_transform(self):
        def transform(perceptron):
            perceptron = copy.deepcopy(perceptron)

            w = PerceptronTransformer().get_effective_weight(perceptron)
            b = PerceptronTransformer().get_effective_bias(perceptron)

            p_max = torch.max(w.abs().max(), b.abs().max()).item()
            scale = 1.0 / p_max

            adjusted_scale = scale * self.rate + 1.0 * (1.0 - self.rate)

            if not isinstance(perceptron.normalization, nn.Identity):
                perceptron.normalization.running_mean *= adjusted_scale
                perceptron.normalization.running_var *= (adjusted_scale ** 2)

            PerceptronTransformer().apply_effective_parameter_transform(perceptron, lambda x: x * adjusted_scale)
            

            return perceptron.to(self.device)
        
        return transform
    
    def _update_and_evaluate(self, rate):
        self.rate = rate
        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        return self.trainer.validate()
