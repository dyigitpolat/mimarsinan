from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

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

            if isinstance(perceptron.normalization, nn.Identity):
                w, b = perceptron.layer.weight.data, perceptron.layer.bias.data
            else:
                w_, b_ = perceptron.layer.weight.data, perceptron.layer.bias.data
                u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
                b = (b_ - mean) * u + beta
                w = w_ * u.unsqueeze(1)

            p_max = torch.max(w.abs().max(), b.abs().max()).item()
            scale = 1.0 / p_max
            adjusted_scale = scale * self.rate + 1.0 * (1.0 - self.rate)

            if isinstance(perceptron.normalization, nn.Identity):
                perceptron.layer.weight.data *= adjusted_scale
                perceptron.layer.bias.data *= adjusted_scale
            else:
                perceptron.layer.weight.data *= adjusted_scale
                b = perceptron.layer.bias.data
                perceptron.layer.bias.data = \
                    (adjusted_scale * ((b-mean)*u + beta) + mean * u - beta) / u
                
                perceptron.normalization.running_var *= scale 
                
            

            return perceptron.to(self.device)
        
        return transform
    
    def _get_u_beta_mean(self, bn_layer):
        bn = bn_layer
        gamma = bn.weight.data
        beta = bn.bias.data
        var = bn.running_var.data.to(gamma.device)
        mean = bn.running_mean.data.to(gamma.device)

        u = gamma / torch.sqrt(var + bn.eps)

        return u, beta, mean
    
    def _update_and_evaluate(self, rate):
        self.rate = rate
        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        return self.trainer.validate()
