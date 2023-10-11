from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch
class ScaleTuner(BasicTuner):
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
        self.previous_scales = [
            perceptron.activation_scale for perceptron in self.model.get_perceptrons()]

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x
    
    def _update_parameters(self, perceptron, prev_act_scale, new_act_scale):
        scale = new_act_scale / prev_act_scale
        PerceptronTransformer().apply_effective_parameter_transform(perceptron, lambda x: x * scale)
        
        if not isinstance(perceptron.normalization, nn.Identity):
            perceptron.normalization.running_mean *= scale
            perceptron.normalization.running_var *= (scale ** 2)

    def _update_and_evaluate(self, rate):
        for perceptron, prev_scale in zip(self.model.get_perceptrons(), self.previous_scales):
            target_scale = 1.0
            new_scale = target_scale * rate + prev_scale * (1.0 - rate)
            perceptron.activation_scale = new_scale

            self._update_parameters(perceptron, prev_scale, perceptron.activation_scale)
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        return self.trainer.validate()
