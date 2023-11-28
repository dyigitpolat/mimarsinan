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
        self.original_scales = [
            perceptron.activation_scale for perceptron in self.model.get_perceptrons()]
        self.previous_rates = [ 0.0 for _ in self.original_scales ]

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x
    
    def _update_parameters(self, perceptron, original_scale, prev_rate, rate):
        # prev_scale = original_scale * (1.0 - prev_rate) + 1.0 * prev_rate
        new_scale = original_scale * (1.0 - rate) + 1.0 * rate
        # scale_factor = new_scale / prev_scale
        
        # if not isinstance(perceptron.normalization, nn.Identity):
        #     perceptron.normalization.running_mean *= scale_factor
        #     perceptron.normalization.running_var *= (scale_factor ** 2)
        
        perceptron.set_activation_scale(max(1.0, new_scale))
        #PerceptronTransformer().apply_effective_parameter_transform(perceptron, lambda x: x * scale_factor)

    def _update_and_evaluate(self, rate):
        for idx, perceptron in enumerate(self.model.get_perceptrons()):
            prev_rate, original_scale = self.previous_rates[idx], self.original_scales[idx]

            self._update_parameters(perceptron, original_scale, prev_rate, rate)
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

            self.previous_rates[idx] = rate

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()
        return self.trainer.validate()
