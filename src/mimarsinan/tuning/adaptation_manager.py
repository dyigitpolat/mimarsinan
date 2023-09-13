from mimarsinan.models.layers import *

import torch.nn as nn

class AdaptationManager(nn.Module):
    def __init__(self, pipeline_config):
        super(AdaptationManager, self).__init__()
        
        self.clamp_rate = 0.0
        self.scale_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0
        
        self.Tq = pipeline_config['target_tq']

    def update_activation(self, perceptron):
        decorators = [
            self.get_rate_adjusted_quantization_decorator(),
            self.get_shift_decorator(perceptron),
            self.get_rate_adjusted_clamp_decorator(perceptron),
            self.get_rate_adjusted_scale_decorator(perceptron)]

        print(perceptron.base_threshold)
        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

    def get_rate_adjusted_clamp_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.clamp_rate, ClampDecorator(0.0, perceptron.base_threshold), RandomMaskAdjustmentStrategy())

    def get_rate_adjusted_scale_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.scale_rate, ScaleDecorator(1.0 / perceptron.base_threshold), RandomMaskAdjustmentStrategy())
    
    def get_shift_decorator(self, perceptron):
        shift_amount = 0.5 / (self.Tq * perceptron.base_threshold)
        return ShiftDecorator(shift_amount * self.shift_rate)
    
    def get_rate_adjusted_quantization_decorator(self):
        return RateAdjustedDecorator(
            self.quantization_rate, QuantizeDecorator(self.Tq), RandomMaskAdjustmentStrategy())