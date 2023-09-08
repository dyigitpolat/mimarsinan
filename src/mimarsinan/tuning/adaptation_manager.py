from mimarsinan.models.layers import *

import torch.nn as nn

class AdaptationManager(nn.Module):
    def __init__(self, pipeline_config):
        super(AdaptationManager, self).__init__()
        
        self.clamp_rate = 0.0
        self.scale_rate = 0.0
        self.quantization_rate = 0.0

        self.shift_amount = 0.0
        
        self.Tq = pipeline_config['target_tq']

    def update_activation(self, perceptron):
        decorators = [
            self.get_rate_adjusted_clamp_decorator(),
            self.get_rate_adjusted_scale_decorator(),
            self.get_shift_decorator(),
            self.get_rate_adjusted_quantization_decorator()]

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

    def get_rate_adjusted_clamp_decorator(self):
        return RateAdjustedDecorator(
            self.clamp_rate, ClampDecorator(0.0, 1.0), RandomMaskAdjustmentStrategy())

    def get_rate_adjusted_scale_decorator(self):
        return RateAdjustedDecorator(
            self.scale_rate, ScaleDecorator(1.0), RandomMaskAdjustmentStrategy())
    
    def get_shift_decorator(self):
        return ShiftDecorator(self.shift_amount)
    
    def get_rate_adjusted_quantization_decorator(self):
        return RateAdjustedDecorator(
            self.quantization_rate, QuantizeDecorator(self.Tq), RandomMaskAdjustmentStrategy())