from mimarsinan.models.layers import *
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

import torch.nn as nn

class AdaptationManager(nn.Module):
    def __init__(self):
        super(AdaptationManager, self).__init__()
        
        self.clamp_rate = 0.0
        self.scale_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0

    def update_activation(self, pipeline_config, perceptron):
        decorators = [
            self.get_rate_adjusted_quantization_decorator(pipeline_config, perceptron),
            self.get_shift_decorator(pipeline_config, perceptron),
            self.get_rate_adjusted_clamp_decorator(perceptron),
            self.get_rate_adjusted_scale_decorator(perceptron)]

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

    def get_rate_adjusted_clamp_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.clamp_rate, 
            ClampDecorator(0.0, perceptron.base_threshold), 
            RandomMaskAdjustmentStrategy())

    def get_rate_adjusted_scale_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.scale_rate, 
            ScaleDecorator(1.0 / perceptron.base_threshold), 
            RandomMaskAdjustmentStrategy())
    
    def get_shift_decorator(self, pipeline_config, perceptron):
        shift_amount = calculate_activation_shift(pipeline_config["target_tq"], perceptron.base_threshold) * self.shift_rate
        return ShiftDecorator(shift_amount)
    
    def get_rate_adjusted_quantization_decorator(self, pipeline_config, perceptron):
        shift_back_amount = -calculate_activation_shift(pipeline_config["target_tq"], perceptron.base_threshold) * self.shift_rate

        return RateAdjustedDecorator(
            self.quantization_rate, 
            NestedDecoration(
                [ShiftDecorator(shift_back_amount), 
                QuantizeDecorator(pipeline_config["target_tq"], perceptron.base_threshold)]),
            RandomMaskAdjustmentStrategy())