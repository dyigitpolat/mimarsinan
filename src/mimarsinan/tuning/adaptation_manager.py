from mimarsinan.models.layers import *
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

import torch.nn as nn

class AdaptationManager(nn.Module):
    def __init__(self):
        super(AdaptationManager, self).__init__()
        
        self.clamp_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0
        self.scale_rate = 0.0

        self.noise_rate = 0.0

    def update_activation(self, pipeline_config, perceptron):
        use_ttfs = pipeline_config.get("spiking_mode", "rate") in ("ttfs", "ttfs_quantized")
        decorators = [
            self.get_rate_adjusted_clamp_decorator(perceptron),
            self.get_rate_adjusted_quantization_decorator(pipeline_config, perceptron),
        ]
        if not use_ttfs:
            decorators.append(self.get_shift_decorator(pipeline_config, perceptron))

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))
        
        target_noise_amount = (1.0 / (pipeline_config['target_tq'] * 2.5))
        perceptron.set_regularization(
            NoisyDropout(torch.tensor(0.0), self.noise_rate, target_noise_amount * perceptron.activation_scale)
        )

        # perceptron.set_scaler(
        #     TransformedActivation(
        #         nn.Identity(),
        #         [RateAdjustedDecorator(
        #             self.scale_rate, 
        #             ScaleDecorator(1.0 / perceptron.scale_factor), 
        #             MixAdjustmentStrategy())]
        #     )
        # )
        
    def get_rate_adjusted_clamp_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.clamp_rate, 
            ClampDecorator(torch.tensor(0.0), perceptron.activation_scale), 
            NestedAdjustmentStrategy([RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]))

    # def get_rate_adjusted_scale_decorator(self, perceptron):
    #     return RateAdjustedDecorator(
    #         self.scale_rate, 
    #         ScaleDecorator(1.0 / perceptron.activation_scale), 
    #         RandomMaskAdjustmentStrategy())
    
    def get_shift_decorator(self, pipeline_config, perceptron):
        shift_amount = calculate_activation_shift(pipeline_config["target_tq"], perceptron.activation_scale) * self.shift_rate
        return ShiftDecorator(shift_amount)
    
    def get_rate_adjusted_quantization_decorator(self, pipeline_config, perceptron):
        # For TTFS: shift the other way — use shift_back = -shift so ReLU sees (x + shift) → staircase(ReLU(x + shift)).
        # For rate: shift_back = -shift * shift_rate to undo the outer shift.
        use_ttfs = pipeline_config.get("spiking_mode", "rate") in ("ttfs", "ttfs_quantized")
        shift = calculate_activation_shift(
            pipeline_config["target_tq"], perceptron.activation_scale
        )
        if use_ttfs:
            shift_back_amount = -shift  # ReLU gets (x - (-shift)) = (x + shift)
        else:
            shift_back_amount = -shift * self.shift_rate

        return RateAdjustedDecorator(
            self.quantization_rate, 
            NestedDecoration(
                [ShiftDecorator(shift_back_amount), 
                QuantizeDecorator(torch.tensor(pipeline_config["target_tq"]), perceptron.activation_scale)]),
            NestedAdjustmentStrategy([RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]))
            #RandomMaskAdjustmentStrategy())
    
    # def get_rate_adjusted_scale_decorator(self, perceptron):
    #     return RateAdjustedDecorator(
    #         self.scale_rate, 
    #         AnyDecorator(perceptron.scaler), 
    #         MixAdjustmentStrategy())
    