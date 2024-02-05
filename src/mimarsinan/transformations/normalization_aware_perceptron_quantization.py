from mimarsinan.transformations.transformation_utils import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch
import copy

class NormalizationAwarePerceptronQuantization:
    def __init__(self, bits, device, rate=1.0):
        self.device = device
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.rate = rate
    
    def transform(self, perceptron):
        w = PerceptronTransformer().get_effective_weight(perceptron)
        b = PerceptronTransformer().get_effective_bias(perceptron)
        
        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        p_max = max(w_max, b_max)

        scale = self.q_max * (1.0 / p_max)

        # do magick here:
        # scale_09 = scale * 0.99
        # scale_09_r = max(torch.floor(scale_09), 1.0)
        # scale = scale_09_r / 0.99 # end magick

        scale = torch.round(scale)
        perceptron.set_parameter_scale(scale)
        def quantize_param(param):
            scaled_param = param * scale
            quantized_param = torch.minimum(torch.round(scaled_param), torch.tensor(self.q_max))
            quantized_param = torch.maximum(quantized_param, torch.tensor(self.q_min))
            rescaled_param = quantized_param / (scale)
            return rescaled_param
        
        PerceptronTransformer().apply_effective_parameter_transform(perceptron, quantize_param) 

    def _verify_fuse_quantization(self, perceptron):
        perceptron.to(self.device)

        _fused_w = PerceptronTransformer().get_effective_weight(perceptron)
        _fused_b = PerceptronTransformer().get_effective_bias(perceptron)
        
        w_max = torch.max(torch.abs(_fused_w))
        b_max = torch.max(torch.abs(_fused_b))
        p_max = max(w_max, b_max)

        natural_scale = 1.0
        target_scale = 1.0 / p_max
        adjusted_scale = target_scale * self.rate + natural_scale * (1.0 - self.rate)
        adjusted_scale_correction = 1.0 * self.rate + adjusted_scale * (1.0 - self.rate)

        q_scale = self.q_max * adjusted_scale_correction

        assert torch.allclose(
            _fused_w * q_scale, torch.round(_fused_w * q_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_w * q_scale}"

        assert torch.allclose(
            _fused_b * q_scale, torch.round(_fused_b * q_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_b * q_scale}"
