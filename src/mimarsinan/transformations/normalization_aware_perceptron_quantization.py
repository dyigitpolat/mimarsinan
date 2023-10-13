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
        out_perceptron = copy.deepcopy(perceptron)
        w = PerceptronTransformer().get_effective_weight(perceptron)
        b = PerceptronTransformer().get_effective_bias(perceptron)
        
        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        p_max = max(w_max, b_max)

        natural_scale = 1.0 / p_max
        target_scale = 1.0
        adjusted_scale = target_scale * self.rate + natural_scale * (1.0 - self.rate)
        adjusted_clamp = 1.0 * self.rate + p_max * (1.0 - self.rate)
        def quantize_param(param):
            scaled_param = param * adjusted_scale
            clipped_param = torch.clamp(scaled_param, -adjusted_clamp, adjusted_clamp)
            return torch.round(clipped_param * self.q_max) / (self.q_max)
        
        PerceptronTransformer().apply_effective_parameter_transform(out_perceptron, quantize_param) 

        self._verify_fuse_quantization(out_perceptron)
        return out_perceptron.to(self.device)

    def _verify_fuse_quantization(self, perceptron):
        perceptron.to(self.device)

        _fused_w = PerceptronTransformer().get_effective_weight(perceptron)
        _fused_b = PerceptronTransformer().get_effective_bias(perceptron)
        
        q_scale = self.q_max

        assert torch.allclose(
            _fused_w * q_scale, torch.round(_fused_w * q_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_w * q_scale}"

        assert torch.allclose(
            _fused_b * q_scale, torch.round(_fused_b * q_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_b * q_scale}"
