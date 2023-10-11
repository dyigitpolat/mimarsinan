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

        scale = self.q_max / p_max
        def quantize_param(param):
            return torch.round(param * scale) / scale

        PerceptronTransformer().apply_effective_parameter_transform(out_perceptron, quantize_param) 

        self._verify_fuse_quantization(out_perceptron)
        return out_perceptron.to(self.device)

    def _verify_fuse_quantization(self, perceptron):
        perceptron.to(self.device)

        _fused_w = PerceptronTransformer().get_effective_weight(perceptron)
        _fused_b = PerceptronTransformer().get_effective_bias(perceptron)

        w_max = torch.max(torch.abs(_fused_w))
        b_max = torch.max(torch.abs(_fused_b))
        p_max = max(w_max, b_max)

        param_scale = self.q_max / p_max

        assert torch.allclose(
            _fused_w * param_scale, torch.round(_fused_w * param_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_w * param_scale}"

        assert torch.allclose(
            _fused_b * param_scale, torch.round(_fused_b * param_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_b * param_scale}"
