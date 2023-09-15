from mimarsinan.transformations.transformation_utils import *
from mimarsinan.mapping.mapping_utils import get_fused_weights
from mimarsinan.models.layers import FrozenStatsNormalization

import torch.nn as nn
import torch
import copy

class NormalizationAwarePerceptronQuantization:
    def __init__(self, bits, device):
        self.device = device
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
    
    def transform(self, perceptron):
        if isinstance(perceptron.normalization, nn.Identity):
            out_perceptron = self._handle_no_normalization(perceptron)
            self._verify_fuse_quantization(out_perceptron)
            return out_perceptron

        assert isinstance(perceptron.normalization, FrozenStatsNormalization)

        u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
        fused_w, fused_b = get_fused_weights(perceptron.layer, perceptron.normalization)
        out_perceptron = self._get_new_perceptron(perceptron, fused_w, fused_b, u, beta, mean)
            
        self._verify_fuse_quantization(out_perceptron)
        return out_perceptron.to(self.device)
    
    def _handle_no_normalization(self, perceptron):
        out_perceptron = copy.deepcopy(perceptron)
        if perceptron.layer.bias is not None:
            q_w, q_b = self._get_quantized_params(perceptron.layer.weight.data, perceptron.layer.bias.data)
            out_perceptron.layer.weight.data = q_w
            out_perceptron.layer.bias.data = q_b
        else:
            q_w = self._get_quantized_param(perceptron.layer.weight.data)
            out_perceptron.layer.weight.data = q_w
        return out_perceptron

    def _get_u_beta_mean(self, bn_layer):
        bn = bn_layer
        gamma = bn.weight.data
        beta = bn.bias.data
        var = bn.running_var.data
        mean = bn.running_mean.data
        u = gamma / torch.sqrt(var + bn.eps)

        return u, beta, mean

    def _get_new_perceptron(self, perceptron, fused_w, fused_b, u, beta, mean):
        new_w, new_b = self._get_transformed_parameters(fused_w, fused_b, u, beta, mean)
        out_perceptron = copy.deepcopy(perceptron).to(self.device)

        out_perceptron.layer.weight.data = new_w

        if out_perceptron.layer.bias is not None:
            out_perceptron.layer.bias.data = new_b
        else:
            out_perceptron.layer.bias = torch.nn.Parameter(new_b)

        return out_perceptron
    
    def _verify_fuse_quantization(self, perceptron):
        perceptron.to(self.device)
        if isinstance(perceptron.normalization, nn.Identity):
            _fused_w, _fused_b = perceptron.layer.weight.data, perceptron.layer.bias.data
        else:
            _fused_w, _fused_b = get_fused_weights(perceptron.layer, perceptron.normalization)

        w_max = torch.max(torch.abs(_fused_w))
        b_max = torch.max(torch.abs(_fused_b))
        param_scale = self.q_max / max(w_max, b_max)

        assert torch.allclose(
            _fused_w * param_scale, torch.round(_fused_w * param_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_w * param_scale}"

        assert torch.allclose(
            _fused_b * param_scale, torch.round(_fused_b * param_scale),
            atol=1e-3, rtol=1e-3), f"{_fused_b * param_scale}"
        
    def _get_quantized_param(self, param):
        param_max = torch.max(torch.abs(param))
        param_scale = self.q_max / param_max
        q_param = torch.round(param * param_scale) / param_scale
        return q_param
        
    def _get_quantized_params(self, w, b):
        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        param_scale = self.q_max / max(w_max, b_max)
        q_w = torch.round(w * param_scale) / param_scale
        q_b = torch.round(b * param_scale) / param_scale
        return q_w, q_b
        
    def _get_transformed_parameters(self, fused_w, fused_b, u, beta, mean):
        q_fused_w, q_fused_b = self._get_quantized_params(fused_w, fused_b)
        new_w = q_fused_w / u.unsqueeze(1)
        new_b = ((q_fused_b - beta) / u) + mean
        return new_w, new_b
