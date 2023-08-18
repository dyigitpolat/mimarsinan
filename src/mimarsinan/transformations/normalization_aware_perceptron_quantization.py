from mimarsinan.transformations.transformation_utils import *
from mimarsinan.mapping.mapping_utils import get_fused_weights

import torch.nn as nn
import torch
import copy

class NormalizationAwarePerceptronQuantization:
    def __init__(self, bits):
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1

    def _get_transform_rate(self, param):
        param_scale = self.q_max / torch.max(torch.abs(param))
        quantized_param = torch.round(param * param_scale) / param_scale
        return quantized_param / param
    
    def transform(self, perceptron):
        if isinstance(perceptron.layer, nn.BatchNorm1d):
            fused_weight, fused_bias = get_fused_weights(perceptron.layer, perceptron.normalization)
        else:
            fused_weight, fused_bias = perceptron.layer.weight, perceptron.layer.bias
        
        weight_transform_rate = self._get_transform_rate(fused_weight)
        new_weight = perceptron.layer.weight * weight_transform_rate

        if fused_bias is not None:
            bias_transform_rate = self._get_transform_rate(fused_bias)
            new_bias = perceptron.layer.bias * bias_transform_rate
        else:
            new_bias = None

        out_perceptron = copy.deepcopy(perceptron)
        out_perceptron.layer.weight.data[:] = new_weight.data[:]

        if new_bias is not None:
            out_perceptron.layer.bias.data[:] = new_bias.data[:]

        return out_perceptron
