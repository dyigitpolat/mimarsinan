from mimarsinan.transformations.transformation_utils import *

from mimarsinan.transformations.perceptron.bias_saturation import (
    clip_off_saturated_effective_bias,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

import torch

class NormalizationAwarePerceptronQuantization:
    """Normalization-aware per-perceptron weight quantization.

    Interpolates in weight-value space between the FP and fully-quantized
    effective weights (``rate=0`` identity, ``rate=1`` == legacy full quant).
    ``parameter_scale`` is always the full-range (``rate=1``) scale.
    """

    def __init__(self, bits, device, rate=1.0):
        self.device = device
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.rate = rate

    def transform(self, perceptron):
        # The scale must never be set by functionally-unobservable bias mass:
        # a constant-OFF channel's outlier bias starves the shared weight grid
        # (the t01_19/t0_03 WQ-entry crater). Idempotent, function-preserving.
        clip_off_saturated_effective_bias(perceptron)
        w = PerceptronTransformer().get_effective_weight(perceptron)
        b = PerceptronTransformer().get_effective_bias(perceptron)

        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        p_max = max(w_max, b_max)

        p_max = max(p_max, 1e-12)

        scale = self.q_max * (1.0 / p_max)

        perceptron.set_parameter_scale(scale)

        rate = float(self.rate)
        q_min_t = torch.tensor(self.q_min)
        q_max_t = torch.tensor(self.q_max)

        def quantize_param(param):
            scaled = param * scale
            quantized = torch.round(scaled)
            quantized = torch.minimum(quantized, q_max_t)
            quantized = torch.maximum(quantized, q_min_t)
            rescaled = quantized / scale
            if rate >= 1.0:
                return rescaled
            if rate <= 0.0:
                return param
            return rate * rescaled + (1.0 - rate) * param

        PerceptronTransformer().apply_effective_parameter_transform(perceptron, quantize_param)
