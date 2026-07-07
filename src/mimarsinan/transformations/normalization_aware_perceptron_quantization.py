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
        transformer = PerceptronTransformer()
        with transformer.deferred_finite_checks(perceptron):
            self._transform(perceptron, transformer)

    def _transform(self, perceptron, transformer):
        # The scale must never be set by functionally-unobservable bias mass:
        # a constant-OFF channel's outlier bias starves the shared weight grid
        # (the t01_19/t0_03 WQ-entry crater). Idempotent, function-preserving.
        clip_off_saturated_effective_bias(perceptron, transformer=transformer)
        w = transformer.get_effective_weight(perceptron)
        b = transformer.get_effective_bias(perceptron)

        # All-tensor scale chain: this runs per perceptron per optimizer step,
        # and a python max()/CPU-tensor bound forces a GPU sync per call.
        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        if torch.is_tensor(b_max) and b_max.device != w_max.device:
            b_max = b_max.to(w_max.device)
        p_max = torch.clamp(torch.maximum(w_max, b_max), min=1e-12)

        scale = self.q_max * (1.0 / p_max)

        perceptron.set_parameter_scale(scale)

        rate = float(self.rate)
        q_min = float(self.q_min)
        q_max = float(self.q_max)

        def quantize_param(param):
            scaled = param * scale
            quantized = torch.round(scaled).clamp_(min=q_min, max=q_max)
            rescaled = quantized / scale
            if rate >= 1.0:
                return rescaled
            if rate <= 0.0:
                return param
            return rate * rescaled + (1.0 - rate) * param

        transformer.apply_effective_parameter_transform(perceptron, quantize_param)
