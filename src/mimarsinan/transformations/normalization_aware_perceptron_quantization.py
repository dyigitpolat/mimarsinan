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

    ``two_scale=True`` derives the weight grid from ``max|w|`` alone and puts
    the effective bias on its OWN ``q_max`` register range: the bias grid is
    integer-ratio-snapped to the weight grid (``bias_scale = parameter_scale/r``),
    so the quantized bias stays an exact integer on the weight-grid lattice —
    the torch<->chip parity contract (``wq_cascade_crater_repair.md`` §4.3/§5).
    """

    def __init__(self, bits, device, rate=1.0, two_scale=False):
        self.device = device
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.rate = rate
        self.two_scale = bool(two_scale)

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

        if self.two_scale and perceptron.layer.bias is not None:
            self._transform_two_scale(perceptron, transformer, w_max, b_max)
            return

        p_max = torch.clamp(torch.maximum(w_max, b_max), min=1e-12)

        scale = self.q_max * (1.0 / p_max)

        # set_parameter_scale re-declares the shared grid (bias_scale follows).
        perceptron.set_parameter_scale(scale)

        transformer.apply_effective_parameter_transform(
            perceptron, self._quantize_param_fn(scale)
        )

    def _transform_two_scale(self, perceptron, transformer, w_max, b_max):
        weight_scale = self.q_max * (1.0 / torch.clamp(w_max, min=1e-12))
        # Integer-ratio snap: a bias grid of r whole weight-grid steps keeps
        # `bias * weight_scale = r * bias_int` exactly integer, which is the
        # lattice the chip export emits and the NF<->SCM parity consumes.
        ratio = torch.clamp(torch.ceil(b_max * weight_scale / self.q_max), min=1.0)
        bias_scale = weight_scale / ratio

        perceptron.set_parameter_scale(weight_scale)
        perceptron.set_bias_scale(bias_scale)

        transformer.apply_effective_weight_transform(
            perceptron, self._quantize_param_fn(weight_scale)
        )
        transformer.apply_effective_bias_transform(
            perceptron, self._quantize_param_fn(bias_scale)
        )

    def _quantize_param_fn(self, scale):
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

        return quantize_param
