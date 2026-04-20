from mimarsinan.transformations.transformation_utils import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch
import copy

class NormalizationAwarePerceptronQuantization:
    """Normalization-aware per-perceptron weight quantization.

    At ``rate=1.0`` (the default) this is standard symmetric integer
    quantization: every effective weight and bias is snapped to the
    nearest multiple of ``1/scale``, where ``scale = q_max / p_max``
    is computed from the current effective-weight distribution.

    At intermediate ``rate`` values it linearly interpolates, in
    weight-value space, between the FP effective weight and the fully
    quantized effective weight::

        w_rate = rate * q(w_fp) + (1 - rate) * w_fp

    Important properties of this interpolation, which motivate the
    choice over the legacy random-mask stochastic mix applied by
    ``PerceptronTransformTuner``:

    * ``rate=0.0`` is a true identity (no effective-weight change).
    * ``rate=1.0`` matches the legacy full-quantization output bit-
      exactly, so downstream IR/simulation paths are unaffected.
    * Every partial rate produces a *coherent* small perturbation of
      the FP model, not a random hybrid of FP and integer weights.
      The maximum per-weight perturbation is bounded by
      ``rate * max |q(w_fp) - w_fp|``, which is at most
      ``rate * 0.5 / scale`` plus clipping contributions.
    * ``parameter_scale`` is always set to the full-range scale derived
      at ``rate=1``, regardless of the effective rate. Downstream IR
      mapping needs the scale that the final quantized weights will
      use, not a rate-scaled variant.
    """

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

        # Guard against zero/near-zero p_max (e.g. after pruning zeroes weights)
        p_max = max(p_max, 1e-12)

        scale = self.q_max * (1.0 / p_max)

        # scale is always the full-range scale: downstream IR mapping
        # depends on the final (rate=1) scale, not a rate-scaled proxy.
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
            # Linear interpolation in weight-value space between FP and
            # quantized. At rate=0 this is identity; at rate=1 it is the
            # legacy full-quantization output.
            if rate >= 1.0:
                return rescaled
            if rate <= 0.0:
                return param
            return rate * rescaled + (1.0 - rate) * param

        PerceptronTransformer().apply_effective_parameter_transform(perceptron, quantize_param)
