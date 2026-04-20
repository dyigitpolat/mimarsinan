from mimarsinan.transformations.transformation_utils import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.lsq_quantization import LSQQuantizer

import torch.nn as nn
import torch


class NormalizationAwarePerceptronQuantization:
    """Weight quantisation for a perceptron using LSQ + STE (Phase C1).

    Historical behaviour
    --------------------
    The legacy implementation produced ``w_q`` by closed-form
    round-to-nearest with a max-abs-based scale and let
    ``PerceptronTransformTuner`` stochastically mix ``w_fp`` and
    ``w_q`` at training time to transition smoothly.  The gradients
    then came through the mixed tensor but were applied to the
    FP master copy, which is the core "what loss surface am I
    actually on" awkwardness outlined in the Phase C refactor plan.

    New behaviour
    -------------
    The transform installs an :class:`LSQQuantizer` as a child module of
    the perceptron (``perceptron.weight_quantizer``), seeds its step
    from the effective weight's max-abs statistic (the exact legacy
    formula), then runs the quantizer's forward pass to bake the
    hard-quantized values into ``perceptron.layer.weight.data`` (and
    bias).  The quantizer's ``log_scale`` is a proper trainable
    parameter, so every subsequent training step back-propagates
    through STE into ``log_scale`` *and* the underlying FP weights in
    one pass -- no shadow copy needed at the quantizer level.

    Backwards compatibility
    -----------------------
    The ``rate`` parameter is accepted for API compatibility with the
    ``PerceptronTransformTuner`` call sites that still mix FP and Q
    tensors *externally* (via ``_mixed_perceptron_transform``).  The
    internal quantiser is always hard -- LSQ+STE is already a
    fully-differentiable transition on its own, and rate-based
    post-hoc mixing on top of it still works because ``rate=1.0``
    just selects the fully-quantized branch.
    """

    def __init__(self, bits, device, rate=1.0):
        self.device = device
        self.bits = int(bits)
        self.q_min = -(2 ** (bits - 1))
        self.q_max = (2 ** (bits - 1)) - 1
        self.rate = float(rate)

    def _get_or_create_quantizer(self, perceptron) -> LSQQuantizer:
        """Return the perceptron's LSQ quantiser, creating one if needed.

        The quantiser is created lazily so (1) existing models saved
        before Phase C1 load cleanly and (2) the bit-width is always
        derived from the current step's config instead of a stale
        state-dict entry.  If an existing quantiser has the same
        bit-width we reuse it so its learnt ``log_scale`` survives
        repeated transform calls (the PerceptronTransformTuner path
        invokes ``transform`` on every forward rebuild)."""
        q = getattr(perceptron, "weight_quantizer", None)
        if q is not None and getattr(q, "bits", None) == self.bits:
            return q
        q = LSQQuantizer(bits=self.bits).to(self.device)
        perceptron.set_weight_quantizer(q)
        return q

    def transform(self, perceptron):
        # Normalization-aware view: work on the "effective" weight and
        # bias post-fusion so the closed-form seed for log_scale matches
        # what the hardware will see after eventual BN-fusion.
        w = PerceptronTransformer().get_effective_weight(perceptron)
        b = PerceptronTransformer().get_effective_bias(perceptron)

        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        p_max = max(float(w_max.item()), float(b_max.item()), 1e-12)

        quantizer = self._get_or_create_quantizer(perceptron)

        # Only (re)seed the step when the quantiser has no meaningful
        # history yet -- i.e. either it was just created (log_scale is
        # still 0.0 from the default init) or the current max-abs
        # statistic is larger than the step's implied range.  This keeps
        # the learnt ceiling from being wiped out on every ``transform``
        # call while still catching the first-call initialisation.
        current_step = float(torch.exp(quantizer.log_scale.detach()).item())
        implied_range = self.q_max * current_step
        if implied_range <= 0.0 or abs(quantizer.log_scale.detach().item()) < 1e-9:
            quantizer.init_from_tensor(torch.tensor([p_max]))
        elif implied_range < p_max * 0.5:
            quantizer.init_from_tensor(torch.tensor([p_max]))

        # Publish the legacy ``parameter_scale`` so downstream mapping /
        # simulation code (which still reads ``parameter_scale`` to undo
        # the quantization) sees the same "integers per FP unit" value
        # LSQ is using internally.
        step = torch.exp(quantizer.log_scale.detach())
        legacy_scale = torch.tensor(1.0) / step
        perceptron.set_parameter_scale(legacy_scale)

        # Bake the hard-quantized effective weights/biases into the
        # perceptron's layer so non-differentiable downstream steps see
        # the final integer grid.  Uses ``no_grad`` because we are
        # overwriting ``.data`` directly; gradient flow during QAT
        # training is handled by the quantiser's forward pass (see
        # ``_mixed_perceptron_transform`` and ``PerceptronTransformTuner``).
        with torch.no_grad():
            def quantize_param(param):
                return quantizer(param)

            PerceptronTransformer().apply_effective_parameter_transform(
                perceptron, quantize_param
            )

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
