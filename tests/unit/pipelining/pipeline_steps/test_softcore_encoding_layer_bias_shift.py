"""SoftCoreMappingStep: the TTFS bias shift must skip encoding-layer Perceptrons.

Encoding-layer Perceptrons (``is_encoding_layer=True``) are mapped to
host-side ``ComputeOp(op_type="module", params={"module": perceptron, ...})``.
At spiking-sim time, the IR executes ``Perceptron.forward()`` directly — which
already applies the training-time ``+ shift`` via the ``QuantizeDecorator``'s
nested ``ShiftDecorator``. Applying the SoftCoreMappingStep's additional
``bias += shift / activation_scale`` would double-shift the host-side forward
and break numerical parity between the FP model and the TTFS-quantized sim.

The shift compensation is only valid for on-chip ``NeuralCore`` Perceptrons
that use the plain staircase formula ``floor(V*tq)/tq`` without the
decorator-baked shift.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer


def _make_perceptron(is_encoding: bool, bias_value: float) -> Perceptron:
    p = Perceptron(4, 3, normalization=nn.Identity(), base_activation_name="ReLU")
    p.set_activation_scale(1.0)
    with torch.no_grad():
        p.layer.bias.fill_(bias_value)
    p.is_encoding_layer = is_encoding
    return p


def _apply_softcore_bias_shift(perceptrons, tq: int) -> None:
    """Exact copy of the SoftCoreMappingStep loop under test."""
    pt = PerceptronTransformer()
    for p in perceptrons:
        if getattr(p, "is_encoding_layer", False):
            continue
        shift = calculate_activation_shift(tq, p.activation_scale)
        bias_shift = shift / p.activation_scale
        pt.apply_effective_bias_transform(p, lambda b, s=bias_shift: b + s)


class TestEncodingLayerBiasShiftSkip:
    def test_chip_side_perceptron_gets_shift_added(self):
        p = _make_perceptron(is_encoding=False, bias_value=0.25)
        _apply_softcore_bias_shift([p], tq=8)
        # shift = 1.0 * 0.5 / 8 = 0.0625; bias_shift = shift/act_scale = 0.0625.
        assert torch.allclose(p.layer.bias, torch.full((4,), 0.25 + 0.0625))

    def test_encoding_layer_perceptron_bias_unchanged(self):
        p = _make_perceptron(is_encoding=True, bias_value=0.25)
        _apply_softcore_bias_shift([p], tq=8)
        assert torch.allclose(p.layer.bias, torch.full((4,), 0.25))

    def test_mixed_batch_only_chip_side_shifted(self):
        p_chip = _make_perceptron(is_encoding=False, bias_value=0.1)
        p_enc = _make_perceptron(is_encoding=True, bias_value=0.1)
        _apply_softcore_bias_shift([p_chip, p_enc], tq=16)
        # shift = 0.5/16 = 0.03125 for both (same act_scale=1.0).
        assert torch.allclose(p_chip.layer.bias, torch.full((4,), 0.1 + 0.03125))
        assert torch.allclose(p_enc.layer.bias, torch.full((4,), 0.1))
