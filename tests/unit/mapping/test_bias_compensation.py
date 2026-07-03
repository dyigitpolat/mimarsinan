"""Characterization pins for bias_compensation: exact tensors in = exact tensors out.

These literals were captured from the pre-merge neg_shift_bias / ttfs_bias
modules; every comparison is bit-exact (torch.equal) so the merged module
cannot drift numerically. The synchronized floor-collapse depends on the
ttfs bake's +shift semantics — a double-shift here once cost -1.9pp.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.support.bias_compensation import (
    apply_additive_effective_bias_shift,
    apply_negative_shift_bias,
    apply_ttfs_quantization_bias_compensation,
    apply_ttfs_quantized_bias_shift,
    negative_shifts_from_min,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

_W = torch.tensor([[1.0, -2.0], [0.5, 4.0], [-3.0, 0.25]])
_B = torch.tensor([0.5, -1.5, 2.0])


def _perceptron(activation_scale=None, bias=True):
    p = Perceptron(3, 2, bias=bias, normalization=nn.Identity())
    p.layer.weight.data = _W.clone()
    if bias:
        p.layer.bias.data = _B.clone()
    if activation_scale is not None:
        p.activation_scale.data = torch.tensor(activation_scale)
    return p


class _Model:
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def get_perceptrons(self):
        return self._perceptrons


def _assert_bias_exact(perceptron, expected):
    assert torch.equal(perceptron.layer.bias.data, torch.tensor(expected))


class TestApplyAdditiveEffectiveBiasShift:
    def test_adds_shift_in_effective_domain_identity_scale(self):
        p = _perceptron()
        baked = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked is True
        _assert_bias_exact(p, [0.625, -1.375, 2.125])
        assert p._flag is True

    def test_effective_domain_rescales_through_activation_scale(self):
        p = _perceptron(0.5)
        apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        _assert_bias_exact(p, [0.5625, -1.4375, 2.0625])

    def test_idempotent_per_flag(self):
        p = _perceptron()
        apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        baked_again = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked_again is False
        _assert_bias_exact(p, [0.625, -1.375, 2.125])

    def test_bias_none_sets_flag_without_crash(self):
        p = _perceptron(bias=False)
        baked = apply_additive_effective_bias_shift(p, 0.125, baked_flag="_flag")
        assert baked is True and p.layer.bias is None and p._flag is True


class TestApplyNegativeShiftBias:
    def test_per_axon_shift_exact(self):
        p = _perceptron()
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])
        assert p._neg_shift_baked is True

    def test_scalar_shift_exact(self):
        p = _perceptron()
        apply_negative_shift_bias(p, 1.5)
        _assert_bias_exact(p, [2.0, -8.25, 6.125])

    def test_activation_scale_cancels_for_identity_norm(self):
        p = _perceptron(0.5)
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])

    def test_idempotent(self):
        p = _perceptron()
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        _assert_bias_exact(p, [0.5, -2.75, 3.4375])

    def test_bias_none_sets_flag_without_crash(self):
        p = _perceptron(bias=False)
        apply_negative_shift_bias(p, torch.tensor([0.5, 0.25]))
        assert p.layer.bias is None and p._neg_shift_baked is True


class TestNegativeShiftsFromMin:
    def test_exact_shift_table(self):
        out = negative_shifts_from_min(
            {5: np.array([-0.4, 0.1, -1.2]), 7: np.array([0.0, 0.3, 0.5])}
        )
        assert set(out) == {5}
        assert out[5].dtype == np.float64
        np.testing.assert_array_equal(out[5], np.array([0.4, 0.0, 1.2]))


class TestApplyTtfsQuantizationBiasCompensation:
    def test_plus_half_step_shift_exact(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])
        assert model.get_perceptrons()[0]._ttfs_shift_baked_into_bias is True

    def test_activation_scale_cancels_in_stored_bias_step(self):
        model = _Model([_perceptron(0.5)])
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.5625, -1.4375, 2.0625])

    def test_idempotent_no_double_shift(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantization_bias_compensation(model, 4)
        apply_ttfs_quantization_bias_compensation(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])

    def test_encoding_layer_skipped_and_unflagged(self):
        encoder = _perceptron()
        encoder.is_encoding_layer = True
        apply_ttfs_quantization_bias_compensation(_Model([encoder]), 4)
        _assert_bias_exact(encoder, [0.5, -1.5, 2.0])
        assert not getattr(encoder, "_ttfs_shift_baked_into_bias", False)

    def test_alias_matches(self):
        model = _Model([_perceptron()])
        apply_ttfs_quantized_bias_shift(model, 4)
        _assert_bias_exact(model.get_perceptrons()[0], [0.625, -1.375, 2.125])
