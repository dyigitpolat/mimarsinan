"""The WQ mechanism under bias-row splitting: the grid comes from ``max|w|``
alone and the bias grid IS the row count.

This is the measured pathology, in one place: with a shared max(|w|,|b|) grid a
layer whose bias dominates by 7x spends the whole register range on the bias
and quantizes its weights to a handful of levels. The split projection prices
the bias in rows instead, and the weight grid recovers.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.bias_rows import bias_rows_from_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

BITS = 4
Q_MAX = (2 ** (BITS - 1)) - 1

# The measured platform-J layer-0 shape: max|b| / max|w| = 7.04.
W_MAX = 0.1
B_MAX = 0.704


def _bias_dominated_perceptron():
    torch.manual_seed(0)
    p = Perceptron(6, 8, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    with torch.no_grad():
        p.layer.weight.data.uniform_(-W_MAX, W_MAX)
        p.layer.weight.data[0, 0] = W_MAX
        p.layer.bias.data.uniform_(-B_MAX, B_MAX)
        p.layer.bias.data[0] = B_MAX
    return p


def _quantize(p, **kwargs):
    NormalizationAwarePerceptronQuantization(
        bits=BITS, device="cpu", rate=1.0, **kwargs
    ).transform(p)
    return p


def _weight_levels(p):
    w = PerceptronTransformer().get_effective_weight(p)
    return int(torch.unique(torch.round(w * p.parameter_scale)).numel())


class TestTheCollapseAndTheRecovery:
    def test_the_shared_grid_collapses_the_weights(self):
        """The baseline pathology, stated as a test so the fix has a witness."""
        p = _quantize(_bias_dominated_perceptron())
        assert float(p.parameter_scale) == pytest.approx(Q_MAX / B_MAX, rel=0.15)
        assert _weight_levels(p) <= 3

    def test_the_weight_only_grid_recovers_the_levels(self):
        p = _quantize(_bias_dominated_perceptron(), two_scale=True)
        assert float(p.parameter_scale) == pytest.approx(Q_MAX / W_MAX, rel=0.15)
        assert _weight_levels(p) >= 8

    def test_the_recovered_grid_prices_the_bias_in_rows(self):
        p = _quantize(_bias_dominated_perceptron(), two_scale=True)
        # ceil(0.704 * 70 / 7) = 8 rows.
        assert bias_rows_from_scales(p.bias_scale, p.parameter_scale) == 8


class TestRowFloor:
    def test_the_floor_buys_more_rows_than_the_bound_asks_for(self):
        p = _quantize(_bias_dominated_perceptron(), two_scale=True, bias_rows_floor=16)
        assert bias_rows_from_scales(p.bias_scale, p.parameter_scale) == 16

    def test_a_floor_below_the_bound_never_clips_the_bias(self):
        """A declared k the trained bias outgrows RAISES the row count; it must
        never silently saturate the bias at the register edge."""
        p = _quantize(_bias_dominated_perceptron(), two_scale=True, bias_rows_floor=2)
        assert bias_rows_from_scales(p.bias_scale, p.parameter_scale) == 8

    def test_the_floor_is_inert_without_the_weight_only_grid(self):
        a = _quantize(_bias_dominated_perceptron(), bias_rows_floor=16)
        b = _quantize(_bias_dominated_perceptron())
        assert float(a.parameter_scale) == float(b.parameter_scale)
        assert bias_rows_from_scales(a.bias_scale, a.parameter_scale) == 1


class TestBitIdenticalWhenOff:
    def test_the_default_projection_is_untouched(self):
        a = _quantize(_bias_dominated_perceptron())
        b = _quantize(_bias_dominated_perceptron(), bias_rows_floor=None)
        torch.testing.assert_close(
            PerceptronTransformer().get_effective_weight(a),
            PerceptronTransformer().get_effective_weight(b),
            rtol=0, atol=0,
        )
