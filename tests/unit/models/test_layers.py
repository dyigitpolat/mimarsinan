"""Tests for activation layers, decorators, and custom autograd functions."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.layers import (
    LeakyGradReLU,
    StaircaseFunction,
    DifferentiableClamp,
    NoisyDropout,
    SavedTensorDecorator,
    StatsDecorator,
    ShiftDecorator,
    ScaleDecorator,
    ClampDecorator,
    QuantizeDecorator,
    TransformedActivation,
    DecoratedActivation,
    RateAdjustedDecorator,
    MixAdjustmentStrategy,
    RandomMaskAdjustmentStrategy,
    NestedDecoration,
    MaxValueScaler,
    FrozenStatsNormalization,
)


# ---------------------------------------------------------------------------
# Custom autograd functions
# ---------------------------------------------------------------------------

class TestLeakyGradReLU:
    def test_positive_passthrough(self):
        m = LeakyGradReLU()
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(m(x), x)

    def test_negative_zeroed(self):
        m = LeakyGradReLU()
        x = torch.tensor([-1.0, -0.5])
        out = m(x)
        assert (out == 0).all()

    def test_gradient_positive(self):
        m = LeakyGradReLU()
        x = torch.tensor([2.0], requires_grad=True)
        y = m(x)
        y.backward()
        assert x.grad.item() == pytest.approx(1.0)

    def test_gradient_negative_leaky(self):
        slope = 1e-8
        m = LeakyGradReLU(negative_slope=slope)
        x = torch.tensor([-1.0], requires_grad=True)
        y = m(x)
        y.backward()
        assert x.grad.item() == pytest.approx(slope)


class TestStaircaseFunction:
    def test_quantizes_to_floor(self):
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.99])
        Tq = 4.0
        out = StaircaseFunction.apply(x, Tq)
        expected = torch.floor(x * Tq) / Tq
        assert torch.allclose(out, expected)

    def test_straight_through_gradient(self):
        x = torch.tensor([0.5], requires_grad=True)
        y = StaircaseFunction.apply(x, 4.0)
        y.backward()
        assert x.grad.item() == pytest.approx(1.0)

    def test_integer_Tq(self):
        x = torch.tensor([0.33])
        out = StaircaseFunction.apply(x, 3.0)
        assert out.item() == pytest.approx(0.0, abs=0.35)


class TestDifferentiableClamp:
    def test_in_range_passthrough(self):
        x = torch.tensor([0.5])
        out = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        assert out.item() == pytest.approx(0.5)

    def test_clamp_below(self):
        x = torch.tensor([-1.0])
        out = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        assert out.item() == pytest.approx(0.0)

    def test_clamp_above(self):
        x = torch.tensor([2.0])
        out = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        assert out.item() == pytest.approx(1.0)

    def test_gradient_in_range(self):
        x = torch.tensor([0.5], requires_grad=True)
        y = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        y.backward()
        assert x.grad.item() == pytest.approx(1.0)

    def test_gradient_below_is_exponential(self):
        x = torch.tensor([-1.0], requires_grad=True)
        y = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        y.backward()
        assert 0 < x.grad.item() < 1.0


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

class TestClampDecorator:
    def test_output_clamps(self):
        d = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        x = torch.tensor([-0.5, 0.5, 1.5])
        out = d.output_transform(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_input_is_identity(self):
        d = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        x = torch.tensor([2.0])
        assert d.input_transform(x).item() == 2.0


class TestQuantizeDecorator:
    def test_quantization_produces_discrete_values(self):
        d = QuantizeDecorator(torch.tensor(4.0), torch.tensor(1.0))
        x = torch.tensor([0.0, 0.125, 0.26, 0.5, 0.76, 1.0])
        out = d.output_transform(x)
        unique = out.unique()
        assert len(unique) <= 5


class TestShiftDecorator:
    def test_shifts_input(self):
        d = ShiftDecorator(torch.tensor(0.5))
        x = torch.tensor([1.0, 2.0])
        out = d.input_transform(x)
        assert torch.allclose(out, torch.tensor([0.5, 1.5]))

    def test_output_is_identity(self):
        d = ShiftDecorator(torch.tensor(0.5))
        x = torch.tensor([1.0])
        assert d.output_transform(x).item() == 1.0


class TestScaleDecorator:
    def test_scales_output(self):
        d = ScaleDecorator(torch.tensor(2.0))
        x = torch.tensor([3.0])
        assert d.output_transform(x).item() == pytest.approx(6.0)


class TestSavedTensorDecorator:
    def test_saves_output(self):
        d = SavedTensorDecorator()
        x = torch.randn(2, 4)
        d.output_transform(x)
        assert d.latest_output is not None
        assert d.latest_output.shape == (2, 4)

    def test_saves_input(self):
        d = SavedTensorDecorator()
        x = torch.randn(2, 4)
        d.input_transform(x)
        assert d.latest_input is not None

    def test_1d_input_not_saved(self):
        d = SavedTensorDecorator()
        x = torch.tensor([1.0])
        d.input_transform(x)
        assert d.latest_input is None


class TestStatsDecorator:
    def test_computes_stats(self):
        d = StatsDecorator()
        x = torch.randn(4, 8)
        d.output_transform(x)
        assert d.out_mean is not None
        assert d.out_var is not None
        assert d.out_max is not None
        assert d.out_min is not None
        assert d.out_hist is not None


# ---------------------------------------------------------------------------
# TransformedActivation
# ---------------------------------------------------------------------------

class TestTransformedActivation:
    def test_identity_base_no_decorators(self):
        ta = TransformedActivation(nn.Identity(), [])
        x = torch.tensor([1.0, -1.0])
        assert torch.allclose(ta(x), x)

    def test_decorate_and_pop(self):
        ta = TransformedActivation(nn.Identity(), [])
        ta.decorate(ScaleDecorator(torch.tensor(3.0)))
        x = torch.tensor([2.0])
        assert ta(x).item() == pytest.approx(6.0)

        popped = ta.pop_decorator()
        assert isinstance(popped, ScaleDecorator)
        assert ta(x).item() == pytest.approx(2.0)

    def test_multiple_decorators_compose(self):
        clamp = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        scale = ScaleDecorator(torch.tensor(2.0))
        ta = TransformedActivation(LeakyGradReLU(), [clamp, scale])
        x = torch.tensor([5.0])
        out = ta(x)
        assert out.item() <= 2.0

    def test_pop_from_empty_raises(self):
        ta = TransformedActivation(nn.Identity(), [])
        with pytest.raises(IndexError):
            ta.pop_decorator()


class TestRateAdjustedDecorator:
    def test_rate_zero_is_base(self):
        base_val = torch.tensor([1.0, 2.0])
        d = ClampDecorator(torch.tensor(0.0), torch.tensor(0.5))
        rad = RateAdjustedDecorator(0.0, d, MixAdjustmentStrategy())
        out = rad.output_transform(base_val)
        assert torch.allclose(out, base_val)

    def test_rate_one_is_fully_decorated(self):
        base_val = torch.tensor([2.0])
        d = ClampDecorator(torch.tensor(0.0), torch.tensor(0.5))
        rad = RateAdjustedDecorator(1.0, d, MixAdjustmentStrategy())
        out = rad.output_transform(base_val)
        assert out.item() == pytest.approx(0.5)


class TestMaxValueScaler:
    def test_training_updates_max(self):
        scaler = MaxValueScaler()
        scaler.train()
        x = torch.tensor([10.0, 5.0])
        _ = scaler(x)
        assert scaler.max_value.item() > 1.0

    def test_eval_does_not_update(self):
        scaler = MaxValueScaler()
        scaler.eval()
        old_max = scaler.max_value.item()
        _ = scaler(torch.tensor([100.0]))
        assert scaler.max_value.item() == old_max
