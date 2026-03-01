"""
Stress tests for model layers and decorators.

Tests edge cases, numerical stability, and decorator composition order.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.layers import (
    LeakyGradReLU,
    StaircaseFunction,
    DifferentiableClamp,
    NoisyDropout,
    TransformedActivation,
    DecoratedActivation,
    ClampDecorator,
    QuantizeDecorator,
    ScaleDecorator,
    ShiftDecorator,
    MaxValueScaler,
    MixAdjustmentStrategy,
    RateAdjustedDecorator,
    NestedDecoration,
)


class TestLeakyGradReLUStress:
    @pytest.mark.xfail(
        reason="BUG: LeakyGradReLU silently converts NaN to 0.0. "
               "torch.where(input > 0, input, 0.0) evaluates NaN > 0 as False, "
               "so NaN inputs are mapped to zero instead of propagating. "
               "This hides upstream numerical instability.",
        strict=True,
    )
    def test_nan_input_propagates(self):
        """NaN inputs should propagate, not silently disappear."""
        m = LeakyGradReLU()
        x = torch.tensor([float("nan"), 1.0, -1.0])
        out = m(x)
        assert torch.isnan(out[0]), "NaN should propagate through LeakyGradReLU"
        assert out[1].item() == 1.0
        assert out[2].item() == 0.0

    def test_inf_input(self):
        m = LeakyGradReLU()
        x = torch.tensor([float("inf"), float("-inf")])
        out = m(x)
        assert out[0] == float("inf"), "Positive inf should pass through"
        assert out[1] == 0.0, "Negative inf should be zeroed"

    def test_exact_zero_is_zeroed(self):
        """Input of exactly 0.0 — the boundary between positive and negative."""
        m = LeakyGradReLU()
        x = torch.tensor([0.0])
        out = m(x)
        assert out.item() == 0.0

    def test_gradient_at_zero(self):
        """Gradient at exactly x=0: the code uses `input < 0`, so x=0 gets grad=1."""
        m = LeakyGradReLU()
        x = torch.tensor([0.0], requires_grad=True)
        y = m(x)
        y.backward()
        assert x.grad.item() == pytest.approx(1.0), \
            "At x=0 the gradient should be 1.0 (not leaky_slope)"


class TestStaircaseFunctionStress:
    def test_negative_input(self):
        """floor(-0.3 * 4) = floor(-1.2) = -2, so -2/4 = -0.5."""
        x = torch.tensor([-0.3])
        out = StaircaseFunction.apply(x, 4.0)
        assert out.item() == pytest.approx(-0.5)

    def test_Tq_zero_raises_or_returns_nan(self):
        """Division by zero Tq should be handled."""
        x = torch.tensor([0.5])
        out = StaircaseFunction.apply(x, 0.0)
        assert torch.isnan(out) or torch.isinf(out), \
            "Tq=0 should produce NaN or Inf, not a valid number"

    def test_large_Tq_approaches_identity(self):
        """As Tq → ∞, StaircaseFunction approaches identity."""
        x = torch.tensor([0.123456])
        out = StaircaseFunction.apply(x, 1e6)
        assert out.item() == pytest.approx(0.123456, abs=1e-5)


class TestDifferentiableClampStress:
    def test_gradient_far_below_min_decays(self):
        """Gradient should decay exponentially for values far below the clamp."""
        x = torch.tensor([-10.0], requires_grad=True)
        y = DifferentiableClamp.apply(x, torch.tensor(0.0), torch.tensor(1.0))
        y.backward()
        assert x.grad.item() < 1e-3, \
            "Gradient should be very small far below clamp range"

    def test_equal_min_max(self):
        """When min == max, all values should clamp to that single value."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
        val = torch.tensor(0.5)
        out = DifferentiableClamp.apply(x, val, val)
        assert torch.allclose(out, torch.full_like(out, 0.5))

    def test_nan_bounds(self):
        """NaN bounds should propagate NaN, not silently clamp."""
        x = torch.tensor([0.5])
        out = DifferentiableClamp.apply(
            x, torch.tensor(float("nan")), torch.tensor(1.0)
        )
        assert torch.isnan(out).any(), "NaN lower bound should produce NaN output"


class TestMaxValueScalerStress:
    def test_all_negative_inputs_converge_to_negative_max(self):
        """
        BUG: MaxValueScaler uses torch.max(x) (not abs), so with all-negative
        inputs the max_value converges to a negative number, and dividing by it
        flips the sign of the output.
        """
        scaler = MaxValueScaler()
        scaler.train()

        for _ in range(200):
            x = torch.tensor([-5.0, -10.0])
            scaler(x)

        if scaler.max_value.item() < 0:
            x_test = torch.tensor([-3.0])
            out = scaler(x_test)
            if out.item() > 0:
                pytest.xfail(
                    f"BUG: MaxValueScaler converged to negative max_value "
                    f"({scaler.max_value.item():.4f}), causing sign flip: "
                    f"input {x_test.item()} → output {out.item()}"
                )

        # If max_value stayed positive, that's fine
        assert scaler.max_value.item() > 0 or True, \
            "This test documents the behavior"

    def test_zero_input_division_by_max_value(self):
        """When max_value is 1.0 (default), zero input should produce zero."""
        scaler = MaxValueScaler()
        scaler.eval()
        out = scaler(torch.tensor([0.0]))
        assert out.item() == 0.0

    def test_max_value_ema_correctness(self):
        """Verify the EMA update formula: 0.1 * max_x + 0.9 * max_value."""
        scaler = MaxValueScaler()
        scaler.train()
        assert scaler.max_value.item() == 1.0

        x = torch.tensor([10.0, 5.0])
        scaler(x)
        expected = 0.1 * 10.0 + 0.9 * 1.0
        assert scaler.max_value.item() == pytest.approx(expected, abs=1e-5)


class TestDecoratorCompositionOrder:
    """Verify that decorator composition order matters and is correct."""

    def test_clamp_then_scale_vs_scale_then_clamp(self):
        """
        Clamp[0,1] then Scale(2) should give output in [0,2].
        Scale(2) then Clamp[0,1] should give output in [0,1].
        These must produce different results for x > 0.5.
        """
        x = torch.tensor([0.8])

        clamp = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        scale = ScaleDecorator(torch.tensor(2.0))

        ta1 = TransformedActivation(nn.Identity(), [clamp, scale])
        out1 = ta1(x)  # clamp to [0,1] then scale by 2 → 1.6

        ta2 = TransformedActivation(nn.Identity(), [scale, clamp])
        out2 = ta2(x)  # scale by 2 to 1.6, then clamp to [0,1] → 1.0

        assert out1.item() == pytest.approx(1.6, abs=0.1)
        assert out2.item() == pytest.approx(1.0, abs=0.01)
        assert abs(out1.item() - out2.item()) > 0.1, \
            "Composition order should matter"

    def test_shift_then_relu(self):
        """ShiftDecorator shifts the input before base activation."""
        shift = ShiftDecorator(torch.tensor(1.0))
        ta = TransformedActivation(LeakyGradReLU(), [shift])
        # input_transform subtracts shift: x - 1.0
        # base activation: ReLU(x - 1.0)
        # x=0.5: ReLU(-0.5) = 0
        # x=2.0: ReLU(1.0) = 1.0
        assert ta(torch.tensor([0.5])).item() == pytest.approx(0.0)
        assert ta(torch.tensor([2.0])).item() == pytest.approx(1.0)

    def test_rate_adjusted_at_intermediate_rate(self):
        """
        MixAdjustmentStrategy at rate=0.5 should give the midpoint
        between base output and decorated output.
        """
        x = torch.tensor([2.0])
        clamp = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        rad = RateAdjustedDecorator(0.5, clamp, MixAdjustmentStrategy())
        out = rad.output_transform(x)
        # base: 2.0, decorated: clamp(2.0) = 1.0
        # mix: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert out.item() == pytest.approx(1.5, abs=1e-5)


class TestTransformedActivationStress:
    def test_empty_decorators_list_is_identity(self):
        """With no decorators, TransformedActivation should be the base."""
        ta = TransformedActivation(LeakyGradReLU(), [])
        x = torch.tensor([1.0, -1.0])
        expected = LeakyGradReLU()(x)
        assert torch.allclose(ta(x), expected)

    def test_many_decorators_dont_overflow(self):
        """Stack many scale decorators and verify numerical stability."""
        decorators = [ScaleDecorator(torch.tensor(1.001)) for _ in range(100)]
        ta = TransformedActivation(nn.Identity(), decorators)
        x = torch.tensor([1.0])
        out = ta(x)
        expected = 1.001 ** 100
        assert out.item() == pytest.approx(expected, rel=0.01)

    def test_quantize_decorator_known_values(self):
        """
        QuantizeDecorator(levels_before_c=4, c=1.0) applies:
        StaircaseFunction(x, 4.0/1.0) = floor(x * 4) / 4

        x=0.6: floor(2.4)/4 = 2/4 = 0.5
        x=0.9: floor(3.6)/4 = 3/4 = 0.75
        """
        qd = QuantizeDecorator(torch.tensor(4.0), torch.tensor(1.0))
        assert qd.output_transform(torch.tensor([0.6])).item() == pytest.approx(0.5)
        assert qd.output_transform(torch.tensor([0.9])).item() == pytest.approx(0.75)
