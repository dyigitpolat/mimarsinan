"""Custom autograd activations and clamp."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class LeakyGradReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, negative_slope=1e-8):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        return torch.where(input > 0, input, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.where(input < 0, grad_output * ctx.negative_slope, grad_output)
        return grad_input, None


class LeakyGradReLU(nn.Module):
    def __init__(self, negative_slope=1e-8):
        super(LeakyGradReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return LeakyGradReLUFunction.apply(x, self.negative_slope)


class StaircaseFunction(Function):
    """Generic floor quantiser ``floor(x·Tq)/Tq`` with STE gradient.

    Supports non-integer ``Tq`` (QuantizeDecorator's ``levels/c``); TTFS NF
    activations use :class:`TTFSStaircaseFunction` (the deployment ceil kernel)
    instead — the two agree on the unit domain only for integer ``Tq``.
    """

    @staticmethod
    def forward(ctx, x, Tq):
        from mimarsinan.models.spiking.wire_semantics import floor_staircase

        return floor_staircase(x, Tq)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class TTFSStaircaseFunction(Function):
    """Deployment TTFS staircase on the clamped unit domain with STE gradient.

    Forward is the wire kernel pair's ceil form, so the torch NF quantiser is
    bit-identical to the float64 contract reference at exact grid ties.
    """

    @staticmethod
    def forward(ctx, r, S):
        from mimarsinan.models.spiking.wire_semantics import ttfs_quantized_staircase

        one = torch.ones((), dtype=r.dtype, device=r.device)
        return ttfs_quantized_staircase(r, one, int(S))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class RoundedStaircaseFunction(Function):
    """Nearest-neighbour quantiser round(x * T) / T with STE gradient."""

    @staticmethod
    def forward(ctx, x, T):
        return torch.round(x * T) / T

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class ChipInputQuantizer(nn.Module):
    """STE round-to-chip-rate quantiser for encoding-layer inputs."""

    def __init__(self, T: int, activation_scale: nn.Parameter | torch.Tensor | float):
        super().__init__()
        self.T = int(T)
        if isinstance(activation_scale, (int, float)):
            activation_scale = nn.Parameter(
                torch.tensor(float(activation_scale)), requires_grad=False
            )
        self.activation_scale = activation_scale

    def _snap(self, x_norm: torch.Tensor) -> torch.Tensor:
        return RoundedStaircaseFunction.apply(x_norm, self.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)
        x_norm = (x / safe_scale).clamp(0.0, 1.0)
        return self._snap(x_norm) * safe_scale


class TTFSGridSnapFunction(Function):
    """TTFS encode→decode round trip ``(S - round(S·(1-x)))/S`` with STE gradient.

    Forward is the wire kernel pair's grid quantize (bit-identical to
    ``ttfs_encoding.ttfs_input_grid_quantize``); differs from a plain
    ``round(x·S)/S`` at half-step ties.
    """

    @staticmethod
    def forward(ctx, x, S):
        from mimarsinan.models.spiking.wire_semantics import ttfs_grid_quantize

        return ttfs_grid_quantize(x, int(S))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class TTFSInputGridQuantizer(ChipInputQuantizer):
    """STE TTFS grid snap q(x) for synchronized encoding-layer inputs."""

    def _snap(self, x_norm: torch.Tensor) -> torch.Tensor:
        return TTFSGridSnapFunction.apply(x_norm, self.T)


_CLAMP_LEAK = 0.01
"""Minimum out-of-range gradient for DifferentiableClamp."""


class DifferentiableClamp(Function):
    """Differentiable clamp with optional gradient flow to the bounds."""

    @staticmethod
    def forward(ctx, x, a, b):
        assert a.dim() <= 0 and b.dim() <= 0, (
            f"DifferentiableClamp expects scalar bounds; got a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}"
        )
        a_dev = a.to(x.device)
        b_dev = b.to(x.device)
        ctx.save_for_backward(x, a_dev, b_dev)
        return torch.clamp(x, a_dev, b_dev)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        below_grad = torch.clamp_min(torch.exp(x - a), _CLAMP_LEAK)
        above_grad = torch.clamp_min(torch.exp(b - x), _CLAMP_LEAK)
        grad_x = torch.where(
            x < a,
            below_grad,
            torch.where(x > b, above_grad, torch.ones_like(x)),
        )
        # Bounds are scalars; sum grad over the input tensor.
        grad_a = (grad_output * (x < a).to(grad_output.dtype)).sum()
        grad_b = (grad_output * (x > b).to(grad_output.dtype)).sum()
        return grad_output * grad_x, grad_a, grad_b


