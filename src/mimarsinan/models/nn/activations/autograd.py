"""Custom autograd activations and clamp."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from mimarsinan.models.spiking.wire_semantics import (
    floor_staircase,
    ttfs_grid_quantize,
    ttfs_quantized_staircase,
)


class LeakyGradReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, negative_slope=1e-8):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        # ``input <= 0`` (not ``<``) keeps NaN on the pass-through branch so NaN propagates instead of silently becoming 0.
        return torch.where(input <= 0, torch.zeros_like(input), input)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
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
    """Generic floor quantiser ``floor(x·Tq)/Tq`` with STE gradient; supports
    non-integer ``Tq``. TTFS NF activations use :class:`TTFSStaircaseFunction`
    instead (the two agree on the unit domain only for integer ``Tq``)."""

    @staticmethod
    def forward(ctx, x, Tq):
        return floor_staircase(x, Tq)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        grad_input = grad_output.clone()
        return grad_input, None, None


class TTFSStaircaseFunction(Function):
    """Deployment TTFS staircase on the clamped unit domain with STE gradient;
    forward is the wire kernel pair's ceil form, so the torch NF quantiser is
    bit-identical to the float64 contract reference at exact grid ties."""

    @staticmethod
    def forward(ctx, r, S):
        one = torch.ones((), dtype=r.dtype, device=r.device)
        return ttfs_quantized_staircase(r, one, int(S))

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        return grad_output.clone(), None


class TTFSComparatorHalfStepStaircaseFunction(Function):
    """[E3] the deployment ceil staircase with the comparator-side half-step
    (shifted compare ladder ``ceil(S·(1-r) - 1/2)``) and STE gradient — the NF
    twin of the SCM kernel when the contract carries ``comparator_half_step``."""

    @staticmethod
    def forward(ctx, r, S):
        one = torch.ones((), dtype=r.dtype, device=r.device)
        return ttfs_quantized_staircase(r, one, int(S), comparator_half_step=True)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        return grad_output.clone(), None


LIF_EXACT_QAT_THETA_FLOOR = 1e-3
"""Positivity floor for the in-loop trainable theta: the forward clamps at it,
so degenerate shrinkage is self-limiting (lif_exact_qat_program.md §4.2)."""


class LIFCountStaircaseFunction(Function):
    """[lif_exact_qat] the deployed LIF count staircase ``θ·clamp(F(T·z/θ),0,T)/T``
    with a clamp-gated identity STE to z and the LSQ theta gradient
    ``q(r) − r·1[0<r<1]`` (in-band grid-residual descent; saturation pushes θ up)."""

    @staticmethod
    def forward(ctx, z, theta, T, strict):
        safe = theta.clamp(min=LIF_EXACT_QAT_THETA_FLOOR)
        r = z / safe
        c = torch.ceil(T * r) - 1.0 if strict else torch.floor(T * r)
        q = c.clamp(0.0, float(T)) / T
        ctx.save_for_backward(r, q, safe)
        return safe * q

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        r, q, safe = ctx.saved_tensors
        inband = ((r > 0) & (r < 1)).to(g.dtype)
        grad_z = g * inband
        grad_theta = g * (q - r * inband)
        while grad_theta.dim() > safe.dim():
            grad_theta = grad_theta.sum(0)
        for i in range(grad_theta.dim()):
            if safe.shape[i] == 1 and grad_theta.shape[i] != 1:
                grad_theta = grad_theta.sum(i, keepdim=True)
        return grad_z, grad_theta, None, None


class RoundedStaircaseFunction(Function):
    """Nearest-neighbour quantiser round(x * T) / T with STE gradient."""

    @staticmethod
    def forward(ctx, x, T):
        return torch.round(x * T) / T

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
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
    """TTFS encode→decode round trip ``(S - round(S·(1-x)))/S`` with STE gradient;
    forward is the wire kernel pair's grid quantize (bit-identical to
    ``ttfs_encoding.ttfs_input_grid_quantize``), differing at half-step ties."""

    @staticmethod
    def forward(ctx, x, S):
        return ttfs_grid_quantize(x, int(S))

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        return grad_output.clone(), None


class TTFSInputGridQuantizer(ChipInputQuantizer):
    """STE TTFS grid snap q(x) for synchronized encoding-layer inputs."""

    def _snap(self, x_norm: torch.Tensor) -> torch.Tensor:
        return TTFSGridSnapFunction.apply(x_norm, self.T)


_CLAMP_LEAK = 0.01
"""Minimum out-of-range gradient for DifferentiableClamp."""


def _clamp_bound_admissible(bound: torch.Tensor, x: torch.Tensor) -> bool:
    """Scalar (0-dim), or a channels-last vector matching x's last dim (>1 —
    scalars must arrive 0-dim so a stray [1]-shaped bound stays loud)."""
    if bound.dim() == 0:
        return True
    return (
        bound.dim() == 1
        and bound.numel() > 1
        and x.dim() >= 1
        and int(x.shape[-1]) == int(bound.numel())
    )


def _reduce_grad_to_bound(grad: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    if bound.dim() == 0:
        return grad.sum()
    return grad.reshape(-1, bound.shape[0]).sum(dim=0)


class DifferentiableClamp(Function):
    """Differentiable clamp with optional gradient flow to the bounds."""

    @staticmethod
    def forward(ctx, x, a, b):
        assert _clamp_bound_admissible(a, x) and _clamp_bound_admissible(b, x), (
            f"DifferentiableClamp expects scalar bounds or a channels-last "
            f"per-channel vector; got a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)} for x.shape={tuple(x.shape)}"
        )
        a_dev = a.to(x.device)
        b_dev = b.to(x.device)
        ctx.save_for_backward(x, a_dev, b_dev)
        return torch.clamp(x, a_dev, b_dev)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        x, a, b = ctx.saved_tensors
        below_grad = torch.clamp_min(torch.exp(x - a), _CLAMP_LEAK)
        above_grad = torch.clamp_min(torch.exp(b - x), _CLAMP_LEAK)
        grad_x = torch.where(
            x < a,
            below_grad,
            torch.where(x > b, above_grad, torch.ones_like(x)),
        )
        grad_a = _reduce_grad_to_bound(
            grad_output * (x < a).to(grad_output.dtype), a
        )
        grad_b = _reduce_grad_to_bound(
            grad_output * (x > b).to(grad_output.dtype), b
        )
        return grad_output * grad_x, grad_a, grad_b


