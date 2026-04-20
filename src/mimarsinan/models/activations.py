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
    @staticmethod
    def forward(ctx, x, Tq):
        return torch.floor(x * Tq) / Tq

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


_CLAMP_LEAK = 0.01
"""Minimum out-of-range gradient for DifferentiableClamp.

The backward pass uses a *floored exponential*:
- Inside ``[a, b]``: gradient = 1.0  (full STE).
- Outside: ``max(exp(-distance_to_boundary), _CLAMP_LEAK)``.

Near the boundary the gradient smoothly decays from 1.0 (reproducing the
original exponential backward), implicitly regularising weights to produce
activations that stay within the clamp range.  Far from the boundary the
exponential would vanish, so the floor prevents gradient death.  This keeps
the trained weights compatible with the spiking simulation's effective-weight
formula ``W_eff = per_input_scales * W / activation_scale``.
"""


class DifferentiableClamp(Function):
    """Differentiable clamp with optional gradient flow to the bounds.

    Forward: ``out = clamp(x, a, b)``.

    Backward wrt ``x`` uses the floored-exponential STE: full gradient
    inside ``[a, b]``, smoothly-decaying (floored) gradient outside. This
    gently regularises weights to stay within the clamp range without
    killing gradients.

    Backward wrt ``a`` and ``b``: a gradient is returned whenever the
    input tensor is not ``detach()``ed. When the saturating side is
    active (``x < a`` for ``a``, ``x > b`` for ``b``) the clamp output is
    exactly the bound, so ``d output / d bound = 1``; elsewhere the
    gradient is zero. This is what lets ``ClampDecorator`` use a learnable
    scale parameter for the upper bound.
    """

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
        # For a/b: gradient is 1 where saturation pins output to the
        # bound, 0 elsewhere. Sum over the input tensor (bounds are scalars).
        grad_a = (grad_output * (x < a).to(grad_output.dtype)).sum()
        grad_b = (grad_output * (x > b).to(grad_output.dtype)).sum()
        return grad_output * grad_x, grad_a, grad_b
