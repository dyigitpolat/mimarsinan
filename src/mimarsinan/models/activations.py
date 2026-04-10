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
        grad_input = grad_output.clone()
        grad_input[input < 0] *= ctx.negative_slope
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
    @staticmethod
    def forward(ctx, x, a, b):
        a = a.clone().detach().to(x.device)
        b = b.clone().detach().to(x.device)

        ctx.save_for_backward(x, a, b)
        return torch.clamp(x, a, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        grad = torch.ones_like(x)
        below = x < a
        above = x > b
        grad[below] = torch.clamp_min(torch.exp(x[below] - a), _CLAMP_LEAK)
        grad[above] = torch.clamp_min(torch.exp(b - x[above]), _CLAMP_LEAK)
        return grad_output * grad, None, None
