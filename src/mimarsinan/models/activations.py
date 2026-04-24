"""Custom autograd activations and clamp."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LIFActivation(nn.Module):
    """Multi-timestep integrate-and-fire activation with surrogate gradient.

    Simulates ``T`` cycles of subtractive-reset integrate-and-fire on a
    constant pre-activation input ``x = Linear(input) + bias``.  The output
    rate is exactly ``floor(T * relu(x) / scale) / T * scale`` — the same
    discretisation a ``SpikingUnifiedCoreFlow`` rate-mode simulation with
    ``firing_mode='Default'`` produces at the neuron level, so training
    dynamics match deployment.

    Backward uses SpikingJelly's ATan surrogate so gradients flow through
    the otherwise non-differentiable spike function.  This lets the
    downstream Weight Quantization / Normalization Fusion steps continue to
    rely on autograd exactly as they did when the base activation was
    ``LeakyGradReLU``.

    ``activation_scale`` is held as a reference to the owning Perceptron's
    ``nn.Parameter`` so subsequent scale updates propagate without extra
    wiring, and the IR threshold derived from the same parameter stays
    aligned with training semantics.
    """

    def __init__(self, T: int, activation_scale: nn.Parameter | torch.Tensor | float):
        super().__init__()
        self.T = int(T)
        if isinstance(activation_scale, (int, float)):
            activation_scale = nn.Parameter(
                torch.tensor(float(activation_scale)), requires_grad=False
            )
        self.activation_scale = activation_scale

        # Lazy import keeps spikingjelly out of module-import paths that
        # never exercise LIF (e.g. TTFS-only test suites).
        from spikingjelly.activation_based import neuron, surrogate

        # cupy backend is the CUDA-accelerated fast path from the
        # spikingjelly example; fall back to the torch backend when cupy
        # isn't built against the host's CUDA/python combo.
        backend = "cupy" if torch.cuda.is_available() else "torch"
        try:
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=None,  # subtractive (soft) reset — matches nevresim Default
                surrogate_function=surrogate.ATan(),
                step_mode="m",
                backend=backend,
            )
        except (ImportError, RuntimeError, AttributeError):
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=None,
                surrogate_function=surrogate.ATan(),
                step_mode="m",
                backend="torch",
            )

    @property
    def activation_type(self) -> str:
        return "LIF"

    def extra_repr(self) -> str:
        return f"T={self.T}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from spikingjelly.activation_based import functional

        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)

        # Threshold is implicitly 1 inside the IFNode; normalising the input
        # here is equivalent to dividing W and b by ``activation_scale`` —
        # the same effective-weight formula SCM/HCM use.
        x_norm = F.relu(x) / safe_scale
        x_t = x_norm.unsqueeze(0).expand(self.T, *x_norm.shape).contiguous()

        # Reset membrane state between batches to avoid cross-batch leakage.
        functional.reset_net(self.if_node)
        spikes = self.if_node(x_t)  # (T, B, ...)
        rate = spikes.mean(dim=0)
        return rate * safe_scale
