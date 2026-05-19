"""Custom autograd activations and clamp."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def uniform_encode_to_spike_train(rate: torch.Tensor, T: int) -> torch.Tensor:
    """Encode rates in [0, 1] to a uniform-spaced spike train of length T."""
    rate_c = rate.clamp(0.0, 1.0)
    N = torch.round(rate_c * T).to(torch.long)
    N_safe = N.clamp(min=1)
    spacing = T / N_safe.float()
    cycles = torch.arange(T, device=rate.device).reshape((T,) + (1,) * rate.ndim)
    mask_active = (N != 0) & (N != T)
    fire = (
        mask_active
        & (torch.floor(cycles / spacing) < N_safe)
        & (torch.floor(cycles % spacing) == 0)
    ).to(rate.dtype)
    fire = torch.where(
        (N == T).expand_as(fire),
        torch.ones_like(fire),
        fire,
    )
    return fire


def run_cycle_accurate(
    model,
    x: torch.Tensor,
    T: int,
    forward_fn=None,
) -> torch.Tensor:
    """Drive a model in cycle-accurate mode and return mean output."""
    from spikingjelly.activation_based import functional

    if forward_fn is None:
        forward_fn = model.forward

    spike_train = uniform_encode_to_spike_train(x, T)  # (T, B, ...)

    lif_modules = [m for m in model.modules() if isinstance(m, LIFActivation)]
    for m in lif_modules:
        m.set_cycle_accurate(True)
    functional.reset_net(model)
    try:
        outputs = [forward_fn(spike_train[t]) for t in range(T)]
    finally:
        for m in lif_modules:
            m.set_cycle_accurate(False)

    return torch.stack(outputs, dim=0).mean(dim=0)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)
        x_norm = (x / safe_scale).clamp(0.0, 1.0)
        x_q = RoundedStaircaseFunction.apply(x_norm, self.T)
        return x_q * safe_scale


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


import math


class _StrictHeavisideFunction(torch.autograd.Function):
    """Strict (x > 0) forward with spikingjelly ATan-surrogate backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        from spikingjelly.activation_based import surrogate

        x, = ctx.saved_tensors
        return surrogate.atan_backward(grad_output, x, ctx.alpha)


class StrictATanSurrogate(nn.Module):
    """Strict-firing analogue of spikingjelly's ATan surrogate."""

    def __init__(self, alpha: float = 2.0, spiking: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.spiking = bool(spiking)

    def set_spiking_mode(self, spiking: bool) -> None:
        self.spiking = bool(spiking)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, spiking={self.spiking}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spiking:
            return _StrictHeavisideFunction.apply(x, self.alpha)
        return (math.pi / 2 * self.alpha * x).atan() / math.pi + 0.5


class LIFActivation(nn.Module):
    """Multi-timestep integrate-and-fire activation with surrogate gradient."""

    _VALID_THRESHOLDING_MODES = ("<", "<=")

    def __init__(
        self,
        T: int,
        activation_scale: nn.Parameter | torch.Tensor | float,
        thresholding_mode: str = "<=",
    ):
        super().__init__()
        self.T = int(T)
        if isinstance(activation_scale, (int, float)):
            activation_scale = nn.Parameter(
                torch.tensor(float(activation_scale)), requires_grad=False
            )
        self.activation_scale = activation_scale

        if thresholding_mode not in self._VALID_THRESHOLDING_MODES:
            raise ValueError(
                f"LIFActivation thresholding_mode must be one of "
                f"{self._VALID_THRESHOLDING_MODES!r}; got {thresholding_mode!r}"
            )
        self.thresholding_mode = thresholding_mode

        from spikingjelly.activation_based import neuron, surrogate

        if thresholding_mode == "<":
            surrogate_fn = StrictATanSurrogate()
            preferred_backend = "torch"
        else:
            surrogate_fn = surrogate.ATan()
            preferred_backend = "cupy" if torch.cuda.is_available() else "torch"

        try:
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=None,  # subtractive (soft) reset — matches nevresim Default
                surrogate_function=surrogate_fn,
                step_mode="m",
                backend=preferred_backend,
            )
        except (ImportError, RuntimeError, AttributeError):
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=None,
                surrogate_function=surrogate_fn,
                step_mode="m",
                backend="torch",
            )

        self._cycle_accurate_mode = False

    @property
    def activation_type(self) -> str:
        return "LIF"

    def extra_repr(self) -> str:
        return f"T={self.T}, thresholding_mode={self.thresholding_mode!r}"

    def set_cycle_accurate(self, mode: bool) -> None:
        """Toggle single-step (cycle-accurate) vs multi-step (rate) forward."""
        from spikingjelly.activation_based import functional

        self._cycle_accurate_mode = bool(mode)
        self.if_node.step_mode = "s" if mode else "m"
        functional.reset_net(self.if_node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cycle_accurate_mode:
            return self._forward_single_step(x)
        spikes, safe_scale = self._spikes_and_scale(x)
        rate = spikes.mean(dim=0)
        return rate * safe_scale

    def forward_spiking(self, x: torch.Tensor) -> torch.Tensor:
        """Return the actual (T, B, ...) LIF spike train."""
        spikes, _ = self._spikes_and_scale(x)
        return spikes

    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        """One cycle of LIF integration; returns spike * scale."""
        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)

        x_norm = F.relu(x) / safe_scale
        spike = self.if_node(x_norm)
        return spike * safe_scale

    def _spikes_and_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | float]:
        from spikingjelly.activation_based import functional

        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            safe_scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        else:
            safe_scale = max(float(scale), 1e-12)

        # Threshold is 1 inside the IFNode; normalise input by activation_scale.
        x_norm = F.relu(x) / safe_scale
        x_t = x_norm.unsqueeze(0).expand(self.T, *x_norm.shape).contiguous()

        self.if_node.step_mode = "m"
        functional.reset_net(self.if_node)
        spikes = self.if_node(x_t)  # (T, B, ...)
        return spikes, safe_scale
