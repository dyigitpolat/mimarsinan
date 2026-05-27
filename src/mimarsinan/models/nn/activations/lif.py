"""LIF activation and cycle-accurate helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_encode_to_spike_train(rate: torch.Tensor, T: int) -> torch.Tensor:
    """Encode rates in [0, 1] to a uniform-spaced spike train of length T."""
    from mimarsinan.spiking.spike_trains import uniform_spike_train

    return uniform_spike_train(rate, T)


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

    from mimarsinan.spiking.spike_trains import uniform_spike_train

    spike_train = uniform_spike_train(x, T)
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
        firing_mode: str = "Default",
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
        self.firing_mode = firing_mode

        from spikingjelly.activation_based import neuron, surrogate
        from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

        v_reset = FiringStrategyFactory.from_config(
            {"firing_mode": firing_mode, "thresholding_mode": thresholding_mode, "spiking_mode": "lif"}
        ).training_lif_v_reset()

        if thresholding_mode == "<":
            surrogate_fn = StrictATanSurrogate()
            preferred_backend = "torch"
        else:
            surrogate_fn = surrogate.ATan()
            preferred_backend = "cupy" if torch.cuda.is_available() else "torch"

        try:
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=v_reset,
                surrogate_function=surrogate_fn,
                step_mode="m",
                backend=preferred_backend,
            )
        except (ImportError, RuntimeError, AttributeError):
            self.if_node = neuron.IFNode(
                v_threshold=1.0,
                v_reset=v_reset,
                surrogate_function=surrogate_fn,
                step_mode="m",
                backend="torch",
            )

        self._cycle_accurate_mode = False
        self.use_cycle_accurate_trains = False

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
        if self.use_cycle_accurate_trains:
            from mimarsinan.spiking.spike_trains import lif_spike_train

            return lif_spike_train(x, self, self.T)
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

