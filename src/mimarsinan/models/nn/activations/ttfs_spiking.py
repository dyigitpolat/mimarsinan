"""TTFS spike-based node + cycle-accurate forward (genuine spike-train KD)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
from mimarsinan.models.nn.activations.bias_mode import validate_bias_mode
from mimarsinan.models.nn.activations.lif import _StrictHeavisideFunction


def _heaviside_surrogate(pre: torch.Tensor, thresholding_mode: str, alpha: float = 2.0):
    """Spike on ``pre >= 0`` (inclusive) or ``pre > 0`` (strict), ATan-surrogate grad."""
    if thresholding_mode == "<":
        return _StrictHeavisideFunction.apply(pre, alpha)
    return 1.0 - _StrictHeavisideFunction.apply(-pre, alpha)


def _channel_broadcast_view(bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """View a per-output-channel ``bias`` so it broadcasts against ``x``: match the
    axis whose size equals the bias length, preferring the last axis (channel-last
    linear) and falling back to axis 1 (conv channel axis)."""
    n = int(bias.shape[0])
    dim = x.dim()
    if x.shape[-1] == n:
        axis = dim - 1
    elif dim >= 2 and x.shape[1] == n:
        axis = 1
    else:
        raise ValueError(
            f"TTFSActivation: bias length {n} matches no broadcastable axis of "
            f"input shape {tuple(x.shape)}"
        )
    shape = [1] * dim
    shape[axis] = n
    return bias.view(*shape)


class TTFSActivation(nn.Module):
    """Single-spike TTFS spike node (fire-once ramp), surrogate gradient.
    ``input_scale`` is the previous layer's ``activation_scale``, ``activation_scale``
    this layer's θ; ``bias`` is the effective (norm-folded) pre-activation bias."""

    _VALID_THRESHOLDING_MODES = ("<", "<=")

    def __init__(
        self,
        T: int,
        activation_scale: nn.Parameter | torch.Tensor | float,
        input_scale: nn.Parameter | torch.Tensor | float = 1.0,
        bias: torch.Tensor | None = None,
        thresholding_mode: str = "<=",
        firing_mode: str = "TTFS",
        encoding: bool = False,
        bias_mode: str = "on_chip",
    ):
        super().__init__()
        self.T = int(T)
        self.encoding = bool(encoding)
        # The two bias deliveries are dynamically equivalent, so bias_mode is threaded for config fidelity and never branches the forward.
        self.bias_mode = validate_bias_mode(bias_mode)
        if isinstance(activation_scale, (int, float)):
            activation_scale = nn.Parameter(
                torch.tensor(float(activation_scale)), requires_grad=False
            )
        self.activation_scale = activation_scale
        self.input_scale = input_scale
        self.set_bias(bias)
        if thresholding_mode not in self._VALID_THRESHOLDING_MODES:
            raise ValueError(
                f"TTFSActivation thresholding_mode must be one of "
                f"{self._VALID_THRESHOLDING_MODES!r}; got {thresholding_mode!r}"
            )
        self.thresholding_mode = thresholding_mode
        self.firing_mode = firing_mode
        self.surrogate_alpha: float = 2.0

        self._cycle_accurate_mode = False
        self._ramp_current: torch.Tensor | None = None
        self._membrane: torch.Tensor | None = None
        self._has_fired: torch.Tensor | None = None

    @property
    def activation_type(self) -> str:
        return "TTFS"

    def extra_repr(self) -> str:
        return f"T={self.T}, thresholding_mode={self.thresholding_mode!r}"

    def set_bias(self, bias: torch.Tensor | None) -> None:
        """Install the *effective* pre-activation bias this node subtracts. Stored as a
        plain attribute (never a registered parameter) so the segment policy can swap in
        a computed tensor and restore the raw ``layer.bias`` reference afterwards."""
        self._parameters.pop("_bias", None)
        object.__setattr__(self, "_bias", bias)

    def set_surrogate_alpha(self, a: float) -> None:
        """Set the ATan-surrogate sharpness (backward only; forward is unchanged)."""
        self.surrogate_alpha = float(a)

    def set_cycle_accurate(self, mode: bool) -> None:
        self._cycle_accurate_mode = bool(mode)
        self.reset_state()

    def reset_state(self) -> None:
        self._ramp_current = None
        self._membrane = None
        self._has_fired = None

    def _scale_values(self, x):
        sv = self._broadcast_scale(self.activation_scale, x)
        iv = self._broadcast_scale(self.input_scale, x)
        return sv, iv

    @staticmethod
    def _broadcast_scale(scale, x):
        """Clamp a scale and align a per-channel (multi-element) one to ``x``.

        Scalar scales stay scalar. A per-output-channel ``[C]`` scale (the
        ``theta_cotrain`` case) broadcasts correctly against a 2-D linear
        ``[B, C]`` input as-is, but against a conv ``[B, C, H, W]`` it would
        broadcast on the LAST dim (``W``) and crash when ``C != W``; for
        ``ndim > 2`` it is reshaped to broadcast on the channel axis (dim 1),
        matching ``_channel_broadcast_view`` used for the bias.
        """
        if not isinstance(scale, torch.Tensor):
            return max(float(scale), 1e-12)
        sv = scale.to(x.device, x.dtype).clamp(min=1e-12)
        if sv.numel() > 1 and x.dim() > 2:
            return _channel_broadcast_view(sv.reshape(-1), x)
        return sv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cycle_accurate_mode:
            scale_v, _ = self._scale_values(x)
            r = (torch.relu(x) / scale_v).clamp(0.0, 1.0)
            return TTFSStaircaseFunction.apply(r, self.T) * scale_v

        if self.encoding:
            scale_v, _ = self._scale_values(x)
            v_norm = (torch.relu(x) / scale_v).clamp(0.0, 1.0)
            if self._membrane is None:
                # Start one ramp-step low so the first crossing lands at the framework's TTFS spike index round(S*(1 - V/theta)).
                self._membrane = v_norm - (1.0 / float(self.T))
                self._has_fired = torch.zeros_like(x)
            self._membrane = self._membrane + (1.0 / float(self.T))
            pre = self._membrane - 1.0
            spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
            spike = spike_raw * (1.0 - self._has_fired)
            self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
            return spike

        scale_v, in_scale_v = self._scale_values(x)
        bias = self._bias
        if bias is not None:
            bias = bias.to(x.device, x.dtype)
            bias_b = _channel_broadcast_view(bias, x)
            weighted_raw = x - bias_b
            bias_norm = bias_b / scale_v
        else:
            weighted_raw = x
            bias_norm = 0.0
        weighted = weighted_raw * (in_scale_v / scale_v)

        if self._membrane is None:
            self._ramp_current = torch.zeros_like(x)
            self._membrane = torch.zeros_like(x)
            self._has_fired = torch.zeros_like(x)

        self._ramp_current = self._ramp_current + weighted
        self._membrane = self._membrane + self._ramp_current + bias_norm

        pre = self._membrane - 1.0
        spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
        spike = spike_raw * (1.0 - self._has_fired)
        self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
        return spike


def refresh_perceptron_bias_references(perceptron) -> None:
    """Re-point every ``TTFSActivation`` under ``perceptron`` at its live bias.
    Steps that REPLACE ``perceptron.layer`` (normalization fusion, bring_back_bias)
    call this to keep the stored resting-bias reference clean."""
    bias = getattr(perceptron.layer, "bias", None)
    for module in perceptron.modules():
        if isinstance(module, TTFSActivation):
            module.set_bias(bias)


def run_ttfs_cycle_accurate(model, x: torch.Tensor, T: int, forward_fn=None) -> torch.Tensor:
    """Drive ``model`` on a TTFS single-spike train and decode from fire timing;
    returns ``(B, ...)`` decoded values (latched single-spike count over the window
    divided by ``S``, the genuine TTFS value)."""
    from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_single_spike_train

    if forward_fn is None:
        forward_fn = model.forward

    nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
    for m in nodes:
        m.set_cycle_accurate(True)
    try:
        flat = x.reshape(x.shape[0], -1)
        enc = ttfs_single_spike_train(
            flat.detach().clamp(0, 1).cpu().numpy().astype("float64"), T
        )
        spike_train = torch.tensor(
            enc, dtype=x.dtype, device=x.device
        ).permute(2, 0, 1)
        spike_train = spike_train.reshape(T, *x.shape)

        latched = None
        accum = None
        for t in range(T):
            out_spike = forward_fn(spike_train[t])
            latched = out_spike if latched is None else torch.maximum(latched, out_spike)
            accum = latched if accum is None else accum + latched
        return accum / float(T)
    finally:
        for m in nodes:
            m.set_cycle_accurate(False)
            m.reset_state()
