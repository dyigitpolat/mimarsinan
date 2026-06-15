"""TTFS spike-based node + cycle-accurate forward (genuine spike-train KD).

The cascaded ``ttfs_cycle_based`` deployment is a single-spike, ramp-integrate,
fire-once simulation. To fine-tune *through* that (not a pointwise analytical
surrogate), KD must run the model on actual spike trains with a spike node whose
gradient flows through the per-cycle dynamics — exactly the role spikingjelly's
``IFNode`` plays for LIF (see ``lif.py``), which TTFS lacked.

``TTFSActivation`` is that node: in single-step (cycle-accurate) mode it maintains
``ramp_current`` / ``membrane`` / ``has_fired`` per neuron, integrates the ramp
each cycle, and fires **once** through a surrogate Heaviside. Bias is held by the
node (separated from the weighted spike input) and added **linearly** (``b·t``),
not double-integrated. Weights/threshold are mapped into the normalised rate
domain via ``input_scale`` and ``activation_scale`` (= θ), matching the cascade.

``run_ttfs_cycle_accurate`` encodes the model input as a TTFS single-spike train,
propagates it cycle-by-cycle through the model, and decodes each output from its
fire timing (``count_of_latched_output / S``) — the genuine TTFS decode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.lif import (
    StrictATanSurrogate,
    _StrictHeavisideFunction,
)


def _heaviside_surrogate(pre: torch.Tensor, thresholding_mode: str, alpha: float = 2.0):
    """Spike on ``pre >= 0`` (inclusive) or ``pre > 0`` (strict), ATan-surrogate grad."""
    if thresholding_mode == "<":
        return _StrictHeavisideFunction.apply(pre, alpha)          # pre > 0
    # inclusive: spike on pre >= 0  ==  not (-pre > 0)
    return 1.0 - _StrictHeavisideFunction.apply(-pre, alpha)


def _channel_broadcast_view(bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """View a per-output-channel ``bias`` so it broadcasts against ``x``.

    Linear perceptrons emit channel-last tensors (``(B, F)`` / ``(B, T, F)``);
    Conv1D/Conv2D mappers emit channel-first NCL / NCHW. Match the axis whose
    size equals the bias length, preferring the last axis (channel-last
    convention) and falling back to axis 1 (conv channel axis).
    """
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

    ``input_scale`` = scale of the incoming activations (previous layer's
    ``activation_scale``); ``activation_scale`` = this layer's output scale (θ).
    ``bias`` is the *effective* additive constant of the owning perceptron's
    pre-activation path (``effective_preactivation_bias``: the norm-folded bias
    when a normalization is present, else ``layer.bias``), held here so it can
    be added linearly and removed from the post-norm pre-activation. At rest it
    stores the raw ``layer.bias`` reference; ``TtfsSegmentPolicy`` installs the
    effective bias for the duration of each cascade drive.
    """

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
        # encoding=True: value->spike entry of a segment (takes an ideal real value,
        #   emits a TTFS single-spike train via charge V/theta + bias ramp theta/S).
        # encoding=False: cascade neuron (ramps from arriving single spikes).
        self.encoding = bool(encoding)
        # bias_mode records the deployment's physical bias delivery (config-driven,
        # from has_bias / IRMapping.hardware_bias). The two deliveries are
        # *dynamically equivalent* — "on_chip" adds the per-neuron bias to the
        # membrane each cycle; "param_encoded" delivers it as an always-on axon that
        # (for single-spike TTFS) fires once at the core's local window start and is
        # ramp-integrated. Both give cumulative membrane bias·(t_local+1), so this
        # node has a single bias implementation and bias_mode does not branch the
        # forward; it is threaded for config fidelity and consumed by the simulators.
        from mimarsinan.models.nn.activations.bias_mode import validate_bias_mode

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
        # Backward-only surrogate sharpness: shapes the ATan gradient of the fire
        # Heaviside, never the exact ``pre > 0`` forward (see _heaviside_surrogate).
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
        """Install the *effective* pre-activation bias this node subtracts.

        Stored as a plain attribute (never a registered parameter/submodule) so
        the segment policy can swap in a computed tensor — e.g. the norm-folded
        ``effective_preactivation_bias`` — for the duration of a cascade drive
        and restore the raw ``layer.bias`` reference afterwards.
        """
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
        scale = self.activation_scale
        sv = (scale.to(x.device, x.dtype).clamp(min=1e-12)
              if isinstance(scale, torch.Tensor) else max(float(scale), 1e-12))
        ins = self.input_scale
        iv = (ins.to(x.device, x.dtype).clamp(min=1e-12)
              if isinstance(ins, torch.Tensor) else max(float(ins), 1e-12))
        return sv, iv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cycle_accurate_mode:
            # Pointwise analytical value (deployment staircase kernel) for eval /
            # downstream calibration when not driven by run_ttfs_cycle_accurate.
            from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction
            scale_v, _ = self._scale_values(x)
            r = (torch.relu(x) / scale_v).clamp(0.0, 1.0)
            return TTFSStaircaseFunction.apply(r, self.T) * scale_v

        if self.encoding:
            # Value->spike entry: ``x`` is the ideal real pre-activation V (bias
            # included), constant across cycles. Charge V/theta, add the bias ramp
            # theta/S each cycle, fire once at k_fire = S*(1 - V/theta) -> a single
            # TTFS spike encoding V (Stanojevic input encoding). Gradient flows
            # through the charge (relu(V)/theta) and the surrogate fire.
            # A subsumed encoding layer is a host ComputeOp (bias folded analytically),
            # so this charge is bias-mode-agnostic: ``bias_mode`` does not branch here.
            scale_v, _ = self._scale_values(x)
            v_norm = (torch.relu(x) / scale_v).clamp(0.0, 1.0)
            if self._membrane is None:
                # Start one ramp-step low so the first crossing lands at the
                # framework's TTFS spike index round(S*(1 - V/theta)).
                self._membrane = v_norm - (1.0 / float(self.T))
                self._has_fired = torch.zeros_like(x)
            self._membrane = self._membrane + (1.0 / float(self.T))
            pre = self._membrane - 1.0
            spike_raw = _heaviside_surrogate(pre, self.thresholding_mode, alpha=self.surrogate_alpha)
            spike = spike_raw * (1.0 - self._has_fired)
            self._has_fired = (self._has_fired + spike.detach()).clamp(max=1.0)
            return spike

        # ``x`` is the perceptron pre-activation (W @ spikes + b) for this cycle.
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
        # Map raw weighted input into the normalised rate domain (θ_eff = 1):
        # W_eff·spike = (in_scale/θ)·(W·spike).
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

    ``TTFSActivation`` stores ``perceptron.layer.bias`` by reference as its
    resting (pickle-safe) bias; steps that REPLACE ``perceptron.layer``
    (normalization fusion, bring_back_bias) call this to keep the stored
    contract clean. Correctness no longer depends on it: ``TtfsSegmentPolicy``
    recomputes the norm-folded ``effective_preactivation_bias`` fresh on every
    cascade drive (the 2026-06-07/08 offload incidents — a stale reference, then
    a raw-instead-of-effective bias — are both structurally closed by that).
    """
    bias = getattr(perceptron.layer, "bias", None)
    for module in perceptron.modules():
        if isinstance(module, TTFSActivation):
            module.set_bias(bias)


def run_ttfs_cycle_accurate(model, x: torch.Tensor, T: int, forward_fn=None) -> torch.Tensor:
    """Drive ``model`` on a TTFS single-spike train and decode from fire timing.

    Returns ``(B, ...)`` decoded values: each output's count of latched single-spike
    output over the window divided by ``S`` (the genuine TTFS value), then scaled by
    the output activation scale via the per-layer nodes (which return [0,1] spikes).
    """
    from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_single_spike_train

    if forward_fn is None:
        forward_fn = model.forward

    nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
    for m in nodes:
        m.set_cycle_accurate(True)
    try:
        # Encode model input as a single TTFS spike per feature (N, D, S) -> (S, N, D).
        flat = x.reshape(x.shape[0], -1)
        enc = ttfs_single_spike_train(
            flat.detach().clamp(0, 1).cpu().numpy().astype("float64"), T
        )
        spike_train = torch.tensor(
            enc, dtype=x.dtype, device=x.device
        ).permute(2, 0, 1)  # (S, N, D)
        spike_train = spike_train.reshape(T, *x.shape)

        latched = None
        accum = None
        for t in range(T):
            out_spike = forward_fn(spike_train[t])  # (B, out) single spike (0/1)
            latched = out_spike if latched is None else torch.maximum(latched, out_spike)
            accum = latched if accum is None else accum + latched
        return accum / float(T)
    finally:
        for m in nodes:
            m.set_cycle_accurate(False)
            m.reset_state()
