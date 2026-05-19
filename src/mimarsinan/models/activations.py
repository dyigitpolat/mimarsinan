"""Custom autograd activations and clamp."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def uniform_encode_to_spike_train(rate: torch.Tensor, T: int) -> torch.Tensor:
    """Encode a rate tensor ``(B, ...)`` ∈ [0, 1] to a ``(T, B, ...)`` spike train.

    Matches the chip's uniform encoder (``SpikingHybridCoreFlow.to_uniform_spikes``,
    ``_spike_encoding.uniform_rate_encode``): for each element, places
    ``N = round(rate * T)`` spikes at uniformly-spaced cycle indices.
    Saturates to one spike per cycle at rate 1.0.

    Used by cycle-accurate LIF Adaptation training to feed each network
    the same per-cycle pattern the chip will inject at deployment, so the
    membrane trajectories match.
    """
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
    """Drive a model's forward in cycle-accurate mode and return mean output.

    1. Uniform-encode the input rate ``(B, ...)`` to a ``(T, B, ...)``
       spike train, matching the chip's input encoder.
    2. Toggle every :class:`LIFActivation` in ``model`` into single-step
       mode and reset their membranes — once for the whole T-loop, not
       per cycle, so membranes accumulate across cycles like SpikingJelly.
    3. Call ``forward_fn(spike_train[t])`` for ``t in range(T)``. Each
       LIFActivation integrates one cycle of its pre-activation and
       emits a binary spike; the Mapper DAG cascades these through
       every node in topological order (Linear, Conv, BN, einops, …
       are shape-preserving and operate identically in either mode, so
       no model-side changes are required).
    4. Mean-reduce the T outputs to produce ``(B, num_classes)`` logits.

    This is the standard SpikingJelly cycle-accurate evaluation —
    multi-step IFNode dynamics expressed as T sequential single-step
    calls with persistent membrane state. The chip simulators
    (HCM/SCM/SANA-FE/nevresim/Lava) layer per-core scheduling on top
    of the same per-neuron dynamics; cycle-accurate training under
    SpikingJelly gives a model that those simulators can evaluate
    consistently because they share the LIF dynamics, threshold and
    reset semantics.

    On exit each LIF's mode is restored to multi-step so subsequent
    rate-mode forwards continue to work — the trained weights persist,
    only the forward dispatch is transient.

    ``forward_fn`` defaults to ``model.forward``. When the caller has
    monkey-patched ``model.forward`` to invoke this function (the
    typical tuner setup), pass the pre-patch bound forward explicitly
    to avoid the recursion.
    """
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
    """Nearest-neighbour quantiser ``round(x * T) / T`` with STE gradient.

    Matches the chip-side input encoder (``UniformSpikeGenerator`` in
    nevresim, ``to_uniform_spikes`` in ``SpikingUnifiedCoreFlow``) which
    generates ``N = round(T * x)`` spikes over ``T`` cycles.  Applied to
    encoding-layer perceptrons' input activation in LIF mode so training
    sees the same discretised input the chip will receive at runtime.
    """

    @staticmethod
    def forward(ctx, x, T):
        return torch.round(x * T) / T

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class ChipInputQuantizer(nn.Module):
    """STE round-to-chip-rate quantiser for encoding-layer inputs.

    Forward: clamps into ``[0, scale]``, quantises to the nearest
    ``scale / T`` step.  Backward is identity through the ``round`` —
    the surrounding ``clamp`` keeps gradients finite outside the
    representable range.  Used by ``LIFAdaptationStep`` to close the
    20 pp input-boundary gap between training and chip deployment.
    """

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


import math


class _StrictHeavisideFunction(torch.autograd.Function):
    """``(x > 0).to(x)`` forward with spikingjelly's ATan-surrogate backward.

    Used by :class:`StrictATanSurrogate` to make LIF training fire under the
    same strict ``v > v_threshold`` rule the chip path uses. The backward
    is delegated to spikingjelly's ``atan_backward`` so gradient flow is
    identical to the inclusive surrogate it replaces — only the forward
    boundary changes.

    Picklable because both this class and its consumer surrogate live at
    module scope; the pipeline cache and AdaptationManager serialize the
    live LIF model after LIF Adaptation runs.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Lazy import — spikingjelly only needs to load when a LIF model
        # actually runs a backward pass.
        from spikingjelly.activation_based import surrogate

        x, = ctx.saved_tensors
        return surrogate.atan_backward(grad_output, x, ctx.alpha)


class StrictATanSurrogate(nn.Module):
    """Strict-firing analogue of spikingjelly's ``ATan`` surrogate.

    spikingjelly's stock ``ATan`` returns ``heaviside(x) = (x >= 0).to(x)``
    in its spiking forward — i.e. inclusive ``v >= v_threshold`` firing.
    The chip path (nevresim's ``DefaultFirePolicy``,
    :class:`SpikingHybridCoreFlow` with ``thresholding_mode='<'``,
    SANA-FE's ``mimarsinan_soma`` in strict mode, ``SubtractiveLIFReset``
    in strict mode) fires at strict ``memb > threshold``. This module
    flips the heaviside boundary to ``(v > 1)`` so LIFActivation training
    matches the chip exactly; backward keeps the ATan gradient.

    Implements only the surface IFNode's torch backend needs
    (``forward(x)`` + ``set_spiking_mode`` + ``alpha`` / ``spiking``
    attrs), so it slots into ``neuron.IFNode(surrogate_function=...)``
    without subclassing spikingjelly. Standalone ``nn.Module`` keeps it
    picklable for the pipeline cache.
    """

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
        # Primitive (non-spiking) shape: same arctan form spikingjelly's
        # ATan uses, kept for parity with surrogate-only inspection paths.
        return (math.pi / 2 * self.alpha * x).atan() / math.pi + 0.5


class LIFActivation(nn.Module):
    """Multi-timestep integrate-and-fire activation with surrogate gradient.

    Simulates ``T`` cycles of subtractive-reset integrate-and-fire on a
    constant pre-activation input ``x = Linear(input) + bias``.

    The firing comparator is chosen via ``thresholding_mode`` to mirror the
    chip's configured policy:

    * ``"<="`` (default) — inclusive ``memb >= threshold``. Output rate is
      the classical ``floor(T * relu(x) / scale) / T * scale`` staircase
      (spikingjelly's stock ATan heaviside). The default deployment pipeline
      configures every chip path (Python sim, nevresim, lava, SANA-FE) to
      use ``<=`` as well, so training and deployment fire at the same
      boundary. Inclusive thresholding tolerates boundary cases ``v == 1``
      that strict ``<`` discards — under deep single-segment chip workloads
      where membrane trajectories are bursty from upstream NeuralCore output,
      this closes roughly 2 pp of NF→SCM gap.
    * ``"<"`` — strict ``memb > threshold``. Matches nevresim's hardcoded
      ``DefaultFirePolicy`` (``threshold < membrane_potential``).
      LIFActivation swaps the surrogate's forward heaviside for a strict
      ``(v > 1)`` variant so the boundary case ``v == 1`` does *not* fire —
      matching the chip exactly. Output rate is
      ``max(ceil(T * relu(x) / scale) - 1, 0) / T * scale`` clamped at
      ``scale``: the rate staircase loses its top step at each boundary
      ``x = k * scale / T`` exactly the way the chip does.

    Strict mode forces the ``torch`` IFNode backend (cupy's kernel hardcodes
    inclusive heaviside in its emitted CUDA code, so an in-Python surrogate
    can't override it); inclusive mode keeps the cupy fast path on CUDA hosts.

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

        # Lazy import keeps spikingjelly out of module-import paths that
        # never exercise LIF (e.g. TTFS-only test suites).
        from spikingjelly.activation_based import neuron, surrogate

        if thresholding_mode == "<":
            # Strict ``v > 1`` requires the in-Python surrogate; cupy's
            # heaviside emits ``>=`` and would silently revert to inclusive.
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

        # Cycle-accurate mode: when True, forward consumes one cycle's
        # pre-activation ``(B, ...)`` instead of a rate, runs the IFNode
        # single-step, and returns ``spike * scale`` for that cycle.
        # Membrane state persists across forward calls — the outer
        # T-loop in :func:`run_cycle_accurate` resets it once per
        # sample. Toggle via :meth:`set_cycle_accurate`. This is the
        # standard SpikingJelly single-step behaviour — no extra
        # inter-layer delay (the chip simulators encode their per-core
        # latency cascade separately at the HCM/IR level).
        self._cycle_accurate_mode = False

    @property
    def activation_type(self) -> str:
        return "LIF"

    def extra_repr(self) -> str:
        return f"T={self.T}, thresholding_mode={self.thresholding_mode!r}"

    def set_cycle_accurate(self, mode: bool) -> None:
        """Toggle single-step (cycle-accurate) vs multi-step (rate) forward.

        Cycle-accurate semantics: ``forward(x)`` treats ``x`` as one
        cycle's pre-activation and runs the IFNode single-step, with
        membrane state accumulating across successive forward calls.
        The caller (:func:`run_cycle_accurate`) resets the membrane
        between samples.

        Rate-mode semantics: ``forward(x)`` broadcasts ``x`` across ``T``
        cycles internally and returns the mean rate.

        Switching modes resets the IFNode's membrane and step mode so
        residual state from the previous mode cannot leak across.
        """
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
        """Return the actual ``(T, B, ...)`` LIF spike train (binary 0/1).

        The host-side encoding layer uses this so that downstream neural
        segments consume the *real* spike timing the LIF dynamics produce
        — not a uniform re-encoding of the rate. Re-encoding ``rate`` via
        ``to_uniform_spikes`` preserves the spike count but shifts the
        firing cycles, which propagates into a different membrane
        trajectory at every receiving NeuralCore.
        """
        spikes, _ = self._spikes_and_scale(x)
        return spikes

    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        """One cycle of LIF integration; returns ``spike * scale`` ``(B, ...)``.

        Used when :attr:`_cycle_accurate_mode` is on. Implements
        standard SpikingJelly single-step semantics: integrate the
        cycle's input into the persistent IFNode membrane, fire if
        ``v > threshold``, return the spike for this cycle. Membrane
        state accumulates across successive forward calls; the outer
        loop in :func:`run_cycle_accurate` resets the state once per
        sample.

        Multiplying the binary spike by ``scale`` keeps the inter-layer
        signal magnitude consistent with rate-mode (where the LIF
        output is ``mean(spikes) * scale``): the mean over T of
        cycle-accurate output equals the rate-mode output for constant
        input across T.
        """
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

        # Threshold is implicitly 1 inside the IFNode; normalising the input
        # here is equivalent to dividing W and b by ``activation_scale`` —
        # the same effective-weight formula SCM/HCM use.
        x_norm = F.relu(x) / safe_scale
        x_t = x_norm.unsqueeze(0).expand(self.T, *x_norm.shape).contiguous()

        # Multi-step semantics: ensure step_mode and membrane are clean,
        # regardless of any prior cycle-accurate run that may have left
        # the IFNode in single-step mode with accumulated v.
        self.if_node.step_mode = "m"
        functional.reset_net(self.if_node)
        spikes = self.if_node(x_t)  # (T, B, ...)
        return spikes, safe_scale
