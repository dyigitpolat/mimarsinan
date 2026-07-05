"""Wire-domain boundary transcode and spike emission for subsumed ComputeOp boundaries."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import _channel_broadcast_view
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.spike_trains import uniform_spike_train


def normalize_boundary_value(value: torch.Tensor, boundary_scale) -> torch.Tensor:
    """Wire-domain transcode: value-domain boundary tensor -> normalized [0, 1].

    The mode-agnostic segment-boundary wire contract shared by the torch NF walk,
    the hybrid executor and every deployed runner: a boundary spike train encodes
    ``value / boundary_scale`` (TTFS as a spike time, rate/LIF as a uniform train
    of ``round(rate * T)`` spikes), where ``boundary_scale`` is the producer's
    propagated boundary out-scale; the consumer's folded weights multiply the
    same scale back in. Encoding the raw value instead saturates values above
    the scale (bit-exact only at scale 1).
    """
    if isinstance(boundary_scale, torch.Tensor):
        scale = boundary_scale.to(value.device, value.dtype).clamp(min=1e-12)
        if scale.numel() > 1:
            scale = _channel_broadcast_view(scale.reshape(-1), value)
    else:
        scale = max(float(boundary_scale), 1e-12)
    return (value / scale).clamp(0.0, 1.0)


def _resolve_lif_perceptron(module) -> tuple[object, LIFActivation] | None:
    """Return ``(perceptron, lif)`` for a *plain* LIF-Perceptron encoding boundary,
    or ``None`` for wrappers (Conv2DPerceptronMapper) and non-LIF perceptrons."""
    if module is None:
        return None
    # Wrapper mappers expose a child ``perceptron`` distinct from themselves; their boundary stays rate-mode.
    if hasattr(module, "perceptron") and getattr(module, "perceptron") is not module:
        return None
    if not hasattr(module, "activation"):
        return None
    lif = unwrap_lif_activation(getattr(module, "activation", None))
    if lif is None:
        return None
    return module, lif


def encode_compute_boundary(
    *,
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    config: BoundaryConfig,
    hybrid_mapping: HybridHardCoreMapping,
) -> torch.Tensor | None:
    """Return the uniform ``(T, B, D)`` wire train for a plain-LIF-Perceptron
    boundary (``uniform_spike_train(clamp(value / theta))`` over the op's already
    computed value), or ``None`` for non-LIF-Perceptron boundaries."""
    del state_buffer_spikes, hybrid_mapping  # boundary emission needs only the op value
    if not config.use_cycle_accurate_trains:
        return None
    if not isinstance(op, ComputeOp):
        return None

    module = (op.params or {}).get("module") if op.params else None
    if module is None:
        return None
    resolved = _resolve_lif_perceptron(module)
    if resolved is None:
        return None
    _, lif = resolved

    value = state_buffer.get(int(op.id))
    if value is None:
        return None

    wire_rate = normalize_boundary_value(value, lif.activation_scale)
    return uniform_spike_train(wire_rate, config.simulation_length).to(
        config.compute_dtype
    )
