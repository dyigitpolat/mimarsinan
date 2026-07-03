"""Cycle-accurate spike emission for subsumed LIF-Perceptron ComputeOp boundaries."""

from __future__ import annotations

from typing import Dict

import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.spike_trains import uniform_spike_train


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


_warned_negative_boundary_ops: set = set()


def _shifted_rate_slice(rate_slice, src, op, node_output_shifts):
    """Lift a producer rate slice by its ``node_output_shifts`` entry pre-clamp.

    An unshifted negative reaching the [0,1] clamp is silent information loss; warn once per op.
    """
    shift = (node_output_shifts or {}).get(int(src.node_id))
    if shift is not None:
        sh = torch.as_tensor(shift, dtype=rate_slice.dtype, device=rate_slice.device)
        rate_slice = rate_slice + sh.reshape(-1)[src.index]
    if float(rate_slice.min()) < -1e-6 and int(op.id) not in _warned_negative_boundary_ops:
        _warned_negative_boundary_ops.add(int(op.id))
        print(
            f"[segment_boundary] Warning: ComputeOp {op.id} ({op.name}) receives "
            f"negative boundary values (min {float(rate_slice.min()):.4f}) that the "
            "[0,1] spike-encode clamp will drop; enable negative_value_shift to "
            "make this boundary lossless."
        )
    return rate_slice


def _gather_op_input_train(
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    T: int,
    config: BoundaryConfig,
    *,
    node_output_shifts=None,
) -> torch.Tensor | None:
    """Assemble ``(T, B, in_size)`` input train for ``op``: cached trains take precedence."""
    sources = op.input_sources.flatten()
    if len(sources) == 0:
        return None

    raw_input = state_buffer.get(-2)
    sample_batch = None
    if raw_input is not None:
        sample_batch = int(raw_input.shape[0])
    else:
        for src in sources:
            if isinstance(src, IRSource) and src.node_id >= 0:
                t = state_buffer_spikes.get(int(src.node_id))
                if t is None:
                    rate = state_buffer.get(int(src.node_id))
                    if rate is None:
                        continue
                    sample_batch = int(rate.shape[0])
                    break
                sample_batch = int(t.shape[1])
                break
    if sample_batch is None:
        return None

    in_size = len(sources)
    if raw_input is not None:
        device = raw_input.device
    elif state_buffer_spikes:
        device = next(iter(state_buffer_spikes.values())).device
    elif state_buffer:
        device = next(iter(state_buffer.values())).device
    else:
        device = torch.device("cpu")
    out = torch.zeros(T, sample_batch, in_size, dtype=config.compute_dtype, device=device)

    for idx, src in enumerate(sources):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        if src.is_always_on():
            out[:, :, idx] = 1.0
            continue
        if src.is_input():
            if raw_input is None:
                return None
            rate_slice = raw_input[:, src.index].clamp(0.0, 1.0)
            out[:, :, idx] = uniform_spike_train(rate_slice, T).to(config.compute_dtype)
            continue
        cached = state_buffer_spikes.get(int(src.node_id))
        if cached is not None:
            out[:, :, idx] = cached[:, :, src.index].to(config.compute_dtype)
            continue
        rate = state_buffer.get(int(src.node_id))
        if rate is None:
            return None
        rate_slice = _shifted_rate_slice(
            rate[:, src.index], src, op, node_output_shifts,
        ).clamp(0.0, 1.0).to(config.compute_dtype)
        out[:, :, idx] = uniform_spike_train(rate_slice, T).to(config.compute_dtype)

    return out


def _run_perceptron_single_step_T(
    perceptron,
    lif: LIFActivation,
    input_train: torch.Tensor,
    op: ComputeOp,
    config: BoundaryConfig,
) -> torch.Tensor:
    """``T`` single-step forwards of the Perceptron; return ``(T, B, D)`` binary spikes."""
    from spikingjelly.activation_based import functional

    T = input_train.shape[0]
    B = input_train.shape[1]

    scale_attr = getattr(lif, "activation_scale", None)
    if isinstance(scale_attr, torch.Tensor):
        safe_scale = scale_attr.to(dtype=torch.float32).clamp(min=1e-12)
    else:
        safe_scale = torch.tensor(
            max(float(scale_attr) if scale_attr is not None else 1.0, 1e-12),
            dtype=torch.float32,
        )

    was_ca = lif._cycle_accurate_mode
    lif.set_cycle_accurate(True)
    functional.reset_net(lif.if_node)
    try:
        outputs = []
        for t in range(T):
            inp = input_train[t]
            if op.input_shape is not None:
                inp = inp.reshape(B, *op.input_shape)
            out_t = perceptron(inp.to(torch.float32))
            outputs.append((out_t / safe_scale).reshape(B, -1))
        stacked = torch.stack(outputs, dim=0)
    finally:
        lif.set_cycle_accurate(was_ca)

    return stacked.to(config.compute_dtype)


def encode_compute_boundary(
    *,
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    config: BoundaryConfig,
    hybrid_mapping: HybridHardCoreMapping,
) -> torch.Tensor | None:
    """Return ``(T, B, D)`` spike train for ``op``, or ``None`` for non-LIF-Perceptron boundaries."""
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
    perceptron, lif = resolved

    input_train = _gather_op_input_train(
        op, state_buffer, state_buffer_spikes, config.simulation_length, config,
        node_output_shifts=getattr(hybrid_mapping, "node_output_shifts", None),
    )
    if input_train is None:
        return None

    return _run_perceptron_single_step_T(perceptron, lif, input_train, op, config)
