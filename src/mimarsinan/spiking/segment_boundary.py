"""Single source of truth for segment-boundary encode/decode.

Owns both directions of a neural-segment boundary, consumed identically by the
torch-side chip-aligned NF forward and every simulator (HCM/SCM/SANA-FE/Lava/
nevresim):

- **encode** — rates (+ optional cached LIF train) -> ``(T, B, in)`` spike train
  (:func:`encode_segment_input`, :func:`encode_compute_boundary`).
- **decode** — segment output spike counts -> inter-stage rates ``counts / T``
  (:func:`decode_segment_output`).

The :class:`SegmentBoundary` descriptor is the declarative object both sides read
so they cannot drift. Its Round-2 fields (``shift``, ``placement``,
``spike_generation_mode``) are present but default to the identity; they are the
reserved seams for negative-value shifting and the subsume-vs-offload switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.lif_utils import unwrap_lif_activation


@dataclass
class BoundaryConfig:
    """Config for boundary encode/decode.

    ``negative_shift`` (per-boundary positive-domain shift of negative-producing
    ComputeOps) and ``spike_generation_mode`` (subsume vs offload selection) are the
    Round-2 toggles.
    """

    simulation_length: int
    spiking_mode: str
    cycle_accurate: bool
    spike_mode: str = "Uniform"
    thresholding_mode: str = "<="
    firing_mode: str = "Default"
    compute_dtype: torch.dtype = torch.float64
    negative_shift: bool = False
    spike_generation_mode: str = "Uniform"

    @property
    def use_cycle_accurate_trains(self) -> bool:
        return self.spiking_mode == "lif" and self.cycle_accurate


@dataclass(frozen=True)
class SegmentBoundary:
    """Declarative description of how a producer's output crosses into a consumer.

    Round-1 fields drive encode/decode; Round-2 fields are reserved seams that
    default to the identity (``shift=0.0``, ``placement="subsume"``).
    """

    producer_kind: str            # "neural" | "compute" | "input"
    consumer_node_id: int
    encode_mode: str              # "cached_lif" | "uniform" | "verbatim"
    scale: float = 1.0
    decode: str = "count_over_T"  # counts / T
    # --- Round 2a (negative-value shift) — declared, default-inert ---
    shift: Union[np.ndarray, float] = 0.0
    # --- Round 2b (subsume vs offload) — declared, default-inert ---
    placement: str = "subsume"
    spike_generation_mode: str = "Uniform"


def decode_segment_output(
    seg_out_spike_count: np.ndarray,
    simulation_length: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """LIF / rate decode (numpy inter-stage): spike counts ``/ T``, flattened to ``(1, N)``."""
    t = max(int(simulation_length), 1)
    return (
        np.asarray(seg_out_spike_count, dtype=dtype).reshape(1, -1)
        / np.asarray(t, dtype=dtype)
    )


def decode_segment_output_torch(
    spike_counts: torch.Tensor, simulation_length: int
) -> torch.Tensor:
    """LIF / rate decode (torch, batch-preserving): spike counts ``/ T``.

    Same ÷T semantics as :func:`decode_segment_output`; keeps the leading batch
    dimension so the torch hybrid forward and the chip-aligned NF forward share it.
    """
    t = max(int(simulation_length), 1)
    return spike_counts / float(t)


def _resolve_lif_perceptron(module) -> tuple[object, LIFActivation] | None:
    """Return ``(perceptron, lif)`` for a *plain* LIF-Perceptron encoding boundary,
    or ``None`` for wrappers (Conv2DPerceptronMapper) and non-LIF perceptrons."""
    if module is None:
        return None
    # Wrapper mappers (e.g. Conv2DPerceptronMapper) expose a child ``perceptron``
    # whose forward differs from the wrapper's; their boundary stays rate-mode.
    if hasattr(module, "perceptron") and getattr(module, "perceptron") is not module:
        return None
    if not hasattr(module, "activation"):
        return None
    lif = unwrap_lif_activation(getattr(module, "activation", None))
    if lif is None:
        return None
    return module, lif


def _gather_op_input_train(
    op: ComputeOp,
    state_buffer: Dict[int, torch.Tensor],
    state_buffer_spikes: Dict[int, torch.Tensor],
    T: int,
    config: BoundaryConfig,
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
    device = (raw_input.device if raw_input is not None
              else next(iter(state_buffer_spikes.values())).device)
    out = torch.zeros(T, sample_batch, in_size, dtype=config.compute_dtype, device=device)

    from mimarsinan.spiking.spike_trains import uniform_spike_train

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
        rate_slice = rate[:, src.index].clamp(0.0, 1.0).to(config.compute_dtype)
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
    )
    if input_train is None:
        return None

    return _run_perceptron_single_step_T(perceptron, lif, input_train, op, config)


def encode_segment_input(
    stage: HybridStage,
    seg_input_rates_clamped: torch.Tensor,
    state_buffer_spikes: Dict[int, torch.Tensor],
    *,
    config: BoundaryConfig,
    hybrid_mapping: HybridHardCoreMapping,
    T: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``(T, B, in_size)`` segment input. Cached trains take precedence; the rest
    falls back to uniform encoding of the segment input rates."""
    from mimarsinan.spiking.spike_trains import rates_to_spike_train, uniform_spike_train

    in_size = seg_input_rates_clamped.shape[1]
    spike_train = torch.zeros(
        T, batch_size, in_size, device=device, dtype=config.compute_dtype,
    )

    filled_ranges: list[tuple[int, int]] = []
    missing_slices: list[tuple[int, int, int]] = []  # (node_id, offset, size)
    for s in stage.input_map:
        train = state_buffer_spikes.get(int(s.node_id))
        if train is None:
            missing_slices.append((int(s.node_id), int(s.offset), int(s.size)))
            continue
        spike_train[:, :, s.offset : s.offset + s.size] = (
            train[:, :, : s.size].to(config.compute_dtype)
        )
        filled_ranges.append((int(s.offset), int(s.offset) + int(s.size)))

    if not config.use_cycle_accurate_trains:
        if not filled_ranges:
            return rates_to_spike_train(
                seg_input_rates_clamped,
                T,
                spike_mode=config.spike_mode,
                log_fallback=True,
            ).to(config.compute_dtype)
        if not missing_slices:
            return spike_train
        encoded = rates_to_spike_train(
            seg_input_rates_clamped,
            T,
            spike_mode=config.spike_mode,
            log_fallback=False,
        ).to(config.compute_dtype)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    # Cycle-accurate path: uniform-encode whatever isn't cached; cached overlays.
    only_raw_input = (
        len(stage.input_map) == 1
        and int(stage.input_map[0].node_id) == -2
    )
    if not filled_ranges and only_raw_input:
        if config.spike_mode == "SpikeTrain":
            from mimarsinan.spiking.spike_trains import materialized_spike_train

            return materialized_spike_train(
                seg_input_rates_clamped, T,
            ).to(config.compute_dtype)
        return uniform_spike_train(
            seg_input_rates_clamped, T,
        ).to(config.compute_dtype)

    non_raw_missing = [m for m in missing_slices if m[0] != -2]
    raw_missing = [m for m in missing_slices if m[0] == -2]

    if non_raw_missing:
        if filled_ranges:
            raise ValueError(
                f"encode_segment_input: stage {stage.name!r} has cached spike "
                f"trains for some inputs but is missing spike train(s) for node_id(s) "
                f"{[m[0] for m in non_raw_missing]}. Every non-raw input slice must have a "
                f"cached train (cycle-accurate parity)."
            )
        import logging
        logging.getLogger("mimarsinan.spiking.segment_boundary").debug(
            "stage %r: rate-only boundary at non-raw inputs %s — uniform-encoding.",
            stage.name, [m[0] for m in non_raw_missing],
        )
        return uniform_spike_train(seg_input_rates_clamped, T).to(config.compute_dtype)

    if raw_missing:
        encoded = uniform_spike_train(seg_input_rates_clamped, T).to(config.compute_dtype)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    return spike_train
