"""Single source of truth for segment-boundary encode/decode.

Owns both directions of a neural-segment boundary, consumed identically by the
torch-side chip-aligned NF forward and every simulator (HCM/SCM/SANA-FE/Lava/
nevresim):

- **encode** — rates (+ optional cached LIF train) -> ``(T, B, in)`` spike train
  (:func:`encode_segment_input`; :func:`encode_compute_boundary` lives in
  ``compute_boundary.py`` and is re-exported here).
- **decode** — segment output spike counts -> inter-stage rates ``counts / T``
  (:func:`decode_segment_output`).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.compute_boundary import (
    _gather_op_input_train,
    _resolve_lif_perceptron,
    _run_perceptron_single_step_T,
    _shifted_rate_slice,
    encode_compute_boundary,
)

__all__ = [
    "BoundaryConfig",
    "decode_segment_output",
    "decode_segment_output_torch",
    "encode_compute_boundary",
    "encode_segment_input",
]


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
