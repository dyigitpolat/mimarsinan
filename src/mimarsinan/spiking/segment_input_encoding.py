"""Segment-input spike-train assembly at the boundary (cached trains + re-encode)."""

from __future__ import annotations

import logging
from typing import Dict

import torch

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.spike_trains import (
    rates_to_spike_train,
    uniform_spike_train,
)


def _encode_uniform(rates, T, config):
    return uniform_spike_train(
        rates, T, phase_dither=config.phase_dither,
    ).to(config.compute_dtype)


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
    """``(T, B, in_size)`` segment input. Cached trains take precedence EXCEPT
    for retimed level stages, whose input is the COUNT re-encode by definition
    (cached raw rhythm shifts combs at equal counts — the t0_04 s32 catch)."""
    if getattr(stage, "is_retimed_level", False):
        state_buffer_spikes = {}
    in_size = seg_input_rates_clamped.shape[1]
    spike_train = torch.zeros(
        T, batch_size, in_size, device=device, dtype=config.compute_dtype,
    )

    filled_ranges: list[tuple[int, int]] = []
    missing_slices: list[tuple[int, int, int]] = []
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
                seg_input_rates_clamped, T, spike_mode=config.spike_mode,
                log_fallback=True, phase_dither=config.phase_dither,
            ).to(config.compute_dtype)
        if not missing_slices:
            return spike_train
        encoded = rates_to_spike_train(
            seg_input_rates_clamped, T, spike_mode=config.spike_mode,
            log_fallback=False, phase_dither=config.phase_dither,
        ).to(config.compute_dtype)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    only_raw_input = (
        len(stage.input_map) == 1
        and int(stage.input_map[0].node_id) == -2
    )
    if not filled_ranges and only_raw_input:
        if config.spike_mode == "SpikeTrain":
            return _encode_uniform(seg_input_rates_clamped, T, config)
        return _encode_uniform(seg_input_rates_clamped, T, config)

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
        logging.getLogger("mimarsinan.spiking.segment_boundary").debug(
            "stage %r: rate-only boundary at non-raw inputs %s — uniform-encoding.",
            stage.name, [m[0] for m in non_raw_missing],
        )
        return _encode_uniform(seg_input_rates_clamped, T, config)

    if raw_missing:
        encoded = _encode_uniform(seg_input_rates_clamped, T, config)
        for lo, hi in filled_ranges:
            encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
        return encoded

    return spike_train
