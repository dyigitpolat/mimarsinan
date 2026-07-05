"""Single source of truth for segment-boundary encode/decode (torch + every simulator)."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.mapping.support.activation_scales import (
    perceptron_wrapped_activation_scale,
)
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.compute_boundary import (
    _resolve_lif_perceptron as _resolve_lif_perceptron,
    encode_compute_boundary,
    normalize_boundary_value,
)
from mimarsinan.spiking.spike_trains import (
    materialized_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)

__all__ = [
    "BoundaryConfig",
    "boundary_normalization_scales",
    "decode_segment_output",
    "decode_segment_output_torch",
    "encode_compute_boundary",
    "encode_segment_input",
    "normalize_boundary_slices_numpy",
    "normalize_boundary_slices_torch",
    "normalize_boundary_value",
    "normalize_ttfs_boundary_value",
    "warn_once_lossy_negative_clamp",
]

# W1-era name; same transcode (the TTFS spike time encodes the same wire rate).
normalize_ttfs_boundary_value = normalize_boundary_value


def boundary_normalization_scales(
    hybrid_mapping: HybridHardCoreMapping,
) -> dict[int, float | np.ndarray]:
    """Per-producer wire divisors for rate/LIF host state buffers:
    ``wire rate = buffer value / divisor``.

    Host ComputeOps run with ``(1, 1)`` scales on the rate/LIF paths, so an op
    whose module carries a perceptron ``activation_scale`` (plain encoders AND
    wrapper mappers) leaves its *value-domain* result in the state buffer; a
    structural op inherits its compute sources' divisors (scale-homogeneous
    pass-through). Neural producers (``counts / T``) and ScaleNormalizingWrapper
    ops are already wire-domain and are skipped.
    """
    divisors: dict[int, float | np.ndarray] = {}
    for stage in hybrid_mapping.stages:
        op = stage.compute_op
        if stage.kind != "compute" or op is None:
            continue
        module = (op.params or {}).get("module") if op.params else None
        if isinstance(module, ScaleNormalizingWrapper):
            continue
        divisor: float | np.ndarray
        wrapped = perceptron_wrapped_activation_scale(module)
        if wrapped is not None:
            divisor = wrapped
        else:
            src_divisors = [
                float(np.mean(divisors.get(int(src.node_id), 1.0)))
                for src in op.input_sources.flatten()
                if isinstance(src, IRSource) and src.node_id >= 0
            ]
            if not src_divisors:
                continue
            divisor = sum(src_divisors) / len(src_divisors)
        if isinstance(divisor, float) and abs(divisor - 1.0) < 1e-12:
            continue
        divisors[int(op.id)] = divisor
    return divisors


def normalize_boundary_slices_numpy(
    input_map,
    seg_input: np.ndarray,
    boundary_scales: dict[int, float | np.ndarray],
) -> np.ndarray:
    """Rescale each producer slice of an assembled segment input to the wire
    domain (numpy twin; empty divisors => identity, no copy)."""
    if not boundary_scales:
        return seg_input
    out = seg_input
    copied = False
    for s in input_map:
        divisor = boundary_scales.get(int(s.node_id))
        if divisor is None:
            continue
        if not copied:
            out = seg_input.copy()
            copied = True
        d = np.maximum(np.asarray(divisor, dtype=out.dtype).reshape(-1), 1e-12)
        out[:, s.offset : s.offset + s.size] /= (d if d.size > 1 else d[0])
    return out


def normalize_boundary_slices_torch(
    input_map,
    seg_input: torch.Tensor,
    boundary_scales: dict[int, float | np.ndarray],
) -> torch.Tensor:
    """Torch twin of :func:`normalize_boundary_slices_numpy`."""
    if not boundary_scales:
        return seg_input
    out = seg_input
    cloned = False
    for s in input_map:
        divisor = boundary_scales.get(int(s.node_id))
        if divisor is None:
            continue
        if not cloned:
            out = seg_input.clone()
            cloned = True
        d = torch.as_tensor(
            divisor, dtype=out.dtype, device=out.device,
        ).reshape(-1).clamp(min=1e-12)
        out[:, s.offset : s.offset + s.size] /= (d if d.numel() > 1 else d[0])
    return out


_warned_negative_boundary_stages: set = set()


def warn_once_lossy_negative_clamp(stage_name: str, seg_input: torch.Tensor) -> None:
    """An unshifted negative reaching the [0,1] boundary clamp is silent
    information loss; warn once per stage."""
    if stage_name in _warned_negative_boundary_stages:
        return
    if float(seg_input.min()) < -1e-6:
        _warned_negative_boundary_stages.add(stage_name)
        print(
            f"[segment_boundary] Warning: neural stage {stage_name!r} receives "
            f"negative boundary values (min {float(seg_input.min()):.4f}) that the "
            "[0,1] spike-encode clamp will drop; enable negative_value_shift to "
            "make this boundary lossless."
        )


def decode_segment_output(
    seg_out_spike_count: np.ndarray,
    simulation_length: int,
    *,
    dtype: npt.DTypeLike = np.float64,
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
    """LIF / rate decode (torch, batch-preserving): spike counts ``/ T``."""
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

    only_raw_input = (
        len(stage.input_map) == 1
        and int(stage.input_map[0].node_id) == -2
    )
    if not filled_ranges and only_raw_input:
        if config.spike_mode == "SpikeTrain":
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
