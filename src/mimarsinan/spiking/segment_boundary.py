"""Single source of truth for segment-boundary encode/decode (torch + every simulator)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    compute_input_state_with_shifts,
    compute_op_owns_scale_domain,
)
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir.deployment_dtype import SPIKING_ARITHMETIC
from mimarsinan.mapping.ir.gather_plan import gather_plan_for
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.phase_channels import stage_input_channels
from mimarsinan.mapping.support.activation_scales import RAW_INPUT, perceptron_wrapped_activation_scale
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.compute_boundary import (
    _resolve_lif_perceptron as _resolve_lif_perceptron,
    encode_compute_boundary,
    normalize_boundary_value,
)
from mimarsinan.spiking.spike_trains import rates_to_spike_train, uniform_spike_train

__all__ = [
    "BoundaryConfig",
    "boundary_normalization_scales",
    "decode_segment_output",
    "decode_segment_output_torch",
    "encode_compute_boundary",
    "encode_segment_input",
    "host_op_input_state",
    "normalize_boundary_slices_numpy",
    "normalize_boundary_slices_torch",
    "normalize_boundary_value",
    "normalize_ttfs_boundary_value",
    "warn_once_lossy_negative_clamp",
]

# Alias: the TTFS spike time encodes the same wire rate as the shared transcode.
normalize_ttfs_boundary_value = normalize_boundary_value


def boundary_normalization_scales(
    hybrid_mapping: HybridHardCoreMapping,
) -> dict[int, float | np.ndarray]:
    """Per-producer wire divisors for rate/LIF host state buffers:
    ``wire rate = buffer value / divisor``.

    The derived view of the stamped gauge tables:
    ``divisor = kappa_fold / kappa_buf`` (``node_activation_scales`` over
    ``node_buffer_scales``). Neural producers and ScaleNormalizingWrapper ops
    sit at the wire gauge (divisor 1, skipped); a wrapped host module leaves
    its value-domain result in the buffer (divisor == its theta); [U-IN] the raw
    input, where a core reads it, at its wire gauge over its unit buffer (a
    neural-stage read only: ``host_op_input_state``). Mappings predating the
    ``kappa_buf`` stamp fall back to the legacy wrapper walk.
    """
    if not getattr(hybrid_mapping, "node_buffer_scales", None):
        return _legacy_boundary_divisors(hybrid_mapping)
    fold_scales = hybrid_mapping.node_activation_scales
    buffer_scales = hybrid_mapping.node_buffer_scales
    divisors: dict[int, float | np.ndarray] = {}
    producers = [RAW_INPUT] if RAW_INPUT in fold_scales else []
    for stage in hybrid_mapping.stages:
        op = stage.compute_op
        if stage.kind != "compute" or op is None:
            continue
        module = (op.params or {}).get("module") if op.params else None
        if not isinstance(module, ScaleNormalizingWrapper):
            producers.append(int(op.id))
    for nid in producers:
        fold = fold_scales.get(nid, 1.0)
        buf = buffer_scales.get(nid, 1.0)
        divisor: float | np.ndarray
        if isinstance(fold, np.ndarray) or isinstance(buf, np.ndarray):
            divisor = np.asarray(fold, dtype=np.float64) / np.maximum(
                np.asarray(buf, dtype=np.float64), 1e-12,
            )
        else:
            divisor = float(fold) / max(float(buf), 1e-12)
            if abs(divisor - 1.0) < 1e-12:
                continue
        divisors[nid] = divisor
    return divisors


def _legacy_boundary_divisors(
    hybrid_mapping: HybridHardCoreMapping,
) -> dict[int, float | np.ndarray]:
    """Pre-stamp reconstruction: wrapped modules carry their theta; a plain op inherits the mean of its compute sources' divisors."""
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
    """Rescale each producer slice of an assembled segment input to the wire domain (numpy twin; empty divisors =>
    identity, no copy)."""
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
        out[:, s.offset : s.offset + s.size] /= (d[: s.size] if d.size > 1 else d[0])  # the slice's own elements
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
        out[:, s.offset : s.offset + s.size] /= (d[: s.size] if d.numel() > 1 else d[0])  # the slice's own elements
    return out


def host_op_input_state(op, state_buffer, wire_divisors, node_output_shifts):
    """The state buffer a rate/LIF host ComputeOp gathers from: the ONE host-op read every executor shares.

    An op that owns its scale domain (``ScaleNormalizingWrapper``, decoding each source at the producer's kappa_fold)
    reads every source at the wire gauge through the divisor table and per-producer transcode a neural-stage input
    reads (``boundary_normalization_scales``, ``normalize_boundary_slices_*``); any other op reads the buffers as
    stored, its output inheriting their gauge (``kappa_buf``). The producer shifts follow, in the neural-stage order.
    A host-run perceptron (a subsumed encoding layer) stores its raw value, so its domain-owning readers (a ResNet's
    first residual add) meet the wire currency here. [U-IN] The raw input is read as stored: the root has no wrap slot,
    so every host op decodes its raw buffer at one. Nothing to transcode => the buffer itself (no copy)."""
    view = state_buffer
    if wire_divisors and compute_op_owns_scale_domain(op):
        for nid in gather_plan_for(op).referenced_node_ids:
            buf = state_buffer.get(nid)
            if nid in wire_divisors and nid != RAW_INPUT and buf is not None:
                whole = [SegmentIOSlice(node_id=nid, offset=0, size=int(buf.shape[1]))]
                view = dict(view) if view is state_buffer else view
                view[nid] = (normalize_boundary_slices_torch(whole, buf, wire_divisors) if isinstance(buf, torch.Tensor)
                             else normalize_boundary_slices_numpy(whole, buf, wire_divisors))
    return compute_input_state_with_shifts(op, view, node_output_shifts)


_warned_negative_boundary_stages: set = set()


def warn_once_lossy_negative_clamp(stage_name: str, seg_input: torch.Tensor) -> None:
    """A negative value reaching the [0,1] boundary clamp is information loss;
    warn once per stage. With the always-on negative-value shift, this fires
    only for residual negatives BEYOND the calibrated shift (calibration-set
    coverage limit) — never for a config choice."""
    if stage_name in _warned_negative_boundary_stages:
        return
    if float(seg_input.min()) < -1e-6:
        _warned_negative_boundary_stages.add(stage_name)
        print(
            f"[segment_boundary] Warning: neural stage {stage_name!r} receives "
            f"negative boundary values (min {float(seg_input.min()):.4f}) beyond "
            "the calibrated negative-value shift; the [0,1] spike-encode clamp "
            "drops the residual (calibration-coverage limit — widen the "
            "calibration set if this recurs)."
        )


def decode_segment_output(
    seg_out_spike_count: np.ndarray,
    simulation_length: int,
    *,
    dtype: npt.DTypeLike = SPIKING_ARITHMETIC.boundary_np,
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


def _encode_uniform(rates, T, config, channels=None):
    return uniform_spike_train(
        rates, T, phase_dither=config.phase_dither, channels=channels,
    ).to(config.compute_dtype)


def _dither_channels(stage, hybrid_mapping, config: BoundaryConfig, width: int, cached) -> torch.Tensor | None:
    """[E1] Under a phase-dithered encode, the channel at unbounded geometry of every position of ``stage``'s input
    (``phase_channels``): the comb of a slice rotates by its IR neurons, never by where a cost-only decision lays it
    out; refuses by name a channel that has none. None for an undithered encode."""
    if not config.phase_dither:
        return None
    return stage_input_channels(hybrid_mapping, stage, width, cached=cached)


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
    """``(T, B, in_size)`` segment input, slice by slice by its kind: a cached train verbatim, every other slice by
    the encode of its value [U1 R4], a dithered comb keyed on its channels at unbounded geometry [E1]. Retimed level stages read the COUNT re-encode by definition (cached raw rhythm
    shifts combs at equal counts — the t0_04 s32 catch)."""
    if getattr(stage, "is_retimed_level", False):
        state_buffer_spikes = {}
    in_size = seg_input_rates_clamped.shape[1]
    spike_train = torch.zeros(
        T, batch_size, in_size, device=device, dtype=config.compute_dtype,
    )

    filled_ranges: list[tuple[int, int]] = []
    missing = False
    for s in stage.input_map:
        train = state_buffer_spikes.get(int(s.node_id))
        if train is None:
            missing = True
            continue
        spike_train[:, :, s.offset : s.offset + s.size] = (
            train[:, :, : s.size].to(config.compute_dtype)
        )
        filled_ranges.append((int(s.offset), int(s.offset) + int(s.size)))

    if not missing:
        return spike_train
    channels = _dither_channels(stage, hybrid_mapping, config, in_size, cached=state_buffer_spikes)
    if config.use_cycle_accurate_trains:
        encoded = _encode_uniform(seg_input_rates_clamped, T, config, channels)
    else:
        encoded = rates_to_spike_train(
            seg_input_rates_clamped, T, spike_mode=config.spike_mode,
            log_fallback=not filled_ranges, phase_dither=config.phase_dither, channels=channels,
        ).to(config.compute_dtype)
    for lo, hi in filled_ranges:
        encoded[:, :, lo:hi] = spike_train[:, :, lo:hi]
    return encoded
