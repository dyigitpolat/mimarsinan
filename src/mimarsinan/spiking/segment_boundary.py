"""Single source of truth for segment-boundary encode/decode (torch + every simulator)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
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
from mimarsinan.spiking.segment_input_encoding import encode_segment_input

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
    its value-domain result in the buffer (divisor == its theta). Mappings
    predating the ``kappa_buf`` stamp fall back to the legacy wrapper walk.
    """
    if not getattr(hybrid_mapping, "node_buffer_scales", None):
        return _legacy_boundary_divisors(hybrid_mapping)
    fold_scales = hybrid_mapping.node_activation_scales
    buffer_scales = hybrid_mapping.node_buffer_scales
    divisors: dict[int, float | np.ndarray] = {}
    for stage in hybrid_mapping.stages:
        op = stage.compute_op
        if stage.kind != "compute" or op is None:
            continue
        module = (op.params or {}).get("module") if op.params else None
        if isinstance(module, ScaleNormalizingWrapper):
            continue
        fold = fold_scales.get(int(op.id), 1.0)
        buf = buffer_scales.get(int(op.id), 1.0)
        divisor: float | np.ndarray
        if isinstance(fold, np.ndarray) or isinstance(buf, np.ndarray):
            divisor = np.asarray(fold, dtype=np.float64) / np.maximum(
                np.asarray(buf, dtype=np.float64), 1e-12,
            )
        else:
            divisor = float(fold) / max(float(buf), 1e-12)
            if abs(divisor - 1.0) < 1e-12:
                continue
        divisors[int(op.id)] = divisor
    return divisors


def _legacy_boundary_divisors(
    hybrid_mapping: HybridHardCoreMapping,
) -> dict[int, float | np.ndarray]:
    """Pre-stamp reconstruction: wrapped modules carry their theta; a plain op
    inherits the mean of its compute sources' divisors."""
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
