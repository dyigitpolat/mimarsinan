"""Shared compute-op dispatch and segment state plumbing for simulators."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.ir.gather_plan import gather_plan_for
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper


def assemble_segment_input_torch(
    input_map,
    state_buffer: Dict[int, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a segment's composite input tensor from the state buffer."""
    total_size = max((s.offset + s.size for s in input_map), default=0)
    inp = torch.zeros(batch_size, total_size, device=device, dtype=dtype)
    for s in input_map:
        buf = state_buffer[s.node_id]
        inp[:, s.offset : s.offset + s.size] = buf[:, : s.size].to(dtype)
    return inp


def store_segment_output_torch(
    output_map,
    state_buffer: Dict[int, torch.Tensor],
    output_tensor: torch.Tensor,
) -> None:
    """Parse a segment's output tensor into the state buffer."""
    for s in output_map:
        state_buffer[s.node_id] = output_tensor[:, s.offset : s.offset + s.size]


def gather_final_output_torch(
    output_sources: np.ndarray,
    state_buffer: Dict[int, torch.Tensor],
    original_input: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble the network's final output from the state buffer."""
    flat = output_sources.flatten()
    out = torch.zeros(batch_size, len(flat), device=device, dtype=dtype)
    for idx, src in enumerate(flat):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        elif src.is_input():
            out[:, idx] = original_input[:, src.index].to(dtype)
        elif src.is_always_on():
            out[:, idx] = 1.0
        else:
            out[:, idx] = state_buffer[src.node_id][:, src.index].to(dtype)
    return out


def _compute_op_module_dtype(op: ComputeOp) -> torch.dtype:
    """The op module's floating dtype (float32 for parameterless modules)."""
    module = op.params.get("module")
    if module is not None and hasattr(module, "parameters"):
        for p in module.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
    return torch.float32


def execute_compute_op_torch(
    op: ComputeOp,
    original_input: torch.Tensor,
    state_buffer: Dict[int, torch.Tensor],
    *,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute a host-side ComputeOp; optional in/out activation scales."""
    if out_scale is None:
        out_scale = in_scale

    gathered = op.gather_inputs(original_input, state_buffer)
    gathered = gathered.to(_compute_op_module_dtype(op))
    if abs(in_scale - 1.0) > 1e-9:
        gathered = gathered * in_scale

    result = op.execute_on_gathered(gathered)

    if abs(out_scale - 1.0) > 1e-9:
        result = result / out_scale
    if output_dtype is not None and result.dtype != output_dtype:
        result = result.to(output_dtype)
    return result


def assemble_segment_input_numpy(
    input_map,
    state_buffer: Dict[int, np.ndarray],
    num_samples: int,
    *,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Numpy segment-input assembly; pass ``dtype=np.float64`` for HCM parity."""
    total_size = max((s.offset + s.size for s in input_map), default=0)
    inp = np.zeros((num_samples, total_size), dtype=dtype)
    for s in input_map:
        buf = state_buffer[s.node_id]
        inp[:, s.offset : s.offset + s.size] = buf[:, : s.size]
    return inp


def apply_input_shifts_numpy(
    input_map,
    seg_input: np.ndarray,
    node_output_shifts,
) -> np.ndarray:
    """Add per-producer-channel positive shift to a segment input (numpy mirror of
    ``HybridLifStepMixin._apply_input_shifts``). Value-preserving: the consumer bias
    is pre-corrected ``B' = B − W·s``; empty/None ⇒ identity (no copy)."""
    if not node_output_shifts:
        return seg_input
    out = seg_input
    copied = False
    for s in input_map:
        shift = node_output_shifts.get(int(s.node_id))
        if shift is None:
            continue
        if not copied:
            out = seg_input.copy()
            copied = True
        sh = np.asarray(shift, dtype=out.dtype).reshape(-1)
        out[:, s.offset : s.offset + s.size] += sh[: s.size]
    return out


def compute_input_state_with_shifts(
    op: ComputeOp,
    state_buffer,
    node_output_shifts,
):
    """State-buffer view with producer ``node_output_shifts`` added to ``op``'s inputs.

    The NF driver lifts a shifted producer's value once at production; a compute-op
    consumer's baked bias (``B' = B − W·s``) expects lifted inputs, so the host
    value path must gather them lifted too (rate/LIF domain; TTFS transcodes via
    compute scales). No shifted inputs => identity (no copy). Works on torch and
    numpy buffers alike.
    """
    if not node_output_shifts:
        return state_buffer
    shifted_ids = {
        int(src.node_id)
        for src in op.input_sources.flatten()
        if isinstance(src, IRSource) and src.node_id >= 0
    } & set(node_output_shifts)
    shifted_ids = {nid for nid in shifted_ids if nid in state_buffer}
    if not shifted_ids:
        return state_buffer
    view = dict(state_buffer)
    for nid in shifted_ids:
        buf = state_buffer[nid]
        shift = node_output_shifts[nid]
        if isinstance(buf, torch.Tensor):
            sh = torch.as_tensor(
                shift, dtype=buf.dtype, device=buf.device,
            ).reshape(1, -1)
        else:
            sh = np.asarray(shift, dtype=buf.dtype).reshape(1, -1)
        view[nid] = buf + sh
    return view


def store_segment_output_numpy(
    output_map,
    state_buffer: Dict[int, np.ndarray],
    output: np.ndarray,
) -> None:
    """Numpy analogue of :func:`store_segment_output_torch`."""
    for s in output_map:
        state_buffer[s.node_id] = output[:, s.offset : s.offset + s.size]


def gather_final_output_numpy(
    output_sources: np.ndarray,
    state_buffer: Dict[int, np.ndarray],
    original_input: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Numpy analogue of :func:`gather_final_output_torch`."""
    flat_sources = output_sources.flatten()
    out = np.zeros((num_samples, len(flat_sources)), dtype=np.float32)
    for idx, src in enumerate(flat_sources):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        elif src.is_input():
            out[:, idx] = original_input[:, src.index]
        elif src.is_always_on():
            out[:, idx] = 1.0
        else:
            out[:, idx] = state_buffer[src.node_id][:, src.index]
    return out


def execute_compute_op_numpy(
    op: ComputeOp,
    original_input: np.ndarray,
    state_buffer: Dict[int, np.ndarray],
    *,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Execute ComputeOp via torch wrapper; ``dtype=np.float64`` for HCM parity."""
    if out_scale is None:
        out_scale = in_scale

    torch_dtype = (torch.float64 if np.dtype(dtype) == np.float64
                   else torch.float32)
    x_torch = torch.tensor(original_input, dtype=torch_dtype)
    # Convert only the producer buffers this op's gather actually reads.
    referenced = gather_plan_for(op).referenced_node_ids
    buffers_torch = {
        k: torch.tensor(state_buffer[k], dtype=torch_dtype) for k in referenced
    }
    result = execute_compute_op_torch(
        op,
        x_torch,
        buffers_torch,
        in_scale=in_scale,
        out_scale=out_scale,
        output_dtype=torch_dtype,
    )
    return result.detach().numpy()


def compute_op_owns_scale_domain(op) -> bool:
    """Whether the op module transcodes rate<->absolute itself (``ScaleNormalizingWrapper``).

    Such an op computes ``f(r_i * s_i) / s_out`` internally; applying the outer
    stage scales around it would apply the domain transcode twice.
    """
    params = getattr(op, "params", None) or {}
    return isinstance(params.get("module"), ScaleNormalizingWrapper)


def resolve_stage_compute_scales(
    mapping,
    op_id: int,
    *,
    apply_ttfs: bool = True,
    op=None,
) -> tuple[float, float]:
    """Return ``(input_scale, output_scale)`` for a hybrid compute stage.

    ``apply_ttfs`` selects the TTFS value-domain convention: decode the gathered
    rates by the producer scale, run the op in absolute units, re-normalize by the
    op's boundary out-scale. Pass ``op`` so wrapper-owned ops keep ``(1, 1)``.
    """
    out_scales = getattr(mapping, "node_activation_scales", {})
    in_scales = getattr(mapping, "node_input_activation_scales", out_scales)
    if not apply_ttfs:
        return 1.0, 1.0
    if op is not None and compute_op_owns_scale_domain(op):
        return 1.0, 1.0
    return float(in_scales.get(op_id, 1.0)), float(out_scales.get(op_id, 1.0))


def decref_consumers(
    state_buffer,
    remaining: Dict[int, int],
    src_ids: Iterable[int],
) -> None:
    """Drop state-buffer entries whose consumer refcount reaches zero."""
    for nid in src_ids:
        if nid < 0:
            continue
        r = remaining.get(nid)
        if r is None:
            continue
        r -= 1
        if r <= 0:
            remaining.pop(nid, None)
            state_buffer.pop(nid, None)
        else:
            remaining[nid] = r
