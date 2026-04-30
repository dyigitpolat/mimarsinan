"""Shared execution primitives for spiking simulation paths.

Single source of truth for ComputeOp dispatch and segment-boundary state
plumbing across SCM (``SpikingHybridCoreFlow``), nevresim (``SimulationRunner``),
and Loihi (``LavaLoihiRunner``). All three of those simulators must produce
identical numbers on the same input — divergence between them is by definition
a bug — so the shared helpers ensure they execute the same arithmetic.

Two parallel APIs:

  * ``execute_compute_op_torch`` / ``assemble_segment_input_torch`` /
    ``store_segment_output_torch`` / ``gather_final_output_torch`` —
    used by the PyTorch flows (``SpikingHybridCoreFlow``,
    ``SpikingUnifiedCoreFlow``).
  * ``execute_compute_op_numpy`` / ``assemble_segment_input_numpy`` /
    ``store_segment_output_numpy`` / ``gather_final_output_numpy`` —
    used by ``SimulationRunner`` (nevresim) and ``LavaLoihiRunner``.

The numpy variants previously lived as ``@staticmethod``s on
``SimulationRunner``; pulling them up here lets the Loihi runner and the
torch flows share their semantics with nevresim without an import cycle
or a copy.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from mimarsinan.mapping.ir import ComputeOp, IRSource


# ---------------------------------------------------------------------------
# Torch path
# ---------------------------------------------------------------------------


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


def execute_compute_op_torch(
    op: ComputeOp,
    original_input: torch.Tensor,
    state_buffer: Dict[int, torch.Tensor],
    *,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute a host-side ComputeOp in float32 (matches C++ nevresim).

    ``in_scale`` rescales the gathered input back into training range
    before running the wrapped module (NeuralCore outputs arrive in
    [0, 1] but module weights/biases were never divided by activation
    scale). ``out_scale`` divides the result so downstream NeuralCores
    see [0, 1] again. ``out_scale=None`` mirrors the legacy behaviour of
    ``SimulationRunner._execute_compute_op_np`` and falls back to
    ``in_scale``.

    Returns ``out`` in ``output_dtype`` (defaults to the gathered
    float32 result; HCM forward casts to ``_COMPUTE_DTYPE`` before
    storing in its state buffer).
    """
    if out_scale is None:
        out_scale = in_scale

    gathered = op.gather_inputs(original_input, state_buffer)
    gathered = gathered.to(torch.float32)
    if abs(in_scale - 1.0) > 1e-9:
        gathered = gathered * in_scale

    result = op.execute_on_gathered(gathered)

    if abs(out_scale - 1.0) > 1e-9:
        result = result / out_scale
    if output_dtype is not None and result.dtype != output_dtype:
        result = result.to(output_dtype)
    return result


# ---------------------------------------------------------------------------
# Numpy path (used by nevresim SimulationRunner and LavaLoihiRunner)
# ---------------------------------------------------------------------------


def assemble_segment_input_numpy(
    input_map,
    state_buffer: Dict[int, np.ndarray],
    num_samples: int,
) -> np.ndarray:
    """Numpy analogue of :func:`assemble_segment_input_torch`."""
    total_size = max((s.offset + s.size for s in input_map), default=0)
    inp = np.zeros((num_samples, total_size), dtype=np.float32)
    for s in input_map:
        buf = state_buffer[s.node_id]
        inp[:, s.offset : s.offset + s.size] = buf[:, : s.size]
    return inp


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
) -> np.ndarray:
    """Execute a host-side ComputeOp via the torch wrapper, returning numpy.

    Drives the same ``execute_compute_op_torch`` machinery so SCM/HCM/Sim
    follow one code path. Inputs are converted from numpy → torch and
    back; the actual op invocation lives in ``execute_compute_op_torch``.
    """
    if out_scale is None:
        out_scale = in_scale

    x_torch = torch.tensor(original_input, dtype=torch.float32)
    buffers_torch = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in state_buffer.items()
    }
    result = execute_compute_op_torch(
        op,
        x_torch,
        buffers_torch,
        in_scale=in_scale,
        out_scale=out_scale,
    )
    return result.detach().numpy()


# ---------------------------------------------------------------------------
# Decref pruning helper (state-buffer life-time management)
# ---------------------------------------------------------------------------


def decref_consumers(
    state_buffer,
    remaining: Dict[int, int],
    src_ids: Iterable[int],
) -> None:
    """Decrement ``remaining[nid]`` for each source; drop ``state_buffer``
    entries whose refcount hits 0. Source ids < 0 (raw input, always-on,
    off) are ignored.

    Shared between SpikingHybridCoreFlow and any future caller that wants
    the same in-forward memory hygiene.
    """
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
