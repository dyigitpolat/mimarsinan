"""Per-producer positive-shift application at hybrid segment inputs."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import ComputeOp, IRSource


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

    A compute-op's baked bias (``B' = B − W·s``) expects lifted inputs, so the
    host value path must gather them lifted too. No shifted inputs => identity.
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
