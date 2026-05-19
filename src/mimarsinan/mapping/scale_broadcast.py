"""Shared scale-vector broadcast helpers for mapper and IR paths."""

from __future__ import annotations

import torch


def broadcast_scale_to_dim(scale: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Expand a 1-D scale vector to *target_dim* using repeat or mean-fill."""
    if scale.shape[0] == target_dim:
        return scale
    if target_dim % scale.shape[0] == 0:
        return scale.repeat_interleave(target_dim // scale.shape[0])
    return torch.full(
        (target_dim,), scale.mean().item(), dtype=scale.dtype, device=scale.device
    )


def broadcast_scale_pair(s_a: torch.Tensor, s_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the shorter scale vector to match the longer one."""
    n_a, n_b = len(s_a), len(s_b)
    if n_a == n_b:
        return s_a, s_b
    if n_a < n_b:
        short, long_, flipped = s_a, s_b, True
    else:
        short, long_, flipped = s_b, s_a, False

    n_s, n_l = len(short), len(long_)
    if n_l % n_s == 0:
        expanded = short.repeat_interleave(n_l // n_s)
    else:
        expanded = torch.full((n_l,), short.mean().item())

    if flipped:
        return expanded, long_
    return long_, expanded
