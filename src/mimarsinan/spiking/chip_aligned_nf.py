"""Chip-aligned LIF NF forward — thin wrapper over the unified segment driver."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import run_cycle_accurate
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver


def chip_aligned_segment_forward(
    model: nn.Module, x: torch.Tensor, T: int,
    *, compute_min_recorder: dict | None = None,
) -> torch.Tensor:
    """Segment-aware chip-aligned NF forward (matches HCM ``_forward_rate``).

    Runs the unified :class:`SegmentForwardDriver` with the LIF policy:
    perceptrons cascade per-cycle (signed-IF) inside neural segments, each
    ComputeOp runs once on the decoded rate (with optional per-channel min
    recording and ``_negative_shift`` application), and downstream segments
    re-encode — the decode->compute->re-encode HCM performs at each boundary.
    """
    if not hasattr(model, "get_mapper_repr"):
        return run_cycle_accurate(model, x, T)
    mapper_repr = model.get_mapper_repr()
    if mapper_repr is None:
        return run_cycle_accurate(model, x, T)
    driver = SegmentForwardDriver(mapper_repr, T, LifSegmentPolicy())
    return driver(x, compute_min_recorder=compute_min_recorder)
