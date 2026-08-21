"""Chip-aligned LIF NF forward — thin wrapper over the unified segment driver."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.models.nn.activations import run_cycle_accurate
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver


def chip_aligned_segment_forward(
    model: nn.Module, x: torch.Tensor, T: int,
    *, retime: bool = False,
    phase_dither: bool = False,
    synchronized: bool = False,
    soma_law: SomaLaw = DEFAULT_SOMA_LAW,
    compute_min_recorder: dict | None = None,
    node_value_recorder: dict | None = None,
) -> torch.Tensor:
    """Segment-aware chip-aligned NF forward (matches HCM ``_forward_rate``);
    ``retime`` selects the [C3/R5] per-hop re-encoded twin; ``phase_dither``
    the count-exact decorrelated encode combs [calculus sec.15.11];
    ``synchronized`` the two-window integrate-then-emit discipline whose count
    is exactly the strict staircase [calculus sec.16]."""
    if not hasattr(model, "get_mapper_repr"):
        return run_cycle_accurate(model, x, T)
    mapper_repr = cast(Any, model).get_mapper_repr()
    if mapper_repr is None:
        return run_cycle_accurate(model, x, T)
    driver = SegmentForwardDriver(
        mapper_repr, T,
        LifSegmentPolicy(
            retime=retime, phase_dither=phase_dither,
            synchronized=synchronized, soma_law=soma_law,
        ),
    )
    return driver(
        x,
        compute_min_recorder=compute_min_recorder,
        node_value_recorder=node_value_recorder,
    )
