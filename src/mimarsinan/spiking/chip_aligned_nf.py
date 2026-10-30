"""Chip-aligned LIF NF forward — thin wrapper over the unified segment driver."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.nn.activations import run_cycle_accurate
from mimarsinan.models.spiking.serial import refuse_cycle_atomic
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver


def chip_aligned_segment_forward(
    model: nn.Module, x: torch.Tensor, T: int,
    *, soma_law: SomaLaw,
    retime: bool = False,
    phase_dither: bool = False,
    synchronized: bool = False,
    compute_min_recorder: dict | None = None,
    node_value_recorder: dict | None = None,
) -> torch.Tensor:
    """Segment-aware chip-aligned NF forward (matches HCM ``_forward_rate``);
    ``retime`` selects the [C3/R5] per-hop re-encoded twin; ``phase_dither``
    the count-exact decorrelated encode combs [calculus sec.15.11];
    ``synchronized`` the two-window integrate-then-emit discipline whose count
    is exactly the strict staircase [calculus sec.16].

    ``soma_law`` is default-free: this forward IS the deployed twin, and a
    defaulted point silently runs a DIFFERENT physics than the deployment it
    mirrors (the t0_54 incident — the LIF install dropped the point and the
    whole run's NF fired at most one spike per cycle).
    """
    if not hasattr(model, "get_mapper_repr"):
        return _unmapped_cycle_accurate(model, x, T, soma_law)
    mapper_repr = cast(Any, model).get_mapper_repr()
    if mapper_repr is None:
        return _unmapped_cycle_accurate(model, x, T, soma_law)
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


def _unmapped_cycle_accurate(
    model: nn.Module, x: torch.Tensor, T: int, soma_law: SomaLaw,
) -> torch.Tensor:
    """The mapper-repr-less fallback: the plain per-cycle module walk."""
    refuse_cycle_atomic(
        soma_law,
        mechanism="the mapper-repr-less cycle-accurate NF fallback",
        theorem=(
            "it drives each activation's binary IF node once per cycle, whose "
            "hypothesis is that a neuron emits at most ONE spike per cycle."
        ),
    )
    return run_cycle_accurate(model, x, T)
