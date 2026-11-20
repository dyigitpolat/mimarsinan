"""[U-LIF2 W15] The membrane program check of integrate then emit: a declared register holds every window sum.

Under integrate then emit a neuron holds its whole window's charge before it emits, so its membrane passes through every
partial sum of the window, not only the residue below theta that firing while integrating leaves after each cycle. Each
axon delivers at most T spikes a window (a uniform encode, a carried count, a producer's emitted count, an always-on
row) and the bias lands on T cycles, so every partial sum of neuron j lies in

    [v0 theta + T (sum_i min(w_ij, 0) + min(b_j, 0)),  v0 theta + T (sum_i max(w_ij, 0) + max(b_j, 0))]

over the weights the executor reads (``build_segment_tensors``: a bank's or the core's own); the emission window only
lowers the membrane from the window's sum towards [0, theta). A register that asserts its rails holds the bound strictly
inside them; one that clamps, inside them. A program whose bound leaves its declared register refuses integrate then
emit by name at generation (Hard Core Mapping, before any simulation), and the gated executor's rail assertion stays the
guard at execution. Firing while integrating and an unbounded membrane are not judged here. Unlike the static check's
conditions (read-only reports), this rule refuses: a program it refuses is never simulated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.spiking.hybrid.segment_cache import build_segment_tensors


class WindowSumRefusal(ToolchainRefusal, ValueError):
    """A declared register that cannot hold a program's window sum under integrate then emit, refused by name."""


@dataclass(frozen=True)
class WindowSumBound:
    """One core's window-sum interval over its neurons, in its execution stage (the executor's index)."""

    stage_index: int
    stage_name: str
    core_index: int
    lowest: float
    highest: float


def core_window_sum_bounds(weights: torch.Tensor, bias: Optional[torch.Tensor], theta: Any, *, window: int,
                           membrane_init: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per neuron (``weights`` is ``(neurons, axons)``, as the executor holds it), the lowest and highest partial sum of
    a window. ``theta`` is the core's threshold, one value or one per neuron: the guard precharges each neuron by its
    own (``precharge_lif_states``)."""
    w = weights.to(torch.float64)
    b = torch.zeros(w.shape[0], dtype=torch.float64) if bias is None else bias.to(torch.float64).reshape(-1)
    start = float(membrane_init) * torch.as_tensor(theta, dtype=torch.float64).reshape(-1)
    lowest = start + int(window) * (w.clamp(max=0.0).sum(dim=1) + b.clamp(max=0.0))
    highest = start + int(window) * (w.clamp(min=0.0).sum(dim=1) + b.clamp(min=0.0))
    return lowest, highest


def program_window_sum_bounds(mapping: Any, *, window: int, membrane_init: float) -> Tuple[WindowSumBound, ...]:
    """Every core's window-sum interval, over the program's neural execution stages (level stages included)."""
    out: List[WindowSumBound] = []
    cpu = torch.device("cpu")
    for index, _stage, unit in enumerate_execution_stages(mapping.stages):
        if unit.kind != "neural" or getattr(unit, "hard_core_mapping", None) is None:
            continue
        seg = build_segment_tensors(unit.hard_core_mapping, cpu)
        for core, (weights, bias, theta) in enumerate(zip(seg["core_params"], seg["hw_biases"], seg["thresholds"])):
            lowest, highest = core_window_sum_bounds(weights.cpu(), None if bias is None else bias.cpu(),
                                                     theta.cpu(), window=window, membrane_init=membrane_init)
            out.append(WindowSumBound(index, str(getattr(unit, "name", "?")), core, float(lowest.min()),
                                      float(highest.max())))
    return tuple(out)


def require_window_sum_held(mapping: Any, point: Any, *, window: int,
                            membrane_init: float) -> Tuple[WindowSumBound, ...]:
    """Refuse by name a program whose window sum leaves the declared register of a point that integrates then emits;
    the bounds it judged, or ``()`` where the rule does not apply (no point, firing while integrating, no register)."""
    if point is None or not point.integrates_then_emits or point.soma_law.membrane_bounds is None:
        return ()
    law = point.soma_law
    low, high = law.membrane_bounds
    strict = law.asserts_no_saturation
    bounds = program_window_sum_bounds(mapping, window=window, membrane_init=membrane_init)
    for b in bounds:
        inside = (low < b.lowest and b.highest < high) if strict else (low <= b.lowest and b.highest <= high)
        if not inside:
            kind = "signed" if law.is_signed_membrane else "unsigned"
            raise WindowSumRefusal(
                f"lif_window_firing=integrate_then_emit is refused for this program: its window sums reach "
                f"[{b.lowest:g}, {b.highest:g}] (execution stage {b.stage_index} {b.stage_name!r}, core "
                f"{b.core_index}, T={int(window)}), which the declared {int(law.membrane_bits)}-bit {kind} register "
                f"[{low:g}, {high:g}] does not hold {'strictly inside its rails' if strict else 'without clamping'}. "
                f"Integrate then emit holds the whole window's charge before it emits. Widen membrane_bits, or "
                f"declare lif_window_firing=fire_while_integrating.")
    return bounds
