"""Segment-level pass accounting: re-exports the schedule split/budget helpers plus layout-level pass estimation on top of them."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

# Re-exported: wizard / verification callers import both names from this module.
from mimarsinan.mapping.support.schedule.schedule_budget import (
    effective_core_budget as effective_core_budget,
)
from mimarsinan.mapping.support.schedule.schedule_split import (
    split_softcores_by_capacity as split_softcores_by_capacity,
)

def partition_segment_into_passes(
    cores: list[NeuralCore],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
) -> list[list[NeuralCore]]:
    """Identity on the segment's cores: multi-pass splitting would break
    cycle-accurate LIF sync barriers, so segmentation is the layout
    mapper's sole responsibility. Kwargs kept for signature stability."""
    if not cores:
        return []
    return [list(cores)]


def estimate_passes_for_layout(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    core_types: Optional[Sequence] = None,
) -> Tuple[int, List[List[LayoutSoftCoreSpec]]]:
    """Estimate the pass count as one pass per distinct layout segment.

    Returns ``(num_passes, pass_lists)``.  Each entry in *pass_lists* is
    the full softcore list for one segment (``segment_id``-grouped).
    """
    if not softcores or max_cores_per_pass <= 0:
        return (0, []) if not softcores else (1, [list(softcores)])

    by_segment: dict[int, list[LayoutSoftCoreSpec]] = defaultdict(list)
    for sc in softcores:
        seg = sc.segment_id if sc.segment_id is not None else 0
        by_segment[seg].append(sc)

    passes: List[List[LayoutSoftCoreSpec]] = [
        by_segment[seg_id] for seg_id in sorted(by_segment.keys())
    ]
    return len(passes), passes


def estimate_passes_for_layout_validated(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    core_types: Sequence,
) -> Tuple[int, List[List[LayoutSoftCoreSpec]], bool]:
    """Capacity-aware pass estimator.

    For each distinct layout ``segment_id``, split the segment's softcores
    into capacity-feasible sub-segments via
    :func:`split_softcores_by_capacity` and report one pass per
    sub-segment.  Infeasibility is declared only when a sub-segment's
    softcores cannot pack on their own (= a single latency group exceeds
    the hardware pool even after all capacity-driven barriers have been
    inserted).

    The returned ``(num_passes, pass_lists, all_feasible)`` mirrors what
    the hard-core mapper will emit at build time: the wizard now sees the
    same segment count the simulator will run.
    """
    if not softcores or max_cores_per_pass <= 0:
        ok = not softcores
        return (0, [], ok) if not softcores else (1, [list(softcores)], ok)

    hw_types = list(core_types)

    from mimarsinan.mapping.layout.layout_packer import pack_layout

    by_segment: dict[int, List[LayoutSoftCoreSpec]] = defaultdict(list)
    for sc in softcores:
        seg = sc.segment_id if sc.segment_id is not None else 0
        by_segment[seg].append(sc)

    all_pass_lists: List[List[LayoutSoftCoreSpec]] = []
    all_ok = True
    for seg_id in sorted(by_segment.keys()):
        seg_cores = by_segment[seg_id]
        sub_segments = split_softcores_by_capacity(
            seg_cores,
            hw_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_splitting,
        )
        if not sub_segments:
            continue
        for sub in sub_segments:
            pr = pack_layout(
                softcores=sub,
                core_types=hw_types,
                allow_neuron_splitting=allow_splitting,
                allow_coalescing=allow_coalescing,
            )
            if not pr.feasible:
                all_ok = False
        all_pass_lists.extend(sub_segments)

    return len(all_pass_lists), all_pass_lists, all_ok
