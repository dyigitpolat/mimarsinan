"""Segment-level pass accounting and capacity-driven splitting.

The hard-core mapper runs exactly one flush per *sub-segment*, where a
sub-segment is the largest prefix of a layout segment's latency groups
that still fits the hardware pool.  When the full segment doesn't fit,
``split_softcores_by_capacity`` cuts it at the first overflowing latency
boundary and a fresh sub-segment starts with the offending group.  The
rate-level handoff between adjacent ``HybridStage(kind="neural")``
entries (state_buffer read/write on shared node ids) is the sync
barrier at the simulator level — no synthetic ComputeOp is inserted.

A single latency group whose softcores cannot pack alone is genuine
infeasibility: ``estimate_passes_for_layout_validated`` reports
``all_feasible = False``, and the hard-core mapper lets the packer's
``RuntimeError`` propagate.  No silent retry.

``effective_core_budget`` retains the 0.8× heterogeneous-discount
semantics so the wizard and the hard-core mapper quote identical
available-core numbers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)


# Budget helper — single source of truth for both wizard and runtime

from mimarsinan.mapping.support.schedule.schedule_budget import effective_core_budget
from mimarsinan.mapping.support.schedule.schedule_split import split_softcores_by_capacity

def partition_segment_into_passes(
    cores: list[NeuralCore],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
) -> list[list[NeuralCore]]:
    """Return the segment as a single pass.

    The old latency-group split produced multiple passes which the
    simulator subsequently treated as sync barriers, breaking
    cycle-accurate LIF semantics.  Segmentation is now exclusively the
    layout mapper's responsibility; this function is the identity
    function on the segment's cores, preserved only because the unused
    kwargs were part of the historical public signature.
    """
    if not cores:
        return []
    return [list(cores)]


# Layout-level estimator — packs each layout segment whole

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
            try:
                pr = pack_layout(
                    softcores=sub,
                    core_types=hw_types,
                    allow_neuron_splitting=allow_splitting,
                    allow_coalescing=allow_coalescing,
                )
                if not pr.feasible:
                    all_ok = False
            except Exception:
                all_ok = False
        all_pass_lists.extend(sub_segments)

    return len(all_pass_lists), all_pass_lists, all_ok
