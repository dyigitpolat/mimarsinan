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

from mimarsinan.mapping.support.schedule.schedule_budget import _coalescing_bundles

def split_softcores_by_capacity(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    coalescing_group_ids: Sequence[int | None] | None = None,
) -> List[List[LayoutSoftCoreSpec]]:
    """Split a single-segment softcore list into capacity-feasible sub-segments.

    Two-level splitter:

    1. **Latency-group accumulation.**  Groups *softcores* by latency
       (ascending) and greedily accumulates groups into a running
       sub-segment.  After each addition ``pack_layout`` is called; the
       first accumulation that fails closes the running sub-segment and
       starts a fresh one with the offending group.
    2. **Within-group halving fallback.**  When a single latency group
       alone does not pack (e.g. because unique ``threshold_group_id``
       fragmentation forces one hw core per softcore and the group is
       larger than the pool), the group is halved recursively into
       sub-passes until each half packs or is a singleton.  Singletons
       that still fail are returned as-is so the caller's per-sub-segment
       ``pack_layout`` validation can flag infeasibility loudly.

    Cores within a single latency group have no inter-core dependencies
    (same DAG depth), so any partition of the group is functionally
    equivalent — halving simply schedules independent cores across
    successive chip reprograms.  No silent rate aggregation happens
    because adjacent sub-segments communicate through the state buffer
    at segment-level rates (same semantics as a real ComputeOp sync
    barrier).
    """
    if not softcores:
        return []

    from mimarsinan.mapping.layout.layout_packer import pack_layout

    hw_types = list(core_types)

    def _packs(batch: Sequence[LayoutSoftCoreSpec]) -> bool:
        if not batch:
            return True
        try:
            pr = pack_layout(
                softcores=list(batch),
                core_types=hw_types,
                allow_neuron_splitting=allow_splitting,
                allow_coalescing=allow_coalescing,
            )
            return bool(pr.feasible)
        except Exception:
            return False

    def _halve_until_packs(group: List[LayoutSoftCoreSpec]) -> List[List[LayoutSoftCoreSpec]]:
        """Binary-halve a single latency group until each piece packs."""
        if len(group) <= 1 or _packs(group):
            return [list(group)]
        mid = len(group) // 2
        return _halve_until_packs(group[:mid]) + _halve_until_packs(group[mid:])

    use_coalescing_bundles = bool(
        allow_coalescing
        and coalescing_group_ids is not None
        and len(coalescing_group_ids) == len(softcores)
    )
    if use_coalescing_bundles:
        atomic = _coalescing_bundles(softcores, coalescing_group_ids)
    else:
        atomic = [[sc] for sc in softcores]

    def _bundle_latency(bundle: List[LayoutSoftCoreSpec]) -> int:
        return max(int(sc.latency_tag) if sc.latency_tag is not None else 0 for sc in bundle)

    lat_groups: dict[int, List[List[LayoutSoftCoreSpec]]] = defaultdict(list)
    for bundle in atomic:
        lat_groups[_bundle_latency(bundle)].append(bundle)

    sub_segments: List[List[LayoutSoftCoreSpec]] = []
    running: List[LayoutSoftCoreSpec] = []
    for lat in sorted(lat_groups.keys()):
        for bundle in lat_groups[lat]:
            flat = [sc for sc in bundle]
            candidate = running + flat
            if _packs(candidate):
                running = candidate
                continue

            if running:
                sub_segments.append(running)
                running = []

            if _packs(flat):
                running = list(flat)
                continue

            if use_coalescing_bundles and len(flat) > 1:
                # Cannot split a coalescing group across schedule passes.
                sub_segments.append(flat)
                running = []
                continue

            halved = _halve_until_packs(list(flat))
            for piece in halved[:-1]:
                sub_segments.append(piece)
            running = list(halved[-1])

    if running:
        sub_segments.append(running)

    return sub_segments


# NeuralCore partitioner — kept for API stability, always returns 1 pass
