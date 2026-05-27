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

def effective_core_budget(cores_config: Sequence[dict]) -> int:
    """Compute the effective hardware-core budget for schedule partitioning.

    When multiple core types exist, the budget is reduced by 20% to account
    for per-type scarcity (a core needing e.g. 64 neurons can only use types
    that offer >= 64 neurons, even if smaller types have spare capacity).
    """
    total = sum(int(ct["count"]) for ct in cores_config)
    n_types = len(cores_config)
    return int(total * 0.8) if n_types > 1 else total


# Capacity-driven segment splitter (shared by layout verifier + hard-core mapper)


def _coalescing_bundles(
    softcores: Sequence[LayoutSoftCoreSpec],
    coalescing_group_ids: Sequence[int | None],
) -> List[List[LayoutSoftCoreSpec]]:
    """Group layout specs so each coalescing_group_id stays in one atomic bundle."""
    bundles: List[List[LayoutSoftCoreSpec]] = []
    i = 0
    n = len(softcores)
    while i < n:
        gid = coalescing_group_ids[i]
        if gid is None:
            bundles.append([softcores[i]])
            i += 1
            continue
        bundle: List[LayoutSoftCoreSpec] = []
        while i < n and coalescing_group_ids[i] == gid:
            bundle.append(softcores[i])
            i += 1
        bundles.append(bundle)
    return bundles

