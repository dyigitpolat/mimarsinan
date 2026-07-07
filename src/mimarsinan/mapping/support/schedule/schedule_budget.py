"""Effective hardware-core budget: the 0.8x heterogeneous-discount SSOT shared by the wizard and the hard-core mapper."""

from __future__ import annotations

from typing import List, Sequence

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def effective_core_budget(cores_config: Sequence[dict]) -> int:
    """Compute the effective hardware-core budget for schedule partitioning.

    When multiple core types exist, the budget is reduced by 20% to account
    for per-type scarcity (a core needing e.g. 64 neurons can only use types
    that offer >= 64 neurons, even if smaller types have spare capacity).
    """
    total = sum(int(ct["count"]) for ct in cores_config)
    n_types = len(cores_config)
    return int(total * 0.8) if n_types > 1 else total


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

