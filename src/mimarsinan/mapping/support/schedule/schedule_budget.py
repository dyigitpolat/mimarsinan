"""Effective hardware-core budget: the 0.8x heterogeneous-discount SSOT shared by the wizard and the hard-core mapper."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

#: [U-CG] The coalescing role of a core that holds its whole fan-in on fused cores (``layout/ir_mapping/fc.py``).
WHOLE_FAN_IN_ROLE = "master"


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


def splits_between_members(roles: Iterable[Optional[str]]) -> bool:
    """[U-CG] A coalescing group whose members each hold their whole fan-in (every role ``WHOLE_FAN_IN_ROLE``) shares
    no partial sum between its members, so the group may run in several passes, each member in one. A group with any
    other member (a partial sum, its accumulator, an unknown role) is one unit: never cut across passes."""
    roles = list(roles)
    return bool(roles) and all(role == WHOLE_FAN_IN_ROLE for role in roles)


def pass_atoms(bundles: Sequence[List[LayoutSoftCoreSpec]],
               packs: Callable[[List[LayoutSoftCoreSpec]], bool]) -> List[List[LayoutSoftCoreSpec]]:
    """[U-CG] The capacity split's atomic bundles: a coalescing group stays one bundle wherever it packs alone; a group
    of several whole-fan-in members that alone does not pack is its members, each one bundle, so they accumulate into
    passes as ungrouped specs do and a member is never cut. Every other bundle is kept as it is (a partial-sum group
    that does not pack stays one infeasible pass, refused by name)."""
    atoms: List[List[LayoutSoftCoreSpec]] = []
    for bundle in bundles:
        if len(bundle) > 1 and splits_between_members(getattr(sc, "coalescing_role", None) for sc in bundle) \
                and not packs(bundle):
            atoms.extend([sc] for sc in bundle)
        else:
            atoms.append(bundle)
    return atoms
