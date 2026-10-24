"""Capacity-driven splitting of a layout segment's latency groups into sub-segments that fit the hardware pool, cutting at the first overflowing boundary."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.packing.greedy.split import _is_splittable
from mimarsinan.mapping.support.schedule.schedule_budget import _coalescing_bundles

#: The bound of a softcore the greedy cannot place on the pool's core shape (above any core count).
UNPLACEABLE = 1 << 62


@dataclass(frozen=True)
class CapacityBound:
    """Per softcore, the most cores the greedy packer can open for it on a pool of one core shape.

    A softcore (after the packer's axon-coalescing expansion) opens at most 1 core when it fits a fresh one,
    ceil(a / A) when it fits only fused, ceil(n / N) when it is neuron-split, and is unplaceable otherwise. A set
    whose bounds sum to at most ``count`` therefore packs, and so does every subset of it in any order: the greedy
    fails only when a softcore needs a core the pool has run out of.
    """

    per_softcore: Tuple[int, ...]
    count: int

    @property
    def total(self) -> int:
        return sum(self.per_softcore)


def _cores_opened(axons: int, neurons: int, A: int, N: int, splittable: bool) -> int:
    if neurons <= N:
        return 1 if axons <= A else -(-axons // A)
    if axons <= A and splittable:
        return -(-neurons // N)
    return UNPLACEABLE


def capacity_bound(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_coalescing: bool,
    allow_splitting: bool,
) -> Optional[CapacityBound]:
    """The bound of every softcore on the pool ``pack_layout`` builds from ``core_types``; None unless the pool
    holds one core shape (several shapes contend for a softcore by abundance, which no per-softcore bound sees)."""
    from mimarsinan.mapping.layout.layout_packer import _expand_for_axon_coalescing

    shapes = {(int(t.max_axons), int(t.max_neurons)) for t in core_types if int(t.count) > 0}
    if len(shapes) != 1:
        return None
    (A, N), = shapes
    if A <= 0 or N <= 0:
        return None
    count = sum(int(t.count) for t in core_types if (int(t.max_axons), int(t.max_neurons)) == (A, N))
    if allow_coalescing:
        packed, _, origins = _expand_for_axon_coalescing(softcores, core_types)
    else:
        packed, origins = list(softcores), list(range(len(softcores)))
    per = [0] * len(softcores)
    for spec, origin in zip(packed, origins):
        per[origin] += _cores_opened(int(spec.input_count), int(spec.output_count), A, N,
                                     allow_splitting and _is_splittable(spec))
    return CapacityBound(per_softcore=tuple(per), count=count)


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
       alone does not pack (e.g. because unique ``residency_class_id``
       fragmentation forces one hw core per softcore and the group is
       larger than the pool), the group is halved recursively into
       sub-passes until each half packs or is a singleton.  Singletons
       that still fail are returned as-is so the caller's per-sub-segment
       ``pack_layout`` validation can flag infeasibility loudly.

    A batch whose ``capacity_bound`` fits the pool is known to pack, so it is
    never packed; on a pool of one core shape every other batch is decided by
    its residency classes' demands (``packing/class_demand.py``), the packer's
    own verdict, extended a bundle at a time (the decisions are the packer's
    either way).
    """
    return [sub for sub, _ in split_with_verdicts(
        softcores, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
        coalescing_group_ids=coalescing_group_ids)]


def split_with_verdicts(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    coalescing_group_ids: Sequence[int | None] | None = None,
) -> List[Tuple[List[LayoutSoftCoreSpec], bool]]:
    """``split_softcores_by_capacity`` with each sub-segment's ``pack_layout`` feasibility: the verdict of the
    split's own check of that very list, or (for a singleton the halving returns unchecked) a check made here."""
    if not softcores:
        return []

    from mimarsinan.mapping.layout.layout_packer import pack_layout
    from mimarsinan.mapping.packing.class_demand import PoolDemand

    hw_types = list(core_types)
    # pack_layout copies a batch with one memo, so an object listed twice is one softcore placed twice: the
    # bound holds no more, as the pick index then raises, and such a segment is packed as it is.
    distinct = len({id(sc) for sc in softcores}) == len(softcores)
    # The bound and the pool demand certify geometry alone; core types that restrict roles (N2) are judged by
    # ``pack_layout`` itself, which routes them.
    geometric = not any(getattr(t, "accepted_roles", None) is not None for t in hw_types)
    bound = capacity_bound(softcores, hw_types, allow_coalescing=allow_coalescing,
                           allow_splitting=allow_splitting) if distinct and geometric else None
    demand = PoolDemand.for_pool(softcores, hw_types, allow_coalescing=allow_coalescing,
                                 allow_splitting=allow_splitting) if geometric else None
    tally = demand.running() if demand is not None else None
    cost: Dict[int, int] = {} if bound is None else {
        id(sc): b for sc, b in zip(softcores, bound.per_softcore)}

    def _cost(batch: Sequence[LayoutSoftCoreSpec]) -> int:
        return sum(cost[id(sc)] for sc in batch) if bound is not None else 0

    def _certified(batch_cost: int) -> bool:
        return bound is not None and batch_cost <= bound.count

    def _packs(batch: Sequence[LayoutSoftCoreSpec], batch_cost: int) -> bool:
        if not batch or _certified(batch_cost):
            return True
        if demand is not None:
            return demand.feasible(batch)
        pr = pack_layout(
            softcores=list(batch),
            core_types=hw_types,
            allow_neuron_splitting=allow_splitting,
            allow_coalescing=allow_coalescing,
        )
        return bool(pr.feasible)

    def _halve_until_packs(group: List[LayoutSoftCoreSpec]) -> List[Tuple[List[LayoutSoftCoreSpec], Optional[bool]]]:
        """Binary-halve a single latency group until each piece packs (a singleton piece unchecked)."""
        if len(group) <= 1:
            return [(list(group), None)]
        if _packs(group, _cost(group)):
            return [(list(group), True)]
        mid = len(group) // 2
        return _halve_until_packs(group[:mid]) + _halve_until_packs(group[mid:])

    use_coalescing_bundles = False
    if (
        allow_coalescing
        and coalescing_group_ids is not None
        and len(coalescing_group_ids) == len(softcores)
    ):
        use_coalescing_bundles = True
        atomic = _coalescing_bundles(softcores, coalescing_group_ids)
    else:
        atomic = [[sc] for sc in softcores]

    def _bundle_latency(bundle: List[LayoutSoftCoreSpec]) -> int:
        return max(int(sc.latency_tag) if sc.latency_tag is not None else 0 for sc in bundle)

    lat_groups: dict[int, List[List[LayoutSoftCoreSpec]]] = defaultdict(list)
    for bundle in atomic:
        lat_groups[_bundle_latency(bundle)].append(bundle)

    sub_segments: List[Tuple[List[LayoutSoftCoreSpec], Optional[bool]]] = []
    running: List[LayoutSoftCoreSpec] = []
    running_ok: Optional[bool] = True
    running_cost = 0
    for lat in sorted(lat_groups.keys()):
        for bundle in lat_groups[lat]:
            flat = [sc for sc in bundle]
            flat_cost = _cost(flat)
            candidate_cost = running_cost + flat_cost
            # The candidate list is built only for a pack; a certified candidate extends the running list in place.
            if _certified(candidate_cost):
                if tally is not None:
                    tally.take(flat)
                accepted = True
            else:
                accepted = tally.extends(flat) if tally is not None else _packs(running + flat, candidate_cost)
            if accepted:
                running.extend(flat)
                running_ok, running_cost = True, candidate_cost
                continue

            if running:
                sub_segments.append((running, running_ok))
                running, running_cost = [], 0

            if _packs(flat, flat_cost):
                running, running_ok, running_cost = list(flat), True, flat_cost
            elif use_coalescing_bundles and len(flat) > 1:
                # Cannot split a coalescing group across schedule passes.
                sub_segments.append((flat, False))
                running, running_cost = [], 0
            else:
                halved = _halve_until_packs(list(flat))
                for piece in halved[:-1]:
                    sub_segments.append(piece)
                running, running_ok = halved[-1]
                running_cost = _cost(running)
            if tally is not None:
                tally.reset(running)

    if running:
        sub_segments.append((running, running_ok))

    return [(sub, _packs(sub, _cost(sub)) if ok is None else ok) for sub, ok in sub_segments]
