"""The packing engine, pack_layout and the capacity splitter as they stood before U1 planner scaling: the oracle.

A verbatim copy (renamed) of ``mapping/packing/greedy/{pack,split}.py``, ``layout_packer.pack_layout`` with
``_make_instances``, ``schedule_split.split_softcores_by_capacity`` and
``schedule_partitioner.estimate_passes_for_layout_validated`` on the U1 pass-projection tree. The accelerated
engine and splitter must take exactly these decisions; the equivalence tests compare them call for call.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import LayoutMaterializer, _expand_for_axon_coalescing
from mimarsinan.mapping.layout.layout_types import (
    LayoutCoreSnapshot,
    LayoutHardCoreInstance,
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.packing.canonical import HardT, SoftCoreLike, SoftT, _placement_waste, _remaining_capacity
from mimarsinan.mapping.packing.infeasibility_proofs import failed_pack_verdict, tag_with_verdict
from mimarsinan.mapping.packing.pick_index import PickBestIndex
from mimarsinan.mapping.packing.placement_cost import placement_cost
from mimarsinan.mapping.support.schedule.schedule_budget import _coalescing_bundles


def _swap_in_pick_index(pick_index, removed, added) -> None:
    """Keep the incremental pick index in lockstep with a split's remove+append."""
    if pick_index is None:
        return
    pick_index.discard(removed)
    pick_index.add(added)

def _is_splittable(core: SoftCoreLike) -> bool:
    """A soft core is splittable unless it is a coalescing-group fragment (already a partial result)."""
    return getattr(core, "coalescing_group_id", None) is None


def _try_split_into_used(
    core: SoftT,
    used_hardcores: List[HardT],
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]],
    split_threshold: float,
    place: Callable[[int, HardT, SoftT], None],
    softcores: List[SoftT],
    *,
    candidate_indices: "list[int] | None" = None,
    pick_index: "PickBestIndex | None" = None,
) -> bool:
    """Split *core* into the used core with the most remaining neurons (axons fit, neurons don't, remainder above threshold).

    ``candidate_indices`` restricts the scan to a pre-filtered subset; returns
    True if a split was performed.
    """
    best_idx = None
    best_avail_n = -1
    core_tg = getattr(core, "residency_class_id", None)
    core_latency = getattr(core, "latency", None)
    core_in = core.get_input_count()
    core_out = core.get_output_count()

    iterator = (
        ((i, used_hardcores[i]) for i in candidate_indices)
        if candidate_indices is not None
        else enumerate(used_hardcores)
    )

    for idx, hc in iterator:
        avail_a = getattr(hc, "available_axons", int(hc.get_input_count()))
        avail_n = getattr(hc, "available_neurons", int(hc.get_output_count()))
        total_n = int(hc.get_output_count())
        if (
            core_in <= avail_a
            and core_out > avail_n
            and avail_n > split_threshold * total_n
            and avail_n > best_avail_n
        ):
            hc_tg = getattr(hc, "residency_class_id", None)
            if hc_tg is not None and core_tg is not None and int(hc_tg) != int(core_tg):
                continue
            hc_latency = getattr(hc, "latency", None)
            if hc_latency is not None:
                if core_latency is None or core_latency != hc_latency:
                    continue
            best_idx = idx
            best_avail_n = avail_n

    if best_idx is None:
        return False

    frag1, frag2 = split_softcore(core, best_avail_n)
    place(best_idx, used_hardcores[best_idx], frag1)
    softcores.remove(core)
    softcores.append(frag2)
    _swap_in_pick_index(pick_index, core, frag2)
    return True


def _try_split_into_unused(
    core: SoftT,
    used_hardcores: List[HardT],
    unused_hardcores: List[HardT],
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]],
    split_threshold: float,
    place: Callable[[int, HardT, SoftT], None],
    softcores: List[SoftT],
    *,
    type_counts: "Counter | None" = None,
    type_key: Callable[[HardT], tuple] | None = None,
    unused_by_type: "dict | None" = None,
    hard_type_key: Callable[[HardT], tuple] | None = None,
    pick_index: "PickBestIndex | None" = None,
) -> bool:
    """Last-resort split of *core* into a fresh unused core when its neurons exceed every core type's capacity."""
    best_hc = None
    best_total_n = -1
    for hc in unused_hardcores:
        total_a = int(hc.get_input_count())
        total_n = int(hc.get_output_count())
        if (
            core.get_input_count() <= total_a
            and core.get_output_count() > total_n
            and total_n > 0
            and total_n > best_total_n
        ):
            best_hc = hc
            best_total_n = total_n

    if best_hc is None:
        return False

    frag1, frag2 = split_softcore(core, best_total_n)
    for _i, _x in enumerate(unused_hardcores):
        if _x is best_hc:
            del unused_hardcores[_i]
            break
    used_hardcores.append(best_hc)
    if type_counts is not None and type_key is not None:
        type_counts[type_key(best_hc)] -= 1
    if unused_by_type is not None and hard_type_key is not None:
        tk = hard_type_key(best_hc)
        bucket = unused_by_type.get(tk)
        if bucket is not None:
            for _i, _x in enumerate(bucket):
                if _x is best_hc:
                    del bucket[_i]
                    break
            if not bucket:
                del unused_by_type[tk]
    new_idx = len(used_hardcores) - 1
    place(new_idx, used_hardcores[new_idx], frag1)
    softcores.remove(core)
    softcores.append(frag2)
    _swap_in_pick_index(pick_index, core, frag2)
    return True



def reference_greedy_pack_softcores(
    *,
    softcores: List[SoftT],
    used_hardcores: List[HardT],
    unused_hardcores: List[HardT],
    is_mapping_possible: Callable[[SoftT, HardT], bool],
    place: Callable[[int, HardT, SoftT], None],
    pick_softcore: Callable[[List[SoftT]], SoftT] | None = None,
    fuse_hardcores: Callable[[List[HardT]], HardT] | None = None,
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]] | None = None,
    split_threshold: float = 0.2,
) -> None:
    """Greedy bin-packing loop shared by the runtime and layout-only packers.

    Priority per softcore: exact fit in a used core, split into a used core,
    exact fit in an unused core, fusion, split into an unused core, else raise.
    Mutates ``softcores``/``used_hardcores``/``unused_hardcores`` in place.
    ``pick_softcore=None`` (the default) keeps the canonical
    ``pick_best_softcore`` order, accelerated (sorted fast path without
    splitting, incremental ``PickBestIndex`` with it); passing a callable runs
    the O(n) reference scan per pick.
    """

    def _core_type_key(hc: HardT) -> tuple[int, int]:
        k = getattr(hc, "_type_key_cache", None)
        if k is None:
            k = (int(hc.get_input_count()), int(hc.get_output_count()))
            try:
                setattr(hc, "_type_key_cache", k)
            except AttributeError:
                pass
        return k

    def _identity_remove(lst, target) -> bool:
        """Remove ``target`` by identity (``is``); dataclass ``__eq__`` would remove the wrong equal-but-distinct instance."""
        for i, x in enumerate(lst):
            if x is target:
                del lst[i]
                return True
        return False

    def _soft_tg(sc) -> int | None:
        tg = getattr(sc, "residency_class_id", None)
        return int(tg) if tg is not None else None

    def _hard_tg(hc) -> int | None:
        tg = getattr(hc, "residency_class_id", None)
        return int(tg) if tg is not None else None

    unused_by_type: dict[tuple[int, int], list] = {}
    for hc in unused_hardcores:
        unused_by_type.setdefault(_core_type_key(hc), []).append(hc)

    used_by_tg: dict[int | None, list[int]] = {}

    def _register_used(idx: int, hc) -> None:
        used_by_tg.setdefault(_hard_tg(hc), []).append(idx)

    for _i, _hc in enumerate(used_hardcores):
        _register_used(_i, _hc)

    def _reindex_used_tg(idx: int) -> None:
        """Move a used-core index into the tg bucket matching its current hardcore tg."""
        hc = used_hardcores[idx]
        new_tg = _hard_tg(hc)
        for key, bucket in used_by_tg.items():
            if key == new_tg:
                continue
            if idx in bucket:
                bucket.remove(idx)
                used_by_tg.setdefault(new_tg, []).append(idx)
                return

    _use_sort_fast_path = pick_softcore is None and split_softcore is None
    if _use_sort_fast_path:
        softcores.sort(key=lambda c: max(c.get_input_count(), c.get_output_count()))
    pick_index = (
        PickBestIndex(softcores)
        if pick_softcore is None and split_softcore is not None
        else None
    )

    def _pop_unused_of_type(type_key: tuple[int, int]):
        bucket = unused_by_type.get(type_key)
        if not bucket:
            return None
        hc = bucket.pop()
        if not bucket:
            del unused_by_type[type_key]
        return hc

    while softcores:
        if _use_sort_fast_path:
            core = softcores[-1]
        elif pick_index is not None:
            core = pick_index.pick()
        else:
            assert pick_softcore is not None
            core = pick_softcore(softcores)

        core_tg = _soft_tg(core)

        target_idx = None
        best_cost: tuple[int, int] | None = None
        candidate_buckets = [used_by_tg.get(core_tg, ())]
        candidate_buckets.append(used_by_tg.get(None, ()))
        for bucket in candidate_buckets:
            for idx in bucket:
                hc = used_hardcores[idx]
                if is_mapping_possible(core, hc):
                    cost = placement_cost(
                        core, hc, remaining_capacity=_remaining_capacity(core, hc)
                    )
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        target_idx = idx

        if (
            target_idx is None
            and split_softcore is not None
            and _is_splittable(core)
        ):
            candidates = list(used_by_tg.get(core_tg, ()))
            candidates.extend(used_by_tg.get(None, ()))
            if _try_split_into_used(
                core, used_hardcores, split_softcore, split_threshold,
                place, softcores,
                candidate_indices=candidates,
                pick_index=pick_index,
            ):
                continue

        if target_idx is None:
            if not unused_hardcores:
                raise RuntimeError("No more hard cores available")

            chosen: tuple[tuple[int, int], HardT] | None = None
            chosen_score = float("inf")
            for type_key, bucket in unused_by_type.items():
                if not bucket:
                    continue
                hc = bucket[-1]
                if is_mapping_possible(core, hc):
                    waste = _placement_waste(core, hc)
                    abundance = len(bucket)
                    score = waste / max(abundance, 1)
                    if score < chosen_score:
                        chosen_score = score
                        chosen = (type_key, hc)

            if chosen is None:
                s_a = int(core.get_input_count())
                s_n = int(core.get_output_count())
                avail = {k: len(b) for k, b in unused_by_type.items()}

                fused_hc = None
                if fuse_hardcores is not None:
                    for hc_type, bucket in list(unused_by_type.items()):
                        qty = len(bucket)
                        c_a, c_n = hc_type
                        if c_n >= s_n and c_a * qty >= s_a:
                            qty_needed = (s_a + c_a - 1) // c_a
                            fusing_hcs = bucket[-qty_needed:]
                            temp_fused = fuse_hardcores(fusing_hcs)
                            if is_mapping_possible(core, temp_fused):
                                fused_hc = temp_fused
                                for hc in fusing_hcs:
                                    _identity_remove(unused_hardcores, hc)
                                del bucket[-qty_needed:]
                                if not bucket:
                                    del unused_by_type[hc_type]
                                break

                if (
                    fused_hc is None
                    and split_softcore is not None
                    and _is_splittable(core)
                ):
                    if _try_split_into_unused(
                        core, used_hardcores, unused_hardcores,
                        split_softcore, split_threshold, place, softcores,
                        type_counts=None, type_key=None,
                        unused_by_type=unused_by_type,
                        hard_type_key=_core_type_key,
                        pick_index=pick_index,
                    ):
                        continue

                if fused_hc is not None:
                    used_hardcores.append(fused_hc)
                    new_idx = len(used_hardcores) - 1
                    _register_used(new_idx, fused_hc)
                    target_idx = new_idx
                else:
                    raise RuntimeError(
                        f"No more hard cores available: softcore ({s_a} axons, "
                        f"{s_n} neurons) does not fit any unused type "
                        f"even with coalescing. Remaining types: {avail}"
                    )
            else:
                chosen_type, chosen_hc = chosen
                _identity_remove(unused_hardcores, chosen_hc)
                bucket = unused_by_type[chosen_type]
                if bucket and bucket[-1] is chosen_hc:
                    bucket.pop()
                else:
                    _identity_remove(bucket, chosen_hc)
                if not bucket:
                    del unused_by_type[chosen_type]
                used_hardcores.append(chosen_hc)
                new_idx = len(used_hardcores) - 1
                _register_used(new_idx, chosen_hc)
                target_idx = new_idx

        place(target_idx, used_hardcores[target_idx], core)
        _reindex_used_tg(target_idx)
        if _use_sort_fast_path:
            softcores.pop()
        else:
            softcores.remove(core)
            if pick_index is not None:
                pick_index.discard(core)


def reference_run_placement(*, softcores, used_hardcores, unused_hardcores, materializer, allow_neuron_splitting,
                            allow_coalescing=True):
    reference_greedy_pack_softcores(
        softcores=softcores, used_hardcores=used_hardcores, unused_hardcores=unused_hardcores,
        is_mapping_possible=materializer.is_mapping_possible, place=materializer.place,
        fuse_hardcores=(materializer.fuse_hardcores if allow_coalescing else None),
        split_softcore=(materializer.split_softcore if allow_neuron_splitting else None),
    )


def _make_instances(core_types: Sequence[LayoutHardCoreType]) -> List[LayoutHardCoreInstance]:
    out: List[LayoutHardCoreInstance] = []
    for t in core_types:
        for _ in range(int(t.count)):
            out.append(LayoutHardCoreInstance(axons_per_core=int(t.max_axons), neurons_per_core=int(t.max_neurons)))
    return out



def reference_pack_layout(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    collect_placements: bool = False,
) -> LayoutPackingResult:
    """Pack layout-only softcores into a limited pool of hardware cores.

    ``allow_neuron_splitting`` splits along the neuron dimension; ``allow_coalescing``
    pre-expands over-wide softcores into axon-coalescing fragments before packing.
    ``collect_placements`` additionally seals which hardcore each input softcore
    landed on (``result.placements``: ``(origin index, hardcore index)`` per
    placed unit; fragments repeat their origin)."""

    used_hardcores: List[LayoutHardCoreInstance] = []
    unused_hardcores: List[LayoutHardCoreInstance] = _make_instances(core_types)

    unmapped = list(copy.deepcopy(list(softcores)))
    pre_coalesce_count = len(unmapped)

    origins = list(range(pre_coalesce_count))
    coalescing_group_sizes: Tuple[int, ...] = ()
    if allow_coalescing:
        unmapped, coalescing_group_sizes, origins = _expand_for_axon_coalescing(
            unmapped, core_types,
        )
    coalesced_fragment_count = len(unmapped) - pre_coalesce_count

    for i, sc in enumerate(unmapped):
        if sc.name is None:
            unmapped[i] = LayoutSoftCoreSpec(
                input_count=sc.input_count,
                output_count=sc.output_count,
                residency_class_id=sc.residency_class_id,
                latency_tag=sc.latency_tag,
                name=f"__sc_{i}",
            )

    materializer = LayoutMaterializer(collect_placements=collect_placements)
    if collect_placements:
        materializer.origin_of = {
            id(sc): origins[i] for i, sc in enumerate(unmapped)
        }

    try:
        reference_run_placement(
            softcores=unmapped,
            used_hardcores=used_hardcores,
            unused_hardcores=unused_hardcores,
            materializer=materializer,
            allow_neuron_splitting=allow_neuron_splitting,
        )
    except RuntimeError as e:
        # ``run_placement`` fuses hardcores whatever ``allow_coalescing`` declared,
        # so the axon dimension is never tight enough here to prove anything.
        verdict = failed_pack_verdict(
            softcores, core_types,
            neurons_may_split=allow_neuron_splitting, axons_may_spread=True,
        )
        return LayoutPackingResult(
            feasible=False,
            cores_used=0,
            total_capacity=0,
            used_area=0,
            unused_area_total=0,
            avg_unused_area_per_core=float("inf"),
            unusable_space_total=0,
            avg_unusable_space_per_core=0.0,
            error=tag_with_verdict(verdict, str(e)),
            verdict=verdict,
        )

    cores_used = len(used_hardcores)
    total_capacity = sum(int(h.capacity) for h in used_hardcores)
    used_area = sum(int(h.used_area) for h in used_hardcores)
    unused_total = int(total_capacity - used_area)
    avg_unused = float(unused_total / cores_used) if cores_used > 0 else float("inf")

    unusable_total = sum(int(h.unusable_space) for h in used_hardcores)
    avg_unusable_space = (
        float(unusable_total / cores_used) if cores_used > 0 else 0.0
    )

    snapshots = tuple(
        LayoutCoreSnapshot(
            axons_per_core=h.axons_per_core,
            neurons_per_core=h.neurons_per_core,
            used_axons=h.axons_per_core - h.available_axons,
            used_neurons=h.neurons_per_core - h.available_neurons,
            used_area=h.used_area,
            softcore_count=h.softcore_count,
        )
        for h in used_hardcores
    )

    placements: Optional[Tuple[Tuple[int, int], ...]] = None
    if collect_placements:
        hardcore_index = {id(h): i for i, h in enumerate(used_hardcores)}
        resolved: List[Tuple[int, int]] = []
        for origin, hardcore in materializer.placed:
            while id(hardcore) in materializer.fused_alias:
                hardcore = materializer.fused_alias[id(hardcore)]
            resolved.append((int(origin), hardcore_index[id(hardcore)]))
        placements = tuple(resolved)

    return LayoutPackingResult(
        feasible=True,
        cores_used=cores_used,
        total_capacity=int(total_capacity),
        used_area=int(used_area),
        unused_area_total=unused_total,
        avg_unused_area_per_core=avg_unused,
        unusable_space_total=int(unusable_total),
        avg_unusable_space_per_core=avg_unusable_space,
        error=None,
        used_core_softcore_counts=tuple(hc.softcore_count for hc in used_hardcores),
        used_core_snapshots=snapshots,
        coalesced_fragment_count=coalesced_fragment_count,
        split_fragment_count=materializer.split_counter,
        coalescing_group_sizes=coalescing_group_sizes if coalescing_group_sizes else None,
        split_counts_per_sc=(
            tuple(materializer.split_lineage.values())
            if materializer.split_lineage
            else None
        ),
        placements=placements,
    )


def reference_split_softcores_by_capacity(
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

    hw_types = list(core_types)

    def _packs(batch: Sequence[LayoutSoftCoreSpec]) -> bool:
        if not batch:
            return True
        pr = reference_pack_layout(
            softcores=list(batch),
            core_types=hw_types,
            allow_neuron_splitting=allow_splitting,
            allow_coalescing=allow_coalescing,
        )
        return bool(pr.feasible)

    def _halve_until_packs(group: List[LayoutSoftCoreSpec]) -> List[List[LayoutSoftCoreSpec]]:
        """Binary-halve a single latency group until each piece packs."""
        if len(group) <= 1 or _packs(group):
            return [list(group)]
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


def reference_estimate_passes_for_layout_validated(
    softcores: Sequence[LayoutSoftCoreSpec],
    max_cores_per_pass: int,
    *,
    max_hw_axons: int = 0,
    max_hw_neurons: int = 0,
    allow_coalescing: bool = False,
    allow_splitting: bool = False,
    core_types: Sequence,
    coalescing_group_ids: "Sequence[int | None] | None" = None,
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

    by_segment: dict[int, List[LayoutSoftCoreSpec]] = defaultdict(list)
    for sc in softcores:
        seg = sc.segment_id if sc.segment_id is not None else 0
        by_segment[seg].append(sc)

    # Aligned by identity so the per-segment grouping cannot mispair a
    # coalesced fragment with another spec's group id.
    group_of = (
        {id(sc): gid for sc, gid in zip(softcores, coalescing_group_ids)}
        if allow_coalescing and coalescing_group_ids is not None else None
    )

    all_pass_lists: List[List[LayoutSoftCoreSpec]] = []
    all_ok = True
    for seg_id in sorted(by_segment.keys()):
        seg_cores = by_segment[seg_id]
        sub_segments = reference_split_softcores_by_capacity(
            seg_cores,
            hw_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_splitting,
            coalescing_group_ids=(
                [group_of[id(sc)] for sc in seg_cores]
                if group_of is not None else None
            ),
        )
        if not sub_segments:
            continue
        for sub in sub_segments:
            pr = reference_pack_layout(
                softcores=sub,
                core_types=hw_types,
                allow_neuron_splitting=allow_splitting,
                allow_coalescing=allow_coalescing,
            )
            if not pr.feasible:
                all_ok = False
        all_pass_lists.extend(sub_segments)

    return len(all_pass_lists), all_pass_lists, all_ok
