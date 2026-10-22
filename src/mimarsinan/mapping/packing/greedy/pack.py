from __future__ import annotations

from typing import Callable, List, Union

from mimarsinan.mapping.packing.canonical import (
    HardT,
    SoftT,
    _placement_waste,
    _remaining_capacity,
)
from mimarsinan.mapping.packing.greedy.pool import CorePool, ListCorePool
from mimarsinan.mapping.packing.greedy.remaining import RemainingSoftcores
from mimarsinan.mapping.packing.greedy.split import (
    _is_splittable,
    _try_split_into_used,
    _try_split_into_unused,
)
from mimarsinan.mapping.packing.greedy.used_index import UsedBuckets
from mimarsinan.mapping.packing.pick_index import PickBestIndex
from mimarsinan.mapping.packing.placement_cost import placement_cost

__all__ = ["greedy_pack_softcores"]


def _core_type_key(hc) -> tuple[int, int]:
    k = getattr(hc, "_type_key_cache", None)
    if k is None:
        k = (int(hc.get_input_count()), int(hc.get_output_count()))
        try:
            setattr(hc, "_type_key_cache", k)
        except AttributeError:
            pass
    return k


def _soft_tg(sc) -> int | None:
    tg = getattr(sc, "residency_class_id", None)
    return int(tg) if tg is not None else None


def greedy_pack_softcores(
    *,
    softcores: List[SoftT],
    used_hardcores: List[HardT],
    unused_hardcores: Union[List[HardT], CorePool],
    is_mapping_possible: Callable[[SoftT, HardT], bool],
    place: Callable[[int, HardT, SoftT], None],
    pick_softcore: Callable[[List[SoftT]], SoftT] | None = None,
    fuse_hardcores: Callable[[List[HardT]], HardT] | None = None,
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]] | None = None,
    split_threshold: float = 0.2,
    canonical_feasibility: bool = False,
) -> None:
    """Greedy bin-packing loop shared by the runtime and layout-only packers.

    Priority per softcore: exact fit in a used core, split into a used core,
    exact fit in an unused core, fusion, split into an unused core, else raise.
    Mutates ``softcores``/``used_hardcores``/``unused_hardcores`` in place.
    ``pick_softcore=None`` (the default) keeps the canonical
    ``pick_best_softcore`` order, accelerated (sorted fast path without
    splitting, incremental ``PickBestIndex`` with it); passing a callable runs
    the O(n) reference scan per pick. ``unused_hardcores`` is a list or a
    ``CorePool`` (cores built on demand); ``canonical_feasibility`` declares
    that ``is_mapping_possible`` is the canonical predicate, which lets the
    best-fit and split scans read one core per core state.
    """
    pool = unused_hardcores if isinstance(unused_hardcores, CorePool) else ListCorePool(
        unused_hardcores, _core_type_key)
    used = UsedBuckets(used_hardcores, indexed=canonical_feasibility)

    _use_sort_fast_path = pick_softcore is None and split_softcore is None
    if _use_sort_fast_path:
        softcores.sort(key=lambda c: max(c.get_input_count(), c.get_output_count()))
    pick_index = (
        PickBestIndex(softcores)
        if pick_softcore is None and split_softcore is not None
        else None
    )
    remaining = RemainingSoftcores(softcores) if pick_index is not None else softcores

    try:
        while remaining:
            if _use_sort_fast_path:
                core = softcores[-1]
            elif pick_index is not None:
                core = pick_index.pick()
            else:
                assert pick_softcore is not None
                core = pick_softcore(softcores)

            core_tg = _soft_tg(core)
            bank_affine = getattr(core, "weight_bank_id", None) is not None

            target_idx = None
            best_cost: tuple[int, int] | None = None
            for idx in used.scan_order(core_tg, every_core=bank_affine):
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
                split_idx = _try_split_into_used(
                    core, used_hardcores, split_softcore, split_threshold,
                    place, remaining,
                    candidate_indices=used.scan_order(core_tg, every_core=False),
                    pick_index=pick_index,
                )
                if split_idx is not None:
                    used.restate(split_idx)
                    continue

            if target_idx is None:
                if not pool.total():
                    raise RuntimeError("No more hard cores available")

                chosen: tuple[tuple[int, int], HardT] | None = None
                chosen_score = float("inf")
                for type_key in pool.keys():
                    hc = pool.last(type_key)[0]
                    if is_mapping_possible(core, hc):
                        waste = _placement_waste(core, hc)
                        abundance = pool.count(type_key)
                        score = waste / max(abundance, 1)
                        if score < chosen_score:
                            chosen_score = score
                            chosen = (type_key, hc)

                if chosen is None:
                    s_a = int(core.get_input_count())
                    s_n = int(core.get_output_count())
                    avail = {k: pool.count(k) for k in pool.keys()}

                    fused_hc = None
                    if fuse_hardcores is not None:
                        for hc_type in pool.keys():
                            qty = pool.count(hc_type)
                            c_a, c_n = hc_type
                            if c_n >= s_n and c_a * qty >= s_a:
                                # A 0-axon softcore needs no core; the type's whole group fuses, as the
                                # group's last-0 slice (``group[-0:]``) always did.
                                qty_needed = (s_a + c_a - 1) // c_a or qty
                                temp_fused = fuse_hardcores(pool.last(hc_type, qty_needed))
                                if is_mapping_possible(core, temp_fused):
                                    fused_hc = temp_fused
                                    pool.take_last(hc_type, qty_needed)
                                    break

                    if (
                        fused_hc is None
                        and split_softcore is not None
                        and _is_splittable(core)
                    ):
                        if _try_split_into_unused(
                            core, used_hardcores, pool,
                            split_softcore, split_threshold, place, remaining,
                            pick_index=pick_index,
                        ):
                            continue

                    if fused_hc is not None:
                        used_hardcores.append(fused_hc)
                        target_idx = len(used_hardcores) - 1
                        used.register(target_idx)
                    else:
                        raise RuntimeError(
                            f"No more hard cores available: softcore ({s_a} axons, "
                            f"{s_n} neurons) does not fit any unused type "
                            f"even with coalescing. Remaining types: {avail}"
                        )
                else:
                    chosen_type, chosen_hc = chosen
                    pool.take_last(chosen_type)
                    used_hardcores.append(chosen_hc)
                    target_idx = len(used_hardcores) - 1
                    used.register(target_idx)

            place(target_idx, used_hardcores[target_idx], core)
            used.reclass(target_idx)
            if _use_sort_fast_path:
                softcores.pop()
            else:
                remaining.remove(core)
                if pick_index is not None:
                    pick_index.discard(core)
    finally:
        pool.finish()
        if isinstance(remaining, RemainingSoftcores):
            remaining.finish()
