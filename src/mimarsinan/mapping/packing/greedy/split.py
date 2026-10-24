from __future__ import annotations

from typing import Any, Callable, List, Optional

from mimarsinan.mapping.packing.canonical import (
    HardT,
    SoftCoreLike,
    SoftT,
    _read_latency,
)
from mimarsinan.mapping.packing.greedy.pool import CorePool
from mimarsinan.mapping.packing.pick_index import PickBestIndex


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
    softcores,
    *,
    candidate_indices: "list[int] | None" = None,
    pick_index: "PickBestIndex | None" = None,
) -> Optional[int]:
    """Split *core* into the used core with the most remaining neurons (axons fit, neurons don't, remainder above threshold).

    ``candidate_indices`` restricts the scan to a pre-filtered subset (the first of equals wins, so a subset
    holding each candidate state's earliest core selects as the whole set does); returns the index split into.
    """
    best_idx = None
    best_avail_n = -1
    core_tg = getattr(core, "residency_class_id", None)
    core_latency = _read_latency(core)  # [W04] latency or latency_tag: the layout packer splits as the runtime one
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
            hc_latency = _read_latency(hc)
            if hc_latency is not None:
                if core_latency is None or core_latency != hc_latency:
                    continue
            best_idx = idx
            best_avail_n = avail_n

    if best_idx is None:
        return None

    frag1, frag2 = split_softcore(core, best_avail_n)
    place(best_idx, used_hardcores[best_idx], frag1)
    softcores.remove(core)
    softcores.append(frag2)
    _swap_in_pick_index(pick_index, core, frag2)
    return best_idx


def _try_split_into_unused(
    core: SoftT,
    used_hardcores: List[HardT],
    pool: CorePool,
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]],
    split_threshold: float,
    place: Callable[[int, HardT, SoftT], None],
    softcores,
    *,
    pick_index: "PickBestIndex | None" = None,
) -> bool:
    """Last-resort split of *core* into a fresh unused core when its neurons exceed every core type's capacity.

    The core is the first unused one, in list order, of the most neurons that takes the axons; a pool's group
    holds one core shape, so that core is the head of the earliest such group.
    """
    best_key = None
    for key in pool.keys():
        total_a, total_n = key
        if core.get_input_count() <= total_a and core.get_output_count() > total_n and total_n > 0 and (
            best_key is None or total_n > best_key[1]
            or (total_n == best_key[1] and pool.first_position(key) < pool.first_position(best_key))
        ):
            best_key = key

    if best_key is None:
        return False

    frag1, frag2 = split_softcore(core, best_key[1])
    used_hardcores.append(pool.take_first(best_key))
    new_idx = len(used_hardcores) - 1
    place(new_idx, used_hardcores[new_idx], frag1)
    softcores.remove(core)
    softcores.append(frag2)
    _swap_in_pick_index(pick_index, core, frag2)
    return True


class _Window:
    """The first neuron window of a split as a feasibility predicate reads it, before the split is made."""

    def __init__(self, core, neurons: int) -> None:
        self._core, self._neurons = core, int(neurons)

    def get_input_count(self) -> int:
        return int(self._core.get_input_count())

    def get_output_count(self) -> int:
        return self._neurons

    def __getattr__(self, name):
        return getattr(self._core, name)


def _try_split_into_fused(
    core: SoftT,
    used_hardcores: List[HardT],
    pool: CorePool,
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]],
    fuse_hardcores: Callable[[List[HardT]], HardT],
    is_mapping_possible: Callable[[Any, HardT], bool],
    place: Callable[[int, HardT, SoftT], None],
    softcores,
    *,
    pick_index: "PickBestIndex | None" = None,
) -> bool:
    """[WI-5f] Neuron split before axon coalescing, for a softcore no type takes by fusion or by a split alone.

    Among the types whose unused cores can coalesce the axons, the one with the most neurons (the earliest on a tie)
    whose fused core accepts the window cuts the first window; that fragment goes on the fused core of ceil(axons / A)
    cores and the remainder returns to the queue, so a core of a axons and n neurons takes ceil(n / N) windows of
    ceil(a / A) cores each. ``is_mapping_possible`` reads the window (``_Window``, the core with the window's
    neurons) in place of the core, so its soft argument is any core-like object.
    """
    s_a, s_n = int(core.get_input_count()), int(core.get_output_count())
    candidates = []
    for position, key in enumerate(pool.keys()):
        c_a, c_n = key
        if c_a > 0 and 0 < c_n < s_n and -(-s_a // c_a) <= pool.count(key):
            candidates.append((-c_n, position, key))
    for _, _, key in sorted(candidates):
        needed = -(-s_a // key[0])
        fused = fuse_hardcores(pool.last(key, needed))
        if not is_mapping_possible(_Window(core, key[1]), fused):
            continue
        frag1, frag2 = split_softcore(core, key[1])
        pool.take_last(key, needed)
        used_hardcores.append(fused)
        place(len(used_hardcores) - 1, fused, frag1)
        softcores.remove(core)
        softcores.append(frag2)
        _swap_in_pick_index(pick_index, core, frag2)
        return True
    return False
