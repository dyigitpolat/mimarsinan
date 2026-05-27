from __future__ import annotations

from collections import Counter
from typing import Callable, List

from mimarsinan.mapping.packing.canonical import (
    HardCoreLike,
    HardT,
    SoftCoreLike,
    SoftT,
    _placement_waste,
    _remaining_capacity,
    pick_best_softcore,
)
from mimarsinan.mapping.packing.greedy.split import (
    _is_splittable,
    _try_split_into_used,
    _try_split_into_unused,
)

__all__ = ["greedy_pack_softcores"]

def greedy_pack_softcores(
    *,
    softcores: List[SoftT],
    used_hardcores: List[HardT],
    unused_hardcores: List[HardT],
    is_mapping_possible: Callable[[SoftT, HardT], bool],
    place: Callable[[int, HardT, SoftT], None],
    pick_softcore: Callable[[List[SoftT]], SoftT] = pick_best_softcore,
    fuse_hardcores: Callable[[List[HardT]], HardT] | None = None,
    split_softcore: Callable[[SoftT, int], tuple[SoftT, SoftT]] | None = None,
    split_threshold: float = 0.2,
) -> None:
    """
    Greedy packing loop shared by:
    - real HardCoreMapping (parameterized mapping)
    - layout-only packer (shape-only evaluation for search)

    **Used-core selection** — among all feasible already-allocated cores,
    pick the one with the minimum *remaining capacity* after placement.
    This concentrates softcores into tightly-fitting cores, leaving other
    used cores available for differently-shaped softcores.

    **Unused-core selection** — among all feasible un-allocated cores,
    pick the one with the minimum *scarcity-adjusted waste*.  The raw
    waste ``h_a · s_n + s_a · h_n − 2 · s_a · s_n`` is divided by the
    number of remaining instances of that core type.  This naturally
    preserves scarce hardware types for softcores that have no
    alternative, while still preferring shape-matched types when
    abundance is equal.

    **Neuron splitting** (optional) — when ``split_softcore`` is provided
    and the current soft core's neurons exceed a hardware core's remaining
    (or total) neuron width, the soft core is split column-wise.  Fragment 1
    fills the available neurons; Fragment 2 goes back into the unmapped pool.
    Splitting is only attempted when the available neuron width exceeds
    ``split_threshold`` × the hardware core's total neuron width (default 20%).
    This avoids excessive fragmentation and traffic duplication.

    Priority order:
    1. Exact fit in used core
    2. Split into used core (remaining neurons > 20% threshold)
    3. Exact fit in unused core
    4. Fusion
    5. Split into unused core (last resort)
    6. RuntimeError

    This mutates:
    - used_hardcores (may append newly allocated hardcores)
    - unused_hardcores (removes allocated hardcores)
    - softcores (removes packed softcores)
    """

    def _core_type_key(hc: HardT) -> tuple[int, int]:
        k = getattr(hc, "_type_key_cache", None)
        if k is None:
            k = (int(hc.get_input_count()), int(hc.get_output_count()))
            try:
                hc._type_key_cache = k
            except AttributeError:
                pass
        return k

    def _identity_remove(lst, target) -> bool:
        """Remove ``target`` from ``lst`` by *identity* (``is``), not ``__eq__``.

        LayoutHardCoreInstance uses dataclass-generated ``__eq__`` which treats
        two fresh instances of the same (axons, neurons) as equal until one is
        mutated.  Using ``list.remove`` would then remove the wrong object when
        multiple equivalent fresh instances share a pool.
        """
        for i, x in enumerate(lst):
            if x is target:
                del lst[i]
                return True
        return False

    def _soft_tg(sc) -> int | None:
        tg = getattr(sc, "threshold_group_id", None)
        return int(tg) if tg is not None else None

    def _hard_tg(hc) -> int | None:
        tg = getattr(hc, "threshold_group_id", None)
        return int(tg) if tg is not None else None

    # Unused pool organised by type: {type_key: [instances...]}.  Most unused
    # cores of the same type are interchangeable at decision time, so we only
    # evaluate one representative per type (one waste/is_mapping_possible call
    # per type instead of per instance → ~N× speedup at cifar_vit scale).
    unused_by_type: dict[tuple[int, int], list] = {}
    for hc in unused_hardcores:
        unused_by_type.setdefault(_core_type_key(hc), []).append(hc)

    # Used-core index bucketed by threshold_group_id.  Each softcore only
    # consults cores with a matching tg (plus the "not-yet-pinned" bucket
    # whose tg is None — empty hw cores that haven't had any softcore
    # placed yet).  Each entry is the index into used_hardcores so we can
    # dereference the actual instance.
    used_by_tg: dict[int | None, list[int]] = {}

    def _register_used(idx: int, hc) -> None:
        used_by_tg.setdefault(_hard_tg(hc), []).append(idx)

    # Pre-populate from any pre-existing used cores — callers may invoke
    # greedy_pack_softcores multiple times with an accumulating ``used``
    # list (e.g. scheduled mapping passes).
    for _i, _hc in enumerate(used_hardcores):
        _register_used(_i, _hc)

    def _reindex_used_tg(idx: int) -> None:
        """Move a used-core index into the tg bucket that matches its current
        hardcore tg (called after the first softcore is placed and sets it)."""
        hc = used_hardcores[idx]
        new_tg = _hard_tg(hc)
        for key, bucket in used_by_tg.items():
            if key == new_tg:
                continue
            if idx in bucket:
                bucket.remove(idx)
                used_by_tg.setdefault(new_tg, []).append(idx)
                return

    # Fast path for the default pick heuristic when splitting is off:
    # pre-sort ascending by max(input, output) and process from the end.
    # Saves two O(n) max() scans per placement (→ O(n²) per pack), which
    # dominates on large models.  Disabled when split_softcore is active
    # because splits append smaller fragments that would break the ordering.
    _use_sort_fast_path = (
        pick_softcore is pick_best_softcore and split_softcore is None
    )
    if _use_sort_fast_path:
        softcores.sort(key=lambda c: max(c.get_input_count(), c.get_output_count()))

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
        else:
            core = pick_softcore(softcores)

        core_tg = _soft_tg(core)

        target_idx = None
        best_remaining = float("inf")
        candidate_buckets = [used_by_tg.get(core_tg, ())]
        candidate_buckets.append(used_by_tg.get(None, ()))
        for bucket in candidate_buckets:
            for idx in bucket:
                hc = used_hardcores[idx]
                if is_mapping_possible(core, hc):
                    rem = _remaining_capacity(core, hc)
                    if rem < best_remaining:
                        best_remaining = rem
                        target_idx = idx

        if (
            target_idx is None
            and split_softcore is not None
            and _is_splittable(core)
        ):
            # Restrict the split-target search to compatible tg buckets
            # (+ the unassigned bucket) — avoids a full O(|used|) scan
            # when the used pool is large.
            candidates = list(used_by_tg.get(core_tg, ()))
            candidates.extend(used_by_tg.get(None, ()))
            if _try_split_into_used(
                core, used_hardcores, split_softcore, split_threshold,
                place, softcores,
                candidate_indices=candidates,
            ):
                continue

        if target_idx is None:
            if not unused_hardcores:
                raise RuntimeError("No more hard cores available")

            # Only evaluate one representative per type — all instances of a
            # type are interchangeable for the scoring function.
            chosen_type = None
            chosen_score = float("inf")
            chosen_hc = None
            for type_key, bucket in unused_by_type.items():
                if not bucket:
                    continue
                hc = bucket[-1]  # representative
                if is_mapping_possible(core, hc):
                    waste = _placement_waste(core, hc)
                    abundance = len(bucket)
                    score = waste / max(abundance, 1)
                    if score < chosen_score:
                        chosen_type = type_key
                        chosen_score = score
                        chosen_hc = hc

            if chosen_hc is None:
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
                                # Drop the last ``qty_needed`` slice from bucket
                                # by identity — matches ``fusing_hcs`` since we
                                # sliced from the same list.
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
                _identity_remove(unused_hardcores, chosen_hc)
                bucket = unused_by_type[chosen_type]
                # chosen_hc was bucket[-1] (the representative); pop it.
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
        # The place callback may have just assigned the hardcore's
        # threshold_group_id (first softcore in this core).  Re-bucket so
        # subsequent lookups by tg find it.
        _reindex_used_tg(target_idx)
        if _use_sort_fast_path:
            softcores.pop()
        else:
            softcores.remove(core)
