from __future__ import annotations

from collections import Counter
from typing import Callable, List, Protocol, TypeVar


class SoftCoreLike(Protocol):
    def get_input_count(self) -> int: ...

    def get_output_count(self) -> int: ...


class HardCoreLike(Protocol):
    def get_input_count(self) -> int: ...

    def get_output_count(self) -> int: ...


SoftT = TypeVar("SoftT", bound=SoftCoreLike)
HardT = TypeVar("HardT", bound=HardCoreLike)


def _capacity(hc: HardCoreLike) -> int:
    """Total capacity (area) of a hardware core."""
    return int(hc.get_input_count()) * int(hc.get_output_count())


def _placement_waste(soft: SoftCoreLike, hard: HardCoreLike) -> int:
    """
    Estimate the area wasted by placing *soft* into *hard*.

    In diagonal packing each softcore occupies a block of axons × neurons
    along the diagonal.  The dead zone for a single placement is the
    L-shaped region that neither the softcore nor any future placement
    can use::

        waste = (h_a − s_a) · s_n + s_a · (h_n − s_n)
              = h_a · s_n + s_a · h_n − 2 · s_a · s_n

    This naturally penalises aspect-ratio mismatches: a softcore that is
    narrow in axons but wide in neurons will score much lower on a
    narrow-axon core type than on a square core of the same total area.
    """
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    h_a = int(hard.get_input_count())
    h_n = int(hard.get_output_count())
    return h_a * s_n + s_a * h_n - 2 * s_a * s_n


def _remaining_capacity(soft: SoftCoreLike, hard: HardCoreLike) -> int:
    """
    Remaining usable area after placing *soft* into *hard*.

    Uses ``available_axons`` / ``available_neurons`` when present (used
    cores), otherwise falls back to the full core dimensions (unused cores).

    A lower value indicates a tighter fit — the softcore fills up the
    remaining space more completely, reducing fragmentation.
    """
    avail_a = getattr(hard, "available_axons", int(hard.get_input_count()))
    avail_n = getattr(hard, "available_neurons", int(hard.get_output_count()))
    s_a = int(soft.get_input_count())
    s_n = int(soft.get_output_count())
    return (avail_a - s_a) * (avail_n - s_n)


def pick_best_softcore(unmapped_cores: List[SoftT]) -> SoftT:
    """
    Heuristic used by the existing HardCoreMapping:
    - pick the core with max input count
    - pick the core with max output count
    - choose whichever is more extreme
    """
    if not unmapped_cores:
        raise ValueError("unmapped_cores is empty")

    # Single linear scan is O(n) vs two full sorts O(n log n) — same winner.
    core_a = max(unmapped_cores, key=lambda c: c.get_input_count())
    core_b = max(unmapped_cores, key=lambda c: c.get_output_count())

    if core_a.get_input_count() > core_b.get_output_count():
        return core_a
    return core_b


def _is_splittable(core: SoftCoreLike) -> bool:
    """Check whether a soft core is eligible for neuron splitting.

    Cores that are part of a coalescing group (axon-split fragments) must
    not be neuron-split — they are already partial results.
    """
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
) -> bool:
    """Try to split *core* into a partially-filled used hardware core.

    Picks the used core with the **most** remaining neurons (to maximise
    the useful fragment size) among cores where:
    - axons fit
    - neurons do NOT fit (the whole core is too wide)
    - remaining neurons > split_threshold × total neurons

    ``candidate_indices`` restricts the scan to a pre-filtered subset
    (e.g. only cores with a compatible threshold_group_id).  When omitted
    the full used-core list is scanned.

    Returns True if a split was performed, False otherwise.
    """
    best_idx = None
    best_avail_n = -1
    core_tg = getattr(core, "threshold_group_id", None)
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
            hc_tg = getattr(hc, "threshold_group_id", None)
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
) -> bool:
    """Try to split *core* into a fresh unused hardware core.

    Last resort when the core's neurons exceed every core type's total
    neuron capacity but axons fit.  No threshold is applied here — the
    fragment size equals the hardware core's neuron capacity, which is
    always a reasonable size (the hardware designer chose it).
    """
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
    # Identity-based removal — avoids dataclass __eq__ removing the wrong instance.
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
    return True


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

        # --- Try used cores (only those with a compatible threshold group). ---
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

        # --- Try neuron splitting into a used core ---
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
            # --- No used core fits: allocate an unused core. ---
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

                # --- Try neuron splitting into an unused core (last resort) ---
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
