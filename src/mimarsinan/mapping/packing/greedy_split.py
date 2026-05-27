from __future__ import annotations

from collections import Counter
from typing import Callable, List

from mimarsinan.mapping.packing.canonical import (
    HardCoreLike,
    HardT,
    SoftCoreLike,
    SoftT,
    pick_best_softcore,
)

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

