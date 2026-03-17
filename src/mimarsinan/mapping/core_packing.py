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

    unmapped_cores.sort(key=lambda core: core.get_input_count(), reverse=True)
    core_a = unmapped_cores[0]

    unmapped_cores.sort(key=lambda core: core.get_output_count(), reverse=True)
    core_b = unmapped_cores[0]

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
) -> bool:
    """Try to split *core* into a partially-filled used hardware core.

    Picks the used core with the **most** remaining neurons (to maximise
    the useful fragment size) among cores where:
    - axons fit
    - neurons do NOT fit (the whole core is too wide)
    - remaining neurons > split_threshold × total neurons

    Returns True if a split was performed, False otherwise.
    """
    best_idx = None
    best_avail_n = -1
    for idx, hc in enumerate(used_hardcores):
        avail_a = getattr(hc, "available_axons", int(hc.get_input_count()))
        avail_n = getattr(hc, "available_neurons", int(hc.get_output_count()))
        total_n = int(hc.get_output_count())
        if (
            core.get_input_count() <= avail_a
            and core.get_output_count() > avail_n
            and avail_n > split_threshold * total_n
            and avail_n > best_avail_n
        ):
            # Also verify threshold/latency compatibility by checking that a
            # hypothetical full-fit would pass (except neuron count).  We rely
            # on the caller's is_mapping_possible being captured in the closure
            # that constructed split_softcore — for now we check the attributes
            # directly (same logic as HardCoreMapping.map).
            hc_threshold = getattr(hc, "threshold", None)
            core_threshold = getattr(core, "threshold", None)
            if hc_threshold is not None and core_threshold is not None:
                diff_rate = abs(core_threshold - hc_threshold) / (hc_threshold + 1)
                if diff_rate > 0.1:
                    continue
            hc_latency = getattr(hc, "latency", None)
            core_latency = getattr(core, "latency", None)
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
    used_hardcores.append(best_hc)
    unused_hardcores.remove(best_hc)
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
        return (int(hc.get_input_count()), int(hc.get_output_count()))

    while softcores:
        core = pick_softcore(softcores)

        # --- Try used cores: pick the one with the tightest fit. ---
        target_idx = None
        best_remaining = float("inf")
        for idx, hc in enumerate(used_hardcores):
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
            if _try_split_into_used(
                core, used_hardcores, split_softcore, split_threshold,
                place, softcores,
            ):
                continue

        if target_idx is None:
            # --- No used core fits: allocate an unused core. ---
            if not unused_hardcores:
                raise RuntimeError("No more hard cores available")

            type_counts = Counter(_core_type_key(hc) for hc in unused_hardcores)

            chosen_unused = None
            chosen_score = float("inf")
            for hc in unused_hardcores:
                if is_mapping_possible(core, hc):
                    waste = _placement_waste(core, hc)
                    abundance = type_counts[_core_type_key(hc)]
                    score = waste / max(abundance, 1)
                    if score < chosen_score:
                        chosen_unused = hc
                        chosen_score = score

            if chosen_unused is None:
                s_a = int(core.get_input_count())
                s_n = int(core.get_output_count())
                avail = {k: v for k, v in type_counts.items()}

                fused_hc = None
                if fuse_hardcores is not None:
                    for hc_type, qty in type_counts.items():
                        c_a, c_n = hc_type
                        if c_n >= s_n and c_a * qty >= s_a:
                            qty_needed = (s_a + c_a - 1) // c_a
                            fusing_hcs = [hc for hc in unused_hardcores if _core_type_key(hc) == hc_type][:qty_needed]
                            temp_fused = fuse_hardcores(fusing_hcs)
                            if is_mapping_possible(core, temp_fused):
                                fused_hc = temp_fused
                                for hc in fusing_hcs:
                                    unused_hardcores.remove(hc)
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
                    ):
                        continue

                if fused_hc is not None:
                    used_hardcores.append(fused_hc)
                    target_idx = len(used_hardcores) - 1
                else:
                    raise RuntimeError(
                        f"No more hard cores available: softcore ({s_a} axons, "
                        f"{s_n} neurons) does not fit any unused type "
                        f"even with coalescing. Remaining types: {avail}"
                    )
            else:
                used_hardcores.append(chosen_unused)
                unused_hardcores.remove(chosen_unused)
                target_idx = len(used_hardcores) - 1

        place(target_idx, used_hardcores[target_idx], core)
        softcores.remove(core)
