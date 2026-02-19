from __future__ import annotations

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


def greedy_pack_softcores(
    *,
    softcores: List[SoftT],
    used_hardcores: List[HardT],
    unused_hardcores: List[HardT],
    is_mapping_possible: Callable[[SoftT, HardT], bool],
    place: Callable[[int, HardT, SoftT], None],
    pick_softcore: Callable[[List[SoftT]], SoftT] = pick_best_softcore,
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
    pick the one with the minimum *placement waste*.  The waste metric
    ``h_a · s_n + s_a · h_n − 2 · s_a · s_n`` naturally penalises
    aspect-ratio mismatches, so a narrow-axon/wide-neuron softcore will
    prefer a similarly-shaped hardware core over a square one of the same
    total area.

    This mutates:
    - used_hardcores (may append newly allocated hardcores)
    - unused_hardcores (removes allocated hardcores)
    - softcores (removes packed softcores)
    """

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

        if target_idx is None:
            # --- No used core fits: allocate an unused core. ---
            if not unused_hardcores:
                raise RuntimeError("No more hard cores available")

            # Best-fit by placement waste (aspect-ratio-aware).
            chosen_unused = None
            chosen_waste = float("inf")
            for hc in unused_hardcores:
                if is_mapping_possible(core, hc):
                    waste = _placement_waste(core, hc)
                    if waste < chosen_waste:
                        chosen_unused = hc
                        chosen_waste = waste

            if chosen_unused is None:
                raise RuntimeError(
                    "No more hard cores available (no unused core type can fit this softcore)"
                )

            used_hardcores.append(chosen_unused)
            unused_hardcores.remove(chosen_unused)
            target_idx = len(used_hardcores) - 1

        place(target_idx, used_hardcores[target_idx], core)
        softcores.remove(core)
