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

    Uses a **best-fit** strategy when selecting an unused core: among all
    feasible unused cores, pick the *smallest* one (by capacity).  This
    improves utilisation of heterogeneous core pools.

    This mutates:
    - used_hardcores (may append newly allocated hardcores)
    - unused_hardcores (removes allocated hardcores)
    - softcores (removes packed softcores)
    """

    while softcores:
        core = pick_softcore(softcores)

        target_idx = None
        for idx, hc in enumerate(used_hardcores):
            if is_mapping_possible(core, hc):
                target_idx = idx
                break

        if target_idx is None:
            if not unused_hardcores:
                raise RuntimeError("No more hard cores available")

            # Best-fit: pick the smallest unused core that can accept this softcore.
            chosen_unused = None
            chosen_cap = float("inf")
            for hc in unused_hardcores:
                if is_mapping_possible(core, hc):
                    cap = _capacity(hc)
                    if cap < chosen_cap:
                        chosen_unused = hc
                        chosen_cap = cap

            if chosen_unused is None:
                raise RuntimeError(
                    "No more hard cores available (no unused core type can fit this softcore)"
                )

            used_hardcores.append(chosen_unused)
            unused_hardcores.remove(chosen_unused)
            target_idx = len(used_hardcores) - 1

        place(target_idx, used_hardcores[target_idx], core)
        softcores.remove(core)
