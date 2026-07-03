"""Single greedy placement engine shared by the layout and runtime packers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores


@runtime_checkable
class Materializer(Protocol):
    """Strategy bundling the per-path feasibility, place, fuse, and split callbacks."""

    def is_mapping_possible(self, softcore: Any, hardcore: Any, /) -> bool: ...

    def place(self, core_idx: int, hardcore: Any, softcore: Any, /) -> None: ...

    def fuse_hardcores(self, hardcores: list, /) -> Any: ...

    def split_softcore(self, softcore: Any, available_neurons: int, /) -> tuple[Any, Any]: ...


def run_placement(
    *,
    softcores: list,
    used_hardcores: list,
    unused_hardcores: list,
    materializer: Materializer,
    allow_neuron_splitting: bool,
    allow_coalescing: bool = True,
) -> None:
    """Greedy-pack ``softcores`` onto hardcores using the materializer strategy.

    Mutates the hardcore lists in place; ``split_softcore`` is wired only when
    ``allow_neuron_splitting`` and ``fuse_hardcores`` only when ``allow_coalescing``.
    """
    greedy_pack_softcores(
        softcores=softcores,
        used_hardcores=used_hardcores,
        unused_hardcores=unused_hardcores,
        is_mapping_possible=materializer.is_mapping_possible,
        place=materializer.place,
        fuse_hardcores=(materializer.fuse_hardcores if allow_coalescing else None),
        split_softcore=(
            materializer.split_softcore if allow_neuron_splitting else None
        ),
    )
