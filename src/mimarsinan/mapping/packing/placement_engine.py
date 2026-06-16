"""Single greedy placement engine shared by the layout and runtime packers.

``greedy_pack_softcores`` is the shared bin-packing kernel; the only thing that
differs between the shape-only layout packer (``pack_layout``) and the
weight-bearing runtime packer (``HardCoreMapping.map``) is *how* a softcore is
placed, fused, and split.  A :class:`Materializer` bundles those three
strategy hooks (plus the shared feasibility predicate) so both paths drive the
exact same assignment kernel and can never diverge on placement decisions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores


@runtime_checkable
class Materializer(Protocol):
    """Strategy bundling the per-path placement callbacks.

    ``is_mapping_possible(hardcore, softcore)`` -- feasibility predicate
    (canonical for both paths).
    ``place(core_idx, hardcore, softcore)`` -- commit a softcore onto a hardcore.
    ``fuse_hardcores(hardcores)`` -- build one fused hardcore from a list.
    ``split_softcore(softcore, available_neurons)`` -- split along neurons.
    """

    def is_mapping_possible(self, hardcore: Any, softcore: Any) -> bool: ...

    def place(self, core_idx: int, hardcore: Any, softcore: Any) -> None: ...

    def fuse_hardcores(self, hardcores: list) -> Any: ...

    def split_softcore(self, softcore: Any, available_neurons: int): ...


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

    Mutates ``used_hardcores`` / ``unused_hardcores`` in place exactly as the
    callers' inline ``greedy_pack_softcores`` invocations did. ``split_softcore``
    is wired only when ``allow_neuron_splitting`` is set, and ``fuse_hardcores``
    (combine N cores into one wider crossbar — the coalescing capability) only
    when ``allow_coalescing`` is set; otherwise a softcore too wide for any single
    hard core has no placement and the kernel raises.
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
