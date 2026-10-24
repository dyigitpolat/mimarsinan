from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.packing.core_packing import (
    canonical_fuse_hardcores,
    canonical_is_mapping_possible,
    canonical_split_softcore,
)
from mimarsinan.mapping.packing.canonical import fused_core_roles
from mimarsinan.mapping.packing.greedy.pool import LazyCorePool
from mimarsinan.mapping.packing.infeasibility_proofs import (
    failed_pack_verdict,
    named_message,
    tag_with_verdict,
)
from mimarsinan.mapping.packing.placement_engine import run_placement
from mimarsinan.mapping.layout.layout_types import (
    core_pool_types,
    LayoutCoreSnapshot,
    LayoutHardCoreInstance,
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)


_SPLIT_SUFFIX_RE = re.compile(r"(_split_\d+)+$")


def _split_root(name: str) -> str:
    return _SPLIT_SUFFIX_RE.sub("", name) or name


class LayoutMaterializer:
    """Shape-only ``Materializer`` mirroring runtime placement decisions and
    tracking split lineage for split-fragment statistics.

    With ``collect_placements`` it additionally records which hardcore every
    softcore landed on, keyed by the ORIGIN index seeded by ``pack_layout``
    (split/coalescing fragments inherit their parent's origin) — the NoC
    estimator's placement input. Lookup is strict: a spec the origin map has
    never seen means an engine path copied specs, which must fail loud rather
    than misplace traffic. Fused components stay referenced (``fused_components``),
    so no later core can take a component's id and resolve to its fused core.
    """

    #: The predicate is ``canonical_is_mapping_possible`` (the engine indexes core states).
    canonical_feasibility = True

    def __init__(self, collect_placements: bool = False) -> None:
        self.split_counter = 0
        self.split_lineage: Dict[str, int] = {}
        self.collect_placements = bool(collect_placements)
        self.origin_of: Dict[int, int] = {}
        self.placed: List[Tuple[int, LayoutHardCoreInstance]] = []
        self.fused_alias: Dict[int, LayoutHardCoreInstance] = {}
        self.fused_components: List[LayoutHardCoreInstance] = []
        self.fusion_sizes: List[int] = []  # [W04] cores per fused core, in fusion order

    @staticmethod
    def is_mapping_possible(softcore, hardcore) -> bool:
        return canonical_is_mapping_possible(softcore, hardcore)

    def place(self, core_idx: int, hardcore: LayoutHardCoreInstance, core: LayoutSoftCoreSpec) -> None:
        hardcore.add_softcore(core)
        if self.collect_placements:
            self.placed.append((self.origin_of[id(core)], hardcore))

    def fuse_hardcores(self, hcs):
        def _mk(*, axons, neurons, template, components):
            inst = LayoutHardCoreInstance(
                axons_per_core=int(axons),
                neurons_per_core=int(neurons),
            )
            inst.residency_class_id = getattr(template, "residency_class_id", None)
            inst.latency_tag = getattr(template, "latency_tag", None)
            inst.accepted_roles = fused_core_roles(getattr(template, "accepted_roles", None))
            inst.component_axons = tuple(int(c.axons_per_core) for c in components)
            return inst

        fused = canonical_fuse_hardcores(hcs, make_fused=_mk)
        self.fusion_sizes.append(len(hcs))
        if self.collect_placements:
            self.fused_components.extend(hcs)
            for component in hcs:
                self.fused_alias[id(component)] = fused
        return fused

    def split_softcore(self, core: LayoutSoftCoreSpec, available_neurons: int):
        self.split_counter += 1
        root = _split_root(core.name or f"__noname_{id(core)}")
        self.split_lineage[root] = self.split_lineage.get(root, 0) + 1
        fragments = canonical_split_softcore(
            core, available_neurons, make_fragments=_make_layout_fragments,
        )
        if self.collect_placements:
            for fragment in fragments:
                self.origin_of[id(fragment)] = self.origin_of[id(core)]
        return fragments


def _core_pool(core_types: Sequence[LayoutHardCoreType]) -> LazyCorePool:
    """Every declared core, built only when the engine reads or takes it (with its type's roles, N2)."""
    return LazyCorePool(*core_pool_types(core_types))


def _make_layout_fragments(
    *,
    softcore: LayoutSoftCoreSpec,
    first_neurons: int,
    remaining_neurons: int,
) -> tuple[LayoutSoftCoreSpec, LayoutSoftCoreSpec]:
    """Construct two layout softcore fragments at a neuron boundary; the split
    decision matches runtime ``_make_real_fragments`` (same canonical callback), and
    each fragment keeps its parent's bank, class and segment as the runtime one does."""
    frag1 = replace(softcore, output_count=first_neurons,
                    name=f"{softcore.name}_split_0" if softcore.name else None)
    frag2 = replace(softcore, output_count=remaining_neurons,
                    name=f"{softcore.name}_split_1" if softcore.name else None)
    return frag1, frag2


def pack_layout(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    collect_placements: bool = False,
) -> LayoutPackingResult:
    """Pack layout-only softcores into a limited pool of hardware cores.

    ``allow_neuron_splitting`` splits along the neuron dimension; ``allow_coalescing``
    fuses ceil(a / A) cores for a softcore wider than every core, as the hard-core
    packer does (``run_placement``): [W04] the wide softcore stays whole, never pre-split
    into independent single-core fragments, so this plane packs what the builder packs.
    ``collect_placements`` additionally seals which hardcore each input softcore
    landed on (``result.placements``: ``(origin index, hardcore index)`` per
    placed unit; fragments repeat their origin)."""

    used_hardcores: List[LayoutHardCoreInstance] = []
    unused_hardcores = _core_pool(core_types)

    unmapped = [sc if sc.name is not None else replace(sc, name=f"__sc_{i}")
                for i, sc in enumerate(copy.deepcopy(list(softcores)))]

    materializer = LayoutMaterializer(collect_placements=collect_placements)
    if collect_placements:
        materializer.origin_of = {id(sc): i for i, sc in enumerate(unmapped)}

    try:
        run_placement(
            softcores=unmapped,
            used_hardcores=used_hardcores,
            unused_hardcores=unused_hardcores,
            materializer=materializer,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
    except RuntimeError as e:
        # Fusion spreads a softcore's axons only under declared coalescing (as on the runtime packer).
        verdict = failed_pack_verdict(
            softcores, core_types,
            neurons_may_split=allow_neuron_splitting, axons_may_spread=allow_coalescing,
        )
        return LayoutPackingResult(
            feasible=False,
            cores_used=0,
            total_capacity=0,
            used_area=0,
            unused_area_total=0,
            avg_unused_area_per_core=float("inf"),
            unusable_space_total=0,
            avg_unusable_space_per_core=0.0,
            error=tag_with_verdict(verdict, named_message(e)),
            verdict=verdict,
        )

    cores_used = len(used_hardcores)  # allocated crossbars; a fused core's members: its snapshot's ``tiles``
    total_capacity = sum(int(h.capacity) for h in used_hardcores)
    used_area = sum(int(h.used_area) for h in used_hardcores)
    unused_total = int(total_capacity - used_area)
    avg_unused = float(unused_total / cores_used) if cores_used > 0 else float("inf")

    unusable_total = sum(int(h.unusable_space) for h in used_hardcores)
    avg_unusable_space = (
        float(unusable_total / cores_used) if cores_used > 0 else 0.0
    )

    snapshots = tuple(
        LayoutCoreSnapshot(
            axons_per_core=h.axons_per_core,
            neurons_per_core=h.neurons_per_core,
            used_axons=h.axons_per_core - h.available_axons,
            used_neurons=h.neurons_per_core - h.available_neurons,
            used_area=h.used_area,
            softcore_count=h.softcore_count,
            component_axons=h.component_axons,
        )
        for h in used_hardcores
    )

    placements: Optional[Tuple[Tuple[int, int], ...]] = None
    if collect_placements:
        hardcore_index = {id(h): i for i, h in enumerate(used_hardcores)}
        resolved: List[Tuple[int, int]] = []
        for origin, hardcore in materializer.placed:
            while id(hardcore) in materializer.fused_alias:
                hardcore = materializer.fused_alias[id(hardcore)]
            resolved.append((int(origin), hardcore_index[id(hardcore)]))
        placements = tuple(resolved)

    return LayoutPackingResult(
        feasible=True,
        cores_used=cores_used,
        total_capacity=int(total_capacity),
        used_area=int(used_area),
        unused_area_total=unused_total,
        avg_unused_area_per_core=avg_unused,
        unusable_space_total=int(unusable_total),
        avg_unusable_space_per_core=avg_unusable_space,
        error=None,
        used_core_softcore_counts=tuple(hc.softcore_count for hc in used_hardcores),
        used_core_snapshots=snapshots,
        coalesced_fragment_count=sum(k - 1 for k in materializer.fusion_sizes),
        split_fragment_count=materializer.split_counter,
        coalescing_group_sizes=tuple(materializer.fusion_sizes) or None,
        split_counts_per_sc=(
            tuple(materializer.split_lineage.values())
            if materializer.split_lineage
            else None
        ),
        placements=placements,
        core_regions=tuple(frozenset(h.regions) for h in used_hardcores) if collect_placements else None,
    )
