from __future__ import annotations

import copy
import re
from typing import Dict, List, Sequence, Tuple

from mimarsinan.mapping.packing.core_packing import (
    canonical_fuse_hardcores,
    canonical_is_mapping_possible,
    canonical_split_softcore,
)
from mimarsinan.mapping.packing.placement_engine import run_placement
from mimarsinan.mapping.platform.coalescing import coalescing_fragment_count
from mimarsinan.mapping.layout.layout_types import (
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
    tracking split lineage for split-fragment statistics."""

    def __init__(self) -> None:
        self.split_counter = 0
        self.split_lineage: Dict[str, int] = {}

    @staticmethod
    def is_mapping_possible(hardcore, softcore) -> bool:
        return canonical_is_mapping_possible(hardcore, softcore)

    def place(self, core_idx: int, hardcore: LayoutHardCoreInstance, core: LayoutSoftCoreSpec) -> None:
        hardcore.add_softcore(core)

    def fuse_hardcores(self, hcs):
        def _mk(*, axons, neurons, template, components):
            inst = LayoutHardCoreInstance(
                axons_per_core=int(axons),
                neurons_per_core=int(neurons),
            )
            inst.threshold_group_id = getattr(template, "threshold_group_id", None)
            inst.latency_tag = getattr(template, "latency_tag", None)
            return inst

        return canonical_fuse_hardcores(hcs, make_fused=_mk)

    def split_softcore(self, core: LayoutSoftCoreSpec, available_neurons: int):
        self.split_counter += 1
        root = _split_root(core.name or f"__noname_{id(core)}")
        self.split_lineage[root] = self.split_lineage.get(root, 0) + 1
        return canonical_split_softcore(
            core, available_neurons, make_fragments=_make_layout_fragments,
        )


def _make_instances(core_types: Sequence[LayoutHardCoreType]) -> List[LayoutHardCoreInstance]:
    out: List[LayoutHardCoreInstance] = []
    for t in core_types:
        for _ in range(int(t.count)):
            out.append(LayoutHardCoreInstance(axons_per_core=int(t.max_axons), neurons_per_core=int(t.max_neurons)))
    return out


def _make_layout_fragments(
    *,
    softcore: LayoutSoftCoreSpec,
    first_neurons: int,
    remaining_neurons: int,
) -> tuple[LayoutSoftCoreSpec, LayoutSoftCoreSpec]:
    """Construct two layout softcore fragments at a neuron boundary; the split
    decision matches runtime ``_make_real_fragments`` (same canonical callback)."""
    frag1 = LayoutSoftCoreSpec(
        input_count=softcore.input_count,
        output_count=first_neurons,
        threshold_group_id=softcore.threshold_group_id,
        latency_tag=softcore.latency_tag,
        name=f"{softcore.name}_split_0" if softcore.name else None,
    )
    frag2 = LayoutSoftCoreSpec(
        input_count=softcore.input_count,
        output_count=remaining_neurons,
        threshold_group_id=softcore.threshold_group_id,
        latency_tag=softcore.latency_tag,
        name=f"{softcore.name}_split_1" if softcore.name else None,
    )
    return frag1, frag2


def _expand_for_axon_coalescing(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
) -> Tuple[List[LayoutSoftCoreSpec], Tuple[int, ...]]:
    """Split each over-wide softcore into axon-coalescing fragments (each carrying
    all output neurons), mirroring IRMapping's coalescing at model-building time."""
    if not core_types:
        return list(softcores), ()
    max_avail_axons = max(int(ct.max_axons) for ct in core_types)
    if max_avail_axons <= 0:
        return list(softcores), ()

    result: List[LayoutSoftCoreSpec] = []
    group_sizes: List[int] = []
    for sc in softcores:
        if sc.input_count <= max_avail_axons:
            result.append(sc)
        else:
            k = coalescing_fragment_count(sc.input_count, max_avail_axons)
            base = sc.input_count // k
            rem = sc.input_count - base * k
            for i in range(k):
                frag_ax = base + (1 if i < rem else 0)
                result.append(LayoutSoftCoreSpec(
                    input_count=frag_ax,
                    output_count=sc.output_count,
                    threshold_group_id=sc.threshold_group_id,
                    latency_tag=sc.latency_tag,
                    name=f"{sc.name}_coal{i}" if sc.name else None,
                ))
            group_sizes.append(k)
    return result, tuple(group_sizes)


def pack_layout(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> LayoutPackingResult:
    """Pack layout-only softcores into a limited pool of hardware cores.

    ``allow_neuron_splitting`` splits along the neuron dimension; ``allow_coalescing``
    pre-expands over-wide softcores into axon-coalescing fragments before packing."""

    used_hardcores: List[LayoutHardCoreInstance] = []
    unused_hardcores: List[LayoutHardCoreInstance] = _make_instances(core_types)

    unmapped = list(copy.deepcopy(list(softcores)))
    pre_coalesce_count = len(unmapped)

    coalescing_group_sizes: Tuple[int, ...] = ()
    if allow_coalescing:
        unmapped, coalescing_group_sizes = _expand_for_axon_coalescing(unmapped, core_types)
    coalesced_fragment_count = len(unmapped) - pre_coalesce_count

    for i, sc in enumerate(unmapped):
        if sc.name is None:
            unmapped[i] = LayoutSoftCoreSpec(
                input_count=sc.input_count,
                output_count=sc.output_count,
                threshold_group_id=sc.threshold_group_id,
                latency_tag=sc.latency_tag,
                name=f"__sc_{i}",
            )

    materializer = LayoutMaterializer()

    try:
        run_placement(
            softcores=unmapped,
            used_hardcores=used_hardcores,
            unused_hardcores=unused_hardcores,
            materializer=materializer,
            allow_neuron_splitting=allow_neuron_splitting,
        )
    except RuntimeError as e:
        return LayoutPackingResult(
            feasible=False,
            cores_used=0,
            total_capacity=0,
            used_area=0,
            unused_area_total=0,
            avg_unused_area_per_core=float("inf"),
            unusable_space_total=0,
            avg_unusable_space_per_core=0.0,
            error=str(e),
        )

    cores_used = len(used_hardcores)
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
        )
        for h in used_hardcores
    )

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
        coalesced_fragment_count=coalesced_fragment_count,
        split_fragment_count=materializer.split_counter,
        coalescing_group_sizes=coalescing_group_sizes if coalescing_group_sizes else None,
        split_counts_per_sc=(
            tuple(materializer.split_lineage.values())
            if materializer.split_lineage
            else None
        ),
    )
