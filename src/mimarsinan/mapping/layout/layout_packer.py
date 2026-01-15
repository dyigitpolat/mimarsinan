from __future__ import annotations

import copy
from typing import List, Sequence

from mimarsinan.mapping.core_packing import greedy_pack_softcores
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreInstance,
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)


def _make_instances(core_types: Sequence[LayoutHardCoreType]) -> List[LayoutHardCoreInstance]:
    out: List[LayoutHardCoreInstance] = []
    for t in core_types:
        for _ in range(int(t.count)):
            out.append(LayoutHardCoreInstance(axons_per_core=int(t.max_axons), neurons_per_core=int(t.max_neurons)))
    return out


def pack_layout(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
) -> LayoutPackingResult:
    """
    Pack layout-only softcores into a limited pool of hardware cores.
    """

    used_hardcores: List[LayoutHardCoreInstance] = []
    unused_hardcores: List[LayoutHardCoreInstance] = _make_instances(core_types)

    # Work on a mutable copy; greedy_pack_softcores removes elements.
    unmapped = list(copy.deepcopy(list(softcores)))

    def is_mapping_possible(core: LayoutSoftCoreSpec, hardcore: LayoutHardCoreInstance) -> bool:
        # Threshold-group constraint: an empty hardcore can accept any group; otherwise, must match.
        if hardcore.threshold_group_id is not None and hardcore.threshold_group_id != int(core.threshold_group_id):
            return False

        # Latency constraint: keep parity with real mapper
        if hardcore.latency_tag is not None:
            if core.latency_tag is None or int(core.latency_tag) != int(hardcore.latency_tag):
                return False

        return (
            core.get_input_count() <= hardcore.available_axons
            and core.get_output_count() <= hardcore.available_neurons
        )

    def place(core_idx: int, hardcore: LayoutHardCoreInstance, core: LayoutSoftCoreSpec) -> None:
        hardcore.add_softcore(core)

    try:
        greedy_pack_softcores(
            softcores=unmapped,
            used_hardcores=used_hardcores,
            unused_hardcores=unused_hardcores,
            is_mapping_possible=is_mapping_possible,
            place=place,
        )
    except Exception as e:
        # Infeasible mapping: return a structured result; search backends can apply penalties.
        return LayoutPackingResult(
            feasible=False,
            cores_used=0,
            total_capacity=0,
            used_area=0,
            unused_area_total=0,
            avg_unused_area_per_core=float("inf"),
            error=str(e),
        )

    cores_used = len(used_hardcores)
    total_capacity = sum(int(h.capacity) for h in used_hardcores)
    used_area = sum(int(h.used_area) for h in used_hardcores)
    unused_total = int(total_capacity - used_area)
    avg_unused = float(unused_total / cores_used) if cores_used > 0 else float("inf")

    return LayoutPackingResult(
        feasible=True,
        cores_used=cores_used,
        total_capacity=int(total_capacity),
        used_area=int(used_area),
        unused_area_total=unused_total,
        avg_unused_area_per_core=avg_unused,
        error=None,
    )


