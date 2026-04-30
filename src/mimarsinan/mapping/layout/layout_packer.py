from __future__ import annotations

import copy
import math
import re
from typing import Dict, List, Sequence, Tuple

from mimarsinan.mapping.core_packing import (
    canonical_split_softcore,
    greedy_pack_softcores,
)
from mimarsinan.mapping.layout.layout_types import (
    LayoutCoreSnapshot,
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


def _make_layout_fragments(
    *,
    softcore: LayoutSoftCoreSpec,
    first_neurons: int,
    remaining_neurons: int,
) -> tuple[LayoutSoftCoreSpec, LayoutSoftCoreSpec]:
    """Construct two layout softcore fragments at a neuron boundary.

    Mirrors the runtime-side ``_make_real_fragments`` in
    ``softcore_mapping.py`` — both are ``make_fragments`` callbacks for
    ``canonical_split_softcore``, so the split *decision* (boundary,
    two-fragment protocol) is identical; only the fragment *construction*
    differs (layout only needs shape + carried-over group/latency tags).
    """
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
    """Expand softcores for axon coalescing.

    A softcore whose input_count exceeds the largest hw core's max_axons is split
    into K coalescing fragments, each covering a slice of the inputs but all of
    the output neurons — exactly what IRMapping does at model-building time when
    coalescing is enabled.  This lets the layout packer correctly verify and count
    hardware cores for coalescing configurations.

    Returns (expanded_softcores, coalescing_group_sizes) where coalescing_group_sizes
    contains one entry per softcore that produced more than one coalescing fragment.
    """
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
            k = math.ceil(sc.input_count / max_avail_axons)
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
    """
    Pack layout-only softcores into a limited pool of hardware cores.

    ``allow_neuron_splitting``:
        Softcores may be split along the neuron (output) dimension across
        multiple hardware cores.

    ``allow_coalescing``:
        Softcores whose input count exceeds a single hardware core's max_axons
        are pre-expanded into axon-coalescing fragments before packing.  Each
        fragment carries all output neurons of the original softcore, mirroring
        IRMapping's coalescing behaviour.
    """

    used_hardcores: List[LayoutHardCoreInstance] = []
    unused_hardcores: List[LayoutHardCoreInstance] = _make_instances(core_types)

    # Work on a mutable copy; greedy_pack_softcores removes elements.
    unmapped = list(copy.deepcopy(list(softcores)))
    pre_coalesce_count = len(unmapped)

    # Pre-expand for axon coalescing so the greedy packer sees fragments that
    # fit within individual hardware cores' axon capacities.
    coalescing_group_sizes: Tuple[int, ...] = ()
    if allow_coalescing:
        unmapped, coalescing_group_sizes = _expand_for_axon_coalescing(unmapped, core_types)
    coalesced_fragment_count = len(unmapped) - pre_coalesce_count

    # Pre-assign temp names to anonymous softcores so split lineage can be tracked
    # across multiple split calls via the _split_N suffix naming convention.
    for i, sc in enumerate(unmapped):
        if sc.name is None:
            unmapped[i] = LayoutSoftCoreSpec(
                input_count=sc.input_count,
                output_count=sc.output_count,
                threshold_group_id=sc.threshold_group_id,
                latency_tag=sc.latency_tag,
                name=f"__sc_{i}",
            )

    _SPLIT_SUFFIX_RE = re.compile(r"(_split_\d+)+$")

    split_counter = [0]
    split_lineage: Dict[str, int] = {}  # root_name -> number of times any fragment was split

    def _split_root(name: str) -> str:
        return _SPLIT_SUFFIX_RE.sub("", name) or name

    def _counting_split(core: LayoutSoftCoreSpec, available_neurons: int):
        split_counter[0] += 1
        root = _split_root(core.name or f"__noname_{id(core)}")
        split_lineage[root] = split_lineage.get(root, 0) + 1
        return canonical_split_softcore(
            core, available_neurons, make_fragments=_make_layout_fragments,
        )

    from mimarsinan.mapping.core_packing import (
        canonical_fuse_hardcores,
        canonical_is_mapping_possible,
    )
    is_mapping_possible = canonical_is_mapping_possible

    def place(core_idx: int, hardcore: LayoutHardCoreInstance, core: LayoutSoftCoreSpec) -> None:
        hardcore.add_softcore(core)

    # Shape-only fuse for layout hardcores.  Gives the layout packer the same
    # "try to fuse N unused cores for a wide softcore" branch the runtime
    # packer takes, so ``greedy_pack_softcores`` explores identical decisions
    # in both paths.
    def _layout_fuse(hcs):
        def _mk(*, axons, neurons, template, components):
            inst = LayoutHardCoreInstance(
                axons_per_core=int(axons),
                neurons_per_core=int(neurons),
            )
            inst.threshold_group_id = getattr(template, "threshold_group_id", None)
            inst.latency_tag = getattr(template, "latency_tag", None)
            return inst
        return canonical_fuse_hardcores(hcs, make_fused=_mk)

    try:
        greedy_pack_softcores(
            softcores=unmapped,
            used_hardcores=used_hardcores,
            unused_hardcores=unused_hardcores,
            is_mapping_possible=is_mapping_possible,
            place=place,
            fuse_hardcores=_layout_fuse,
            split_softcore=_counting_split if allow_neuron_splitting else None,
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
        split_fragment_count=split_counter[0],
        coalescing_group_sizes=coalescing_group_sizes if coalescing_group_sizes else None,
        split_counts_per_sc=tuple(split_lineage.values()) if split_lineage else None,
    )
