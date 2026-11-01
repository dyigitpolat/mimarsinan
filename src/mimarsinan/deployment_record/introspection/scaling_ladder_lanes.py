"""[U-E2 H-1] The shape-only census of a chip that runs its declared lanes, as the builder plans it.

The builder maps the lane-free program (its layout records, ``IRGraph.layout_softcores``), fuses the lanes into the
IR at Soft Core Mapping, and plans each segment's passes from ``packing/segment_specs.py``'s specs: a core's layout
record where it has one (its extent plus its operand rows), for a core the fusion inserted (a pool lane's or a
reduction's) the record the fusion wrote for it ([U-L] ``lanes/ir/layout_records.py``: the reconstruction's values,
its bank where its reduction cores are banked ([U-L2]) else none, segment 0, the fused IR's latency), and one source
per segment for the order facts the capacity split reads: the walk's records' where every core of the segment has
one, else the fused program's (segment 0, the fused IR's latency). Here the walk runs twice: lane-free (the records) and with the lane plan applied (``lanes.shape_walk``, the
fused program: its cells, runs, wire census and program graph), and the specs follow that one rule per builder
segment (the IR runs), and per depth segment of the fused program for a declared ``residency_plan`` (the builder's
``depth_partition``). The host census loses the absorbed units; the lane work is the fusion's count.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mimarsinan.deployment_record.introspection.scaling_ladder_types import ShapeCensus
from mimarsinan.mapping.lanes.shape_walk import LaneWalk
from mimarsinan.mapping.layout.ir_mapping.sign_rows import sign_row_census
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.platform.core_residency import BASIS_PROVENANCE, provenance_group_id, ungrouped_fallback_id
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.host_boundary import host_boundary_census


def layout_walk(network: Any, platform: Mapping[str, Any], *, wire_census: bool,
                lanes: Optional[LaneWalk] = None) -> LayoutIRMapping:
    """One shape-only walk of the network on the platform's core geometry (with ``lanes``: the lane plan applied).
    [U-AX] The walk holds the network's weights, so on a per-axon pool it reads each core's rows on the declared
    weight grid (``sign_row_census``), as the build packs them."""
    geometry = resolve_platform_mapping_params(platform["cores"],
                                               allow_coalescing=bool(platform.get("allow_coalescing", False)))
    layout = LayoutIRMapping(
        max_axons=geometry.effective_max_axons, max_neurons=geometry.effective_max_neurons,
        allow_coalescing=geometry.allow_coalescing, hardware_bias=geometry.hardware_bias,
        collect_wire_census=wire_census, rows_per_input=geometry.rows_per_input, lane_walk=lanes,
        sign_rows=sign_row_census(geometry.rows_per_input, platform.get("weight_bits")),
    )
    layout.collect_layout_softcores(network.mapper_repr)
    return layout


def lane_census(census: ShapeCensus, fused: LayoutIRMapping, lanes: LaneWalk,
                free: LayoutIRMapping) -> ShapeCensus:
    """``census`` (taken on the fused walk) with the builder's specs, the absorbed units and the lane work."""
    node_of = {int(sc): int(n) for n, sc in fused._node_id_to_softcore_idx.items()}
    specs = list(fused.layout_softcores)
    records: List[Optional[int]] = [lanes.record_of.get(node_of[i]) for i in range(len(specs))]
    runs = census.runs if census.runs is not None else tuple(0 for _ in specs)
    softcores = builder_specs(specs, list(free.layout_softcores), records, runs, depth=False)
    residency = builder_specs(specs, list(free.layout_softcores), records,
                              [int(s.segment_id or 0) for s in specs], depth=True)
    cells = sum((int(s.input_count) - int(s.operand_axons)) * int(s.output_count) for s in softcores)
    return replace(census, softcores=tuple(softcores), residency_softcores=tuple(residency),
                   as_mapped_cells=cells, lane_absorbed=lanes.absorbed_units, lane_ops=dict(lanes.lane_ops))


def builder_specs(specs: Sequence[Any], records: Sequence[Any], record_of: Sequence[Optional[int]],
                  groups: Sequence[int], *, depth: bool) -> List[Any]:
    """Per fused spec, the spec the builder's planner reads (``segment_specs._segment_specs``) in its segment of
    ``groups``: a kept core is its layout record's (``record_of``: its index there) with the fused extent; a core
    the fusion inserted is its own record's (its bank where [U-L2] its reduction cores are banked, else none; segment
    0, the fused latency, as the walk states it); a segment
    holding an inserted core reads every core's order facts off the fused program, any other the walk's records'. ``depth``: the segments are the
    fused program's depth segments (a declared plan's), which the specs keep."""
    members: Dict[int, List[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(int(group), []).append(i)
    out: List[Any] = list(specs)
    for group in members.values():
        mixed = any(record_of[i] is None for i in group)
        base = len(records) if any(record_of[i] is not None for i in group) else 0
        for position, i in enumerate(group):
            spec, record = specs[i], record_of[i]
            fallback = ungrouped_fallback_id(record if record is not None else base + position)
            row = replace(spec, residency_basis=BASIS_PROVENANCE,
                          residency_class_id=provenance_group_id(spec.perceptron_index, fallback=fallback))
            if record is None:  # [U-L2] an inserted core keeps the walk's bank: a banked reduction core's, else none
                row = replace(row, segment_id=0)
            elif mixed:
                row = replace(row, segment_id=0)
            else:
                row = replace(row, segment_id=records[record].segment_id, latency_tag=records[record].latency_tag)
            out[i] = replace(row, segment_id=spec.segment_id) if depth else row
    return out


def host_census(network: Any, census: Any) -> Dict[str, Any]:
    """The host census the price reads: the network's, less the host-op units the chip's lanes absorb."""
    host = dict(network.host_census or {})
    absorbed = int(getattr(census, "lane_absorbed", 0) or 0)
    if absorbed and host.get("compute_op_count") is not None:
        host["compute_op_count"] = int(host["compute_op_count"]) - absorbed
    return host


def walk_host_boundary(layout: LayoutIRMapping, network: Any) -> Any:
    """[U-J] The walk's host-chip boundary census (``host_boundary_census``), the network input sized from its
    declared resolution; None without the walk's wire census."""
    return host_boundary_census(layout, input_elements=math.prod(int(d) for d in network.resolution))
