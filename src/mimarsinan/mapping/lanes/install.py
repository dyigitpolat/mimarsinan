"""The Soft Core Mapping law hook of a lane program (WI-5e): fuse the planned lanes, solve, stamp IR and model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.mapping.export.law_rules import ReadoutLawSpec, refuse_stale_fixed_range
from mimarsinan.mapping.export.law_stamp import readout_quantizers_by_perceptron
from mimarsinan.mapping.export.output_law import identity_output_indices
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.declaration import LaneDeclaration
from mimarsinan.mapping.lanes.ir.fusion import LANE_REPORT, fuse_ir_lanes
from mimarsinan.mapping.lanes.ir.reduction_banks import declared_reduction_banks, refuse_split_bank_laws
from mimarsinan.mapping.lanes.law.lane_grids import LaneGrid
from mimarsinan.mapping.lanes.plan import plan_lanes
from mimarsinan.mapping.lanes.law.solve import solve_lane_laws
from mimarsinan.mapping.lanes.law.stamp import lane_channel_keys, stamp_lane_laws
from mimarsinan.mapping.support.grid_declarations import ParameterGrids, identity_output_law

def lanes_report(ir_graph) -> Optional[Dict[str, Any]]:
    """The fusion report of an IR whose lanes were fused (``None`` otherwise)."""
    return getattr(ir_graph, LANE_REPORT, None)


def lane_core_geometry(platform: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """The declared core geometry a reduction core stays within (the widest axons and neurons of the chip's types)."""
    cores = platform.get("cores") or ()
    axons = [int(c["max_axons"]) for c in cores if c.get("max_axons") is not None]
    neurons = [int(c["max_neurons"]) for c in cores if c.get("max_neurons") is not None]
    return (max(axons) if axons else None), (max(neurons) if neurons else None)


def _grid(quantizer) -> LaneGrid:
    return LaneGrid(float(quantizer.range), float(quantizer.lane_offset), bool(quantizer.lane_unsigned))


def install_lane_laws(ir_graph, model, platform: Mapping[str, Any], spec: ReadoutLawSpec,
                      declaration: LaneDeclaration, *, weight_bits: int) -> int:
    """Fuse ``declaration``'s lanes into ``ir_graph``, solve every core's lane law and stamp IR and model."""
    readout_law = platform["readout_law"]
    repr_ = model.get_mapper_repr()
    plan = plan_lanes(repr_, declaration)
    quantizers = readout_quantizers_by_perceptron(repr_)
    first_owner = len(list(repr_.get_perceptrons()))
    max_axons, max_neurons = lane_core_geometry(platform)
    fusion = fuse_ir_lanes(ir_graph, plan, first_owner=first_owner, max_axons=max_axons, max_neurons=max_neurons,
                           reduction_banks=declared_reduction_banks(platform))
    owners: Dict[int, LaneGrid] = {index: _grid(entry[2]) for index, entry in quantizers.items()}
    for k, lane in enumerate(plan.pool_lanes):
        owners[first_owner + k] = _grid(lane.pool.module.quantizer)
    identity = frozenset(identity_output_indices(ir_graph, repr_, quantizers)
                         if identity_output_law(readout_law) else ())
    refuse_stale_fixed_range(quantizers, readout_law.get("range_value"), skip=identity)
    laws = solve_lane_laws(ir_graph, owners, spec, real_bias=ParameterGrids.from_config(platform).law_bias,
                           identity_outputs=identity,
                           channel_keys=lane_channel_keys(ir_graph, plan, fusion, first_owner))
    refuse_split_bank_laws(ir_graph, laws, fusion.bank_ids)  # [U-L2] one law per reduction bank's instances
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.readout_law = laws[int(node.id)]  # pyright: ignore[reportAttributeAccessIssue] -- declared-only
    stamp_lane_laws(model, ir_graph, laws, plan, fusion, first_owner=first_owner, weight_bits=weight_bits)
    setattr(ir_graph, LANE_REPORT, {**fusion.as_dict(), "classes": sorted(declaration.classes),
                                    "lanes_per_core": declaration.lanes_per_core, "lane_block": declaration.lane_block,
                                    "plan": plan.counts()})
    return len(laws)
