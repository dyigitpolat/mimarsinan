"""[E85] The executed residency plan: the composition over the builder's IR cores, its pinned marks and its record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir.dynamic_bank import dynamic_bank_ids, operand_cells_by_core, payload_sources
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.layout.segmentation import HostSegment, NeuralSegment, Segment, compute_segment_ids
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES, ResidencySegmentPlan
from mimarsinan.mapping.support.residency_rules import ProgramGraph, check_pinned_fit, declared_pool, fit_pins
from mimarsinan.mapping.support.schedule.pass_cut import cut_nodes_from_cores, with_widths
from mimarsinan.mapping.support.schedule.residency_composition import (
    ResidencyComposition,
    compose_residency_program,
    pin_program,
)
from mimarsinan.mapping.packing.residency_tiles import CoreTiles, PackedCore, core_type_tiles

#: G5's own stage attribute (S0's rule for its two: set only on an executed plan, never as a dataclass field, so a
#: default build pickles byte-identically): the executed pass beside the composition's plan of it.
SCHEDULE_RESIDENCY_PASS = "schedule_residency_pass"


@dataclass(frozen=True)
class ResidencyPassCensus:
    """One executed pass beside the composition's plan of it (the per-pass difference of §9.1). The composition
    packs the phase with the layout packer, the builder the pass's IR cores with the hard-core packer, and each pins
    its own cores by the one rule; the executed fields are the record's (its programmed bytes are the marks')."""

    pass_index: int
    #: R_q: the phase's planned tiles (each unit's layer packed alone).
    planned_tiles: int
    planned_cores: Tuple[PackedCore, ...]
    planned_resident: FrozenSet[int]
    executed_cores: Tuple[PackedCore, ...]
    executed_resident: FrozenSet[int]

    @property
    def executed_cells(self) -> Tuple[int, ...]:
        return tuple(c for c, _ in self.executed_cores)

    @property
    def executed_tiles(self) -> int:
        return sum(t for _, t in self.executed_cores)

    def plane_differences(self) -> Tuple[str, ...]:
        """How the executed pass differs from the composition's packing of it, with the cause; () where equal."""
        out = []
        if self.planned_cores != self.executed_cores:
            out.append(f"packed cores differ (packer divergence): the layout packer packs the phase into cells "
                       f"({_runs(c for c, _ in self.planned_cores)}), the builder's hard-core packer the pass into "
                       f"({_runs(self.executed_cells)})")
        if self.planned_resident != self.executed_resident:
            out.append(f"pinned cores differ (the pinned rule on each packing): the composition pins "
                       f"{sorted(self.planned_resident)}, the builder {sorted(self.executed_resident)}")
        return tuple(out)

    def differences(self) -> Tuple[str, ...]:
        """``plane_differences`` and the pass's packed tiles against its planned count, with the cause."""
        if self.executed_tiles == self.planned_tiles:
            return self.plane_differences()
        cause = ("joint packing of the phase saves tiles" if self.executed_tiles < self.planned_tiles else
                 "packing the phase on the declared counts costs tiles its layers packed alone do not")
        return self.plane_differences() + (
            f"the pass packs into {self.executed_tiles} tiles against the phase's {self.planned_tiles} "
            f"planned tiles ({cause})",)


def _runs(values) -> str:
    """Values in order, a run of equal ones written once with its length (``8192 x3``)."""
    runs: List[List[int]] = []
    for v in values:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([int(v), 1])
    return ", ".join(f"{v} x{n}" if n > 1 else str(v) for v, n in runs)


@dataclass(frozen=True)
class BuilderComposition:
    """The composition on the builder's plane: per neural segment its passes of IR cores, and [F3] the program's
    dynamic operand banks (the built cores holding them are never pinned)."""

    composition: ResidencyComposition
    chunks: Tuple[Tuple[Tuple[Any, ...], ...], ...]
    operand_banks: FrozenSet[int] = frozenset()


def _producers(ir_graph: Any) -> Dict[int, Set[int]]:
    """Per node the nodes it reads: its input sources and [F3] the host op writing its payload (``payload_sources``)."""
    payload = payload_sources(ir_graph)
    return {
        int(node.id): {int(s.node_id) for s in np.asarray(node.input_sources, dtype=object).ravel()
                       if isinstance(s, IRSource) and s.node_id >= 0} | ({payload[int(node.id)]}
                                                                          if int(node.id) in payload else set())
        for node in ir_graph.nodes
    }


def depth_partition(ir_graph: Any) -> List[Segment]:
    """The builder's segments under the key: the IR segmented as the shape-only plane segments it
    (``compute_segment_ids``: a neural core's segment rises across a host dependency): per depth, its neural cores as
    one segment, then that depth's host ops, IR order.

    A topological order (a core at depth d reads hosts of depth < d; a host of depth d reads cores of depth <= d),
    so both planes plan one program; the IR-run partition (``partition_ir_graph``) groups branches by node order
    instead, which on a branchy graph is a different set of segments. The composition admits them only where every
    neural-to-neural edge keeps its class (``residency_rules.require_segmentation_parity``, on both planes)."""
    depth = compute_segment_ids(_producers(ir_graph), {int(n.id): isinstance(n, NeuralCore) for n in ir_graph.nodes})
    out: List[Segment] = []
    for d in sorted(set(depth.values())):
        cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and depth[int(n.id)] == d]
        if cores:
            out.append(NeuralSegment(nodes=cores, label=f"neural_segment_depth{d}"))
        out.extend(HostSegment(compute_op=n) for n in ir_graph.nodes
                   if not isinstance(n, NeuralCore) and depth[int(n.id)] == d)
    return out


def compose_builder_program(
    neural_segments: Sequence[Any], *, ir_graph: Any, cores_config: Sequence[dict],
    residency_plan: Mapping[str, Any],
    segment_specs: Callable[..., Tuple[list, Dict[int, Any], list]], allow_coalescing: bool,
    allow_splitting: bool, max_schedule_passes: int,
) -> BuilderComposition:
    """Plan the whole program once over the IR cores of every neural segment (forced cuts at host ops), under the
    refusal rules both planes share, read over the IR's program graph."""
    specs: List[Any] = []
    cores: List[Any] = []
    segments: List[int] = []
    group_of: Dict[Tuple[int, int], int] = {}
    bonds: List[Tuple[int, int]] = []
    for ordinal, segment in enumerate(neural_segments):
        seg_specs, spec_to_core, groups = segment_specs(segment.nodes, ir_graph=ir_graph)
        for spec, group in zip(seg_specs, groups):
            if group is not None:  # a coalesced wide fan-in's fragments run in one pass
                bonds.append((group_of.setdefault((ordinal, int(group)), len(specs)), len(specs)))
            specs.append(spec)
            cores.append(spec_to_core[id(spec)])
            segments.append(ordinal)
    index_of = {int(core.id): i for i, core in enumerate(cores)}
    nodes = []
    for ordinal, segment in enumerate(neural_segments):
        for node in cut_nodes_from_cores(segment.nodes, [int(c.id) for c in segment.nodes]):
            nodes.append(type(node)(core_id=index_of[node.core_id], latency=node.latency,
                                    sources=tuple(index_of[s] for s in node.sources), out_width=node.out_width))
    nodes = with_widths(sorted(nodes, key=lambda n: n.core_id), specs)
    composition = compose_residency_program(
        specs, nodes, core_types_from_declaration(cores_config), segments=segments, residency_plan=residency_plan,
        program=ProgramGraph.of_ir(ir_graph, index_of), allow_coalescing=allow_coalescing,
        allow_splitting=allow_splitting, max_schedule_passes=max_schedule_passes, bonds=bonds)
    by_segment = composition.segment_passes()
    chunks = tuple(tuple(tuple(cores[i] for i in members) for members in by_segment.get(ordinal, []))
                   for ordinal in range(len(neural_segments)))
    return BuilderComposition(composition=composition, chunks=chunks, operand_banks=dynamic_bank_ids(ir_graph))


def mark_residency_plan(pass_stages: Sequence[Any], builder: BuilderComposition,
                        cores_config: Sequence[dict]) -> Tuple[ResidencySegmentPlan, ...]:
    """Pin each executed pass's cores by the one pinned rule (never a core holding an operand bank [F3]), check the
    fit on the declared pool, record each pass beside the composition's plan of it, and return the plan per
    segment."""
    comp = builder.composition
    if len(pass_stages) != len(comp.facts.phases):
        raise ResidencyPlanError(f"residency_plan: {len(pass_stages)} built passes for {len(comp.facts.phases)} "
                                 f"planned ones")
    hcms = [stage.hard_core_mapping for stage in pass_stages]
    declared = core_types_from_declaration(cores_config)
    payloads = [[CoreOccupancy.from_hard_core(core).cells_used for core in hcm.cores] for hcm in hcms]
    types = [[core_type_tiles(core.axons_per_core, core.neurons_per_core, declared,
                              getattr(core, "fused_component_axons", None)) for core in hcm.cores] for hcm in hcms]
    producer = [operand_cells_by_core(hcm, builder.operand_banks) for hcm in hcms] if builder.operand_banks else ()
    pins = pin_program(payloads, types, comp.r, comp.budget, producer)
    check_pinned_fit(types, pins, declared_pool(declared))
    for q, (stage, pin) in enumerate(zip(pass_stages, pins)):
        setattr(stage, SCHEDULE_RESIDENT_CORES, frozenset(pin))
        setattr(stage, SCHEDULE_RESIDENCY_PASS, ResidencyPassCensus(
            pass_index=q, planned_tiles=int(comp.facts.pass_tiles[q]), planned_cores=comp.facts.pass_cores[q],
            planned_resident=comp.facts.pass_resident_cores[q], executed_resident=frozenset(pin),
            executed_cores=tuple((int(c), len(t)) for c, t in zip(payloads[q], types[q]))))
    plans = []
    segment_of_pass = comp.pass_segments()
    for ordinal in sorted(set(segment_of_pass)):
        mine = [q for q, s in enumerate(segment_of_pass) if s == ordinal]
        cuts = [comp.facts.phases[q][1] for q in mine[:-1]]
        plans.append(ResidencySegmentPlan(
            segment_index=int(pass_stages[mine[0]].schedule_segment_index), step_rule=comp.selection.rule,
            step_index=comp.selection.step_index, rotating_tiles=comp.r, passes=len(mine),
            carried_elements=2 * sum(int(comp.units.live[c - 1]) for c in cuts),
            written_tiles=comp.facts.written_tiles, total_tiles=comp.plan.F, core_budget=comp.budget))
    return tuple(plans)


def _stated_tiles(core: Any, declared: Sequence[Any]) -> Optional[CoreTiles]:
    """A built core's declared tiles, or None where a fused core decomposes into the declared types ambiguously."""
    try:
        return core_type_tiles(core.axons_per_core, core.neurons_per_core, declared,
                               getattr(core, "fused_component_axons", None))
    except ResidencyPlanError:
        return None


def mark_fit_residency(pass_stages: Sequence[Any], cores_config: Sequence[dict]) -> bool:
    """[W03] The planner pins a program that fits: when the executed passes that load their own cores fit the
    declared pool side by side (``fit_pins``, the floor at r = 0), every core of every pass is marked resident, so the
    program is written once; otherwise (or where a core's tiles are undetermined) U1's executed rotation stands.
    Returns whether it pinned."""
    declared = core_types_from_declaration(cores_config)
    stated = [[_stated_tiles(core, declared) for core in stage.hard_core_mapping.cores] for stage in pass_stages]
    if not pass_stages or any(t is None for cores in stated for t in cores):
        return False
    types: List[List[CoreTiles]] = [[t for t in cores if t is not None] for cores in stated]
    pins = fit_pins(types, [bool(getattr(stage, "schedule_weights_resident", False)) for stage in pass_stages],
                    declared_pool(declared))
    if pins is None:
        return False
    for stage, pin in zip(pass_stages, pins):
        setattr(stage, SCHEDULE_RESIDENT_CORES, pin)
    return True
