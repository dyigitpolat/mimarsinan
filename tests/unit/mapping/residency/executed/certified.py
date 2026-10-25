"""CL3 (d)-(e) helpers: an executed residency plan read on its three census planes and certified.

Planes of one executed program at its executed step:
- record: the built program's passes (``mapping_program_passes``, the record plane at build time, which states the
  per-boundary carry) and, where a pipeline run sealed one, the sealed record's (``record_program_passes``);
- plan: ``plan_period_census`` at the executed step on the build's own units (``facts_period_census`` of the
  builder's composition over the executed IR, each phase on its layout-packed cores, A1);
- candidate: the search plane's census (``collect_noc_fragments`` on the deferred layout walk of the model, then
  ``candidate_program_passes``).
The compute steps are each plane's own statement (the record's measured or unstated, the candidate's window, the
plan's none), so they are not compared; a boundary that carries 0 bytes is the same as no boundary.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    mapping_program_passes,
    record_program_passes,
)
from mimarsinan.deployment_record.quantities.write_law import plane_write_census
from mimarsinan.mapping.export.law_solver import stamp_model_laws
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.layout.segmentation import NeuralSegment
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs, build_identity_hybrid_mapping
from mimarsinan.mapping.packing.schedule_residency_plan import compose_builder_program, depth_partition
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENCY_PLAN, SCHEDULE_RESIDENT_CORES
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.residency_units import facts_period_census

PASS_FIELDS = ("segment_index", "resident", "cores", "resident_cores", "written_cells_backing_store",
               "written_cells_producer", "payload_bytes")


@dataclass(frozen=True)
class Walk:
    """One layout walk of a program: its specs, wire census and program graph."""

    specs: Tuple[Any, ...]
    census: Any
    graph: ProgramGraph


def deferred_walk(mapper_repr: Any, cores: Sequence[Mapping[str, Any]], *, allow_coalescing: bool) -> Walk:
    """The deferred layout walk the search plane runs (``layout_hook._collect_softcores``) at the platform's effective
    geometry."""
    pmap = resolve_platform_mapping_params(list(cores), allow_coalescing=allow_coalescing)
    walk = LayoutIRMapping(max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons,
                           allow_coalescing=pmap.allow_coalescing, hardware_bias=pmap.hardware_bias,
                           collect_wire_census=True)
    with contextlib.redirect_stdout(io.StringIO()):
        specs = tuple(walk.collect_layout_softcores(mapper_repr))
    return Walk(specs, census_of_walk(walk), ProgramGraph.of_walk(walk))


def builder_composition(ir: Any, platform: Mapping[str, Any], residency_plan: Mapping[str, Any]) -> Any:
    """The builder's composition of the declared step over the executed IR (``compose_builder_program``, as
    ``hybrid_build_scheduled`` plans it)."""
    kwargs = ChipCapabilities.from_platform_constraints({**platform, "residency_plan": dict(residency_plan)}).layout_kwargs()
    segments = [s for s in depth_partition(ir) if isinstance(s, NeuralSegment)]
    return compose_builder_program(
        segments, ir_graph=ir, cores_config=list(platform["cores"]), residency_plan=dict(residency_plan),
        segment_specs=_segment_specs, allow_coalescing=kwargs["allow_coalescing"],
        allow_splitting=kwargs["allow_neuron_splitting"], max_schedule_passes=kwargs["max_schedule_passes"]).composition


def pass_rows(census: Any) -> List[Tuple[Any, ...]]:
    return [tuple(getattr(p, f) for f in PASS_FIELDS) for p in census.passes]


def carried(census: Any) -> Dict[int, Tuple[int, int]]:
    """Per boundary with a nonzero carry, (bytes read out, bytes injected)."""
    return {b.after_pass: (b.carry_out_bytes, b.carry_in_bytes) for b in census.boundaries
            if b.carry_out_bytes or b.carry_in_bytes}


def differences(a: Any, b: Any) -> List[str]:
    """Every difference of two censuses: Q, each pass field by field, the carry per boundary (empty: equal)."""
    out: List[str] = []
    if len(a.passes) != len(b.passes):
        out.append(f"Q {len(a.passes)} against {len(b.passes)}")
    for q, (x, y) in enumerate(zip(pass_rows(a), pass_rows(b))):
        for name, u, v in zip(PASS_FIELDS, x, y):
            if u != v:
                out.append(f"pass {q} {name} {u} against {v}")
    if carried(a) != carried(b):
        out.append(f"carry per boundary {carried(a)} against {carried(b)}")
    return out


def bytes_per_element(platform: Mapping[str, Any], timesteps: Optional[int]) -> int:
    """A carried element's bytes in the run's domain: a value at the declared grid, else one spike raster row."""
    if platform.get("core_semantics") == "mvm":
        bits = int(platform["activation_bits"])
        assert bits % 8 == 0, "a per-element byte count needs a whole-byte value grid"
        return bits // 8
    return (int(timesteps) + 7) // 8


@dataclass
class Planes:
    """The executed program's census on each plane, and its composition."""

    record: Any
    plan: Any
    candidate: Any
    composition: Any
    sealed: Any = None


def planes(hybrid: Any, ir: Any, deferred: Walk, platform: Mapping[str, Any], residency_plan: Mapping[str, Any],
           *, timesteps: Optional[int], semantics: Any, sealed_record: Any = None) -> Planes:
    """The record, plan and candidate censuses of ``hybrid``, built from ``ir`` under ``residency_plan``."""
    caps = ChipCapabilities.from_platform_constraints({**platform, "residency_plan": dict(residency_plan)})
    core_types = core_types_from_declaration(platform["cores"])
    bits = int(platform["weight_bits"])

    def census(passes, label, surface=platform, weight_bits=bits, steps=timesteps):
        written, reason = plane_write_census(passes, surface, weight_bits=weight_bits, timesteps=steps, label=label)
        assert written is not None, reason
        return written.census

    composition = builder_composition(ir, platform, residency_plan)
    plan = facts_period_census(composition.facts, forced=composition.units.forced,
                               bytes_of_cells=lambda cells: params_bytes(cells, bits), tile_cells=0,
                               bytes_per_element=bytes_per_element(platform, timesteps), label="plan")
    fragments = collect_noc_fragments(softcores=deferred.specs, core_types=core_types, census=deferred.census,
                                      program=deferred.graph, **caps.layout_kwargs())
    sealed = None if sealed_record is None else census(
        record_program_passes(sealed_record), "sealed record", sealed_record.identity.platform,
        sealed_record.utilization.crossbar.weight_bits, int(sealed_record.timing.s_global))
    return Planes(record=census(mapping_program_passes(hybrid), "record"), plan=plan,
                  candidate=census(candidate_program_passes(deferred.specs, fragments, semantics=semantics,
                                                            timesteps=timesteps), "candidate"),
                  composition=composition, sealed=sealed)


def executed_marks(hybrid: Any) -> List[Tuple[int, ...]]:
    return [tuple(sorted(getattr(s, SCHEDULE_RESIDENT_CORES))) for s in hybrid.stages if s.kind == "neural"]


def executed_plan(hybrid: Any) -> Optional[List[Dict[str, Any]]]:
    """The executed composition's plan rows; None where the build executed none ([fix_c] the fit rule's single pool)."""
    plan = getattr(hybrid, SCHEDULE_RESIDENCY_PLAN, None)
    return None if plan is None else [dataclasses.asdict(p) for p in plan]


def record_marks(schedule: Any) -> List[Tuple[int, ...]]:
    return [tuple(s.resident_core_indices) for s in schedule.segments()]


def record_plan(schedule: Any) -> Optional[List[Dict[str, Any]]]:
    """The record's plan rows; None where the record carries none."""
    plan = schedule.residency_plan
    return None if plan is None else [dataclasses.asdict(s) for s in plan.segments]


def deployed_value_model(model: Any, ir: Any, *, weight_bits: int) -> Any:
    """The model the pipeline certifies against its IR: the cached model (stored before Soft Core Mapping) with the
    IR's readout laws stamped, as Soft Core Mapping stamps them (``install_readout_laws`` -> ``stamp_model_laws``)."""
    laws = {int(n.id): n.readout_law for n in ir.nodes if isinstance(n, NeuralCore)
            and getattr(n, "readout_law", None) is not None}
    if laws:
        stamp_model_laws(model, ir, laws, weight_bits=int(weight_bits))
    return model


def integer_certificate(nf: Any, ir: Any, hybrid: Any, x_int: torch.Tensor) -> Dict[str, Tuple[int, int]]:
    """NF, SCM (identity), HCM (the executed program) and the int64 golden on both mappings, on integer inputs:
    (mismatches, compared) per edge, atol 0; the nonzero-output share rides along so no certificate is vacuous."""
    x = x_int.double()
    identity = build_identity_hybrid_mapping(ir_graph=ir)
    with torch.no_grad():
        nf_out = nf.double()(x)
        scm = ValueHybridCoreFlow(identity, dtype=torch.float64)(x)
        hcm = ValueHybridCoreFlow(hybrid, dtype=torch.float64)(x)
    codes, steps = run_golden(identity, x_int)
    codes_h, steps_h = run_golden(hybrid, x_int)

    def edge(a, b):
        a, b = torch.as_tensor(a).reshape(torch.as_tensor(b).shape).double(), torch.as_tensor(b).double()
        return int((a != b).sum()), int(b.numel())

    return {"nf_vs_scm": edge(nf_out, scm), "hcm_vs_scm": edge(hcm, scm),
            "scm_vs_golden": edge(scm, codes * steps), "hcm_vs_golden": edge(hcm, codes_h * steps_h),
            "golden_executed_vs_identity": edge(codes_h, codes),
            "nonzero_output_share": (int(np.count_nonzero(codes)), int(codes.size))}


def exact(certificate: Mapping[str, Tuple[int, int]]) -> bool:
    return all(m == 0 for name, (m, _n) in certificate.items() if name != "nonzero_output_share")
