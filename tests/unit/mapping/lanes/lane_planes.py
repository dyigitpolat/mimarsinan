"""The two planes of one lane program (WI-5e, U-E2 H-1): the exploration's shape-only plane (the scaling ladder's
``shape_census``, ``packed_tiles`` and ``schedule_program``, as CL5 and W48 call them) and the builder's (the lane
program's fused IR built by the pipeline's hybrid builder), read as the same facts: neural segments, host ops, the
planner's specs, passes, physical cores and cells per pass, and the carry census."""

from __future__ import annotations

import contextlib
import functools
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mimarsinan.deployment_record.build.pass_carry import candidate_carry_census
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.packing.segment_specs import segment_specs_for
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.schedule.pass_carry import pass_carry_census
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
from mimarsinan.mapping.support.wavefront_stages import stage_groups
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}
#: One raster step: the carry census needs a declared window (value-domain programs read one).
STEPS = 1
TRANSFER = transfer_for(streamed=False)


@functools.lru_cache(maxsize=1)
def ladder():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import scaling_ladder
    finally:
        sys.path.remove(str(SCRIPTS))
    return scaling_ladder


def plane_platform(classes: Optional[Sequence[str]], *, axons: int, neurons: int, count: int = 999,
                   passes: Optional[int] = None, coalescing: bool = False, hardware_bias: bool = False,
                   multiplexing: bool = False, readout_law: Optional[dict] = LAW) -> Dict[str, Any]:
    """A chip both planes read: one core type, the scheduling permissions, the lane declaration and the integer path
    (``readout_law``; without it the declared lanes do not run)."""
    platform: Dict[str, Any] = {
        "cores": [{"max_axons": axons, "max_neurons": neurons, "count": count, "has_bias": hardware_bias}],
        "allow_scheduling": passes is not None, "max_schedule_passes": passes or 8, "allow_coalescing": coalescing,
        "allow_neuron_splitting": True, "weight_bits": 4, "simulation_steps": STEPS,
    }
    if classes is not None:
        platform["onchip_arithmetic"] = {"classes": list(classes)}
    if readout_law is not None:
        platform["readout_law"] = dict(readout_law)
    if multiplexing:
        platform["position_multiplexing"] = True
    return platform


def ir_geometry(platform: Dict[str, Any]) -> Dict[str, Any]:
    """The tiler limits Soft Core Mapping maps the IR at on ``platform`` (the layout walk's own)."""
    params = resolve_platform_mapping_params(platform["cores"], allow_coalescing=bool(platform["allow_coalescing"]))
    return dict(max_axons=params.effective_max_axons, max_neurons=params.effective_max_neurons,
                hardware_bias=params.hardware_bias, allow_coalescing=params.allow_coalescing)


def network_of(program, num_classes: int):
    """The lane program's own mapper graph as the ladder's network (the host census measured on its flow)."""
    from mimarsinan.deployment_record.introspection.scaling_ladder import LadderNetwork

    with contextlib.redirect_stdout(io.StringIO()):
        params, macs, ops = estimate_onchip_fractions(program.model, program.input_shape, num_classes,
                                                      metrics=("params", "macs", "ops"))
    return LadderNetwork(
        name="lane fixture", mapper_repr=program.model.get_mapper_repr(), resolution=tuple(program.input_shape),
        total_params=float(sum(p.numel() for p in program.model.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip})


def spec_row(spec) -> tuple:
    return (int(spec.input_count), int(spec.output_count), int(getattr(spec, "operand_axons", 0) or 0),
            int(spec.latency_tag or 0), int(spec.segment_id or 0), spec.bank_id, spec.perceptron_index)


def candidate_plane(program, platform: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    from mimarsinan.deployment_record.introspection import scaling_ladder as LD

    network = network_of(program, num_classes)
    census = LD.shape_census(network, platform)
    tiles = LD.packed_tiles(census, platform)
    scheduled = LD.schedule_program(network, census, platform, ladder().program_facts, tiles=tiles)
    assert scheduled.refusal is None, scheduled.refusal
    host = getattr(LD, "host_census", lambda net, _census: dict(net.host_census))(network, census)
    softcores = getattr(census, "residency_softcores", None) if platform.get("residency_plan") else None
    specs = softcores or census.softcores
    return {"segments": int(census.segments), "host_ops": int(host["compute_op_count"]),
            "specs": [spec_row(s) for s in specs], **period_rows(scheduled.facts.passes),
            "carry": candidate_carry_census(specs, scheduled.noc, timesteps=STEPS, pass_transfer=TRANSFER)}


def period_rows(passes) -> Dict[str, Any]:
    """The period census's rows a plane states (``write_census.PassRow``: one per executed (pass, segment)): their
    count, physical cores and cells per row, and the cores each rewrites per period as the census reads the row
    (neither a resident pass nor a marked core: ``write_census``'s ``unkept``)."""
    rows = passes.rows
    return {"passes": len(rows), "tiles": [len(r.core_cells) for r in rows],
            "cells": [int(sum(r.core_cells)) for r in rows],
            "written": [tuple(i for i in range(len(r.core_cells)) if not r.resident and i not in r.marks)
                        for r in rows]}


def builder_plane(program, platform: Dict[str, Any]) -> Dict[str, Any]:
    mapping = build_hybrid_mapping_for_pipeline(program.ir, platform)
    segments = [s for s in partition_ir_graph(program.ir) if isinstance(s, NeuralSegment)]
    reader = segment_specs_for(platform["cores"])
    specs = [spec_row(s) for segment in segments for s in reader(segment.nodes, ir_graph=program.ir)[0]]
    return {"segments": len(segments), "host_ops": sum(isinstance(n, ComputeOp) for n in program.ir.nodes),
            "specs": specs, **period_rows(mapping_program_passes(mapping)),
            # [U-F4] with the stages' wavefront groups, as the record seals them (none: the census unchanged)
            "carry": pass_carry_census(mapping.stages, STEPS, TRANSFER, groups=stage_groups(mapping.stages))}


def builder_depth_specs(program, platform: Dict[str, Any]) -> list:
    """The builder's specs under a declared ``residency_plan`` (``depth_partition``: the fused IR's depth segments),
    each with its depth segment's ordinal in place of the spec's own segment id."""
    from mimarsinan.mapping.packing.schedule_residency_plan import depth_partition

    reader = segment_specs_for(platform["cores"])
    segments = [s for s in depth_partition(program.ir) if isinstance(s, NeuralSegment)]
    return [spec_row(s)[:4] + (ordinal,) + spec_row(s)[5:] for ordinal, segment in enumerate(segments)
            for s in reader(segment.nodes, ir_graph=program.ir)[0]]
