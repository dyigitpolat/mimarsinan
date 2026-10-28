"""Candidate quantity context from the candidate's own resolved platform."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import mimarsinan.mapping.platform.target_bundles as target_bundles
from mimarsinan.deployment_record.cost.absolute.validity import declared_geometry
from mimarsinan.deployment_record.cost.compute_time import busiest_core_events, prices_per_event
from mimarsinan.deployment_record.quantities.lanes import lane_block_quantities
from mimarsinan.mapping.lanes.declaration import lane_declaration, lanes_executed
from mimarsinan.deployment_record.quantities.from_candidate import (
    CandidateQuantityContext,
)
from mimarsinan.deployment_record.quantities.value_domain import (
    ValueDomainFacts,
    names_value_domain,
    refuse_spiking_activity,
    value_bits,
    value_carry,
    value_window,
)
from mimarsinan.deployment_record.platform_physics.converter import declared_adc_bits
from mimarsinan.deployment_record.quantities.compute_technology import compute_split
from mimarsinan.deployment_record.quantities.write_law import plane_writes


def candidate_context_from_platform(
    pcfg: Mapping[str, Any],
    *,
    host_macs: Optional[int] = None,
    onchip_macs: Optional[int] = None,
    host_params: Optional[int] = None,
    onchip_params: Optional[int] = None,
    latency_steps: Optional[int] = None,
    compute_op_count: Optional[int] = None,
    programming: Optional[Any] = None,
    carry: Optional[Any] = None,
    passes: Optional[Any] = None,
    lane_ops: Optional[Mapping[str, int]] = None,
    lane_walked: bool = False,
) -> CandidateQuantityContext:
    """The declarations a candidate's platform carries, as quantity context.

    Every field comes off the RESOLVED platform dict the candidate decodes to
    (the same surface the deployment resolves), so the gate that admits an
    objective and the view that extracts it read one source. Absent or
    sentinel-zero declarations stay ``None`` — no quantity claims them. A
    platform naming the value domain reads the value rule instead of the
    spiking one (``quantities.value_domain``). ``passes`` (the candidate's
    per-(pass, segment) rows, ``ProgramPasses``) feed the value window and,
    when ``inference_batch_K`` is declared, the period census of the write law.
    A chip declaring ``onchip_arithmetic`` [W08] prices its lane block as its
    record does: ``lane_ops`` is the lane plan's estimate by kind
    (``mapping.lanes.output_stage.planned_lane_ops``; absent without a model). ``lane_walked``: the layout walk
    applied the lane plan (``lanes.shape_walk``), so the lane work is the walk's fusion count and the stage says so.
    """
    context = _spiking_rule_context(
        pcfg, host_macs=host_macs, onchip_macs=onchip_macs, host_params=host_params,
        onchip_params=onchip_params, latency_steps=latency_steps,
        compute_op_count=compute_op_count, programming=programming, carry=carry,
    )
    context = replace(context, core_geometry=declared_geometry(list(pcfg.get("cores") or ())),
                      writes=plane_writes(passes, pcfg, weight_bits=pcfg.get("weight_bits"),
                                          timesteps=context.timesteps, label="candidate", carry=carry),
                      pricing_refusal=target_bundles.absolute_pricing_refusal(pcfg),
                      compute=compute_split(pcfg, passes), readout_bits=declared_adc_bits(pcfg))
    context = _lane_context(context, pcfg, lane_ops, lane_walked)
    if not names_value_domain(pcfg):
        return _event_census(context, pcfg, passes)
    refuse_spiking_activity(pcfg)
    sized, withheld = value_carry(carry, context.timesteps, value_bits(pcfg))
    rule = ValueDomainFacts(
        mac_events=context.cells_committed,
        latency_steps=None if passes is None else value_window(
            [latency for row in passes.rows for latency in row.stage_latencies or ()]),
        carry_out_bytes=sized.get("carry_out_bytes"), carry_in_bytes=sized.get("carry_in_bytes"),
        withheld=tuple(sorted(withheld.items())),
    )
    return _event_census(replace(
        context, timesteps=1, activity_factor=None, carried_raster_bytes=None,
        latency_steps=rule.latency_steps, mac_events=rule.mac_events, carry_out_bytes=rule.carry_out_bytes,
        carry_in_bytes=rule.carry_in_bytes, value_bits=value_bits(pcfg), value_rule=rule,
    ), pcfg, passes)


def _event_census(context: CandidateQuantityContext, pcfg: Mapping[str, Any],
                  passes: Optional[Any]) -> CandidateQuantityContext:
    """[targeted M2] The per-core event census of a chip priced per synaptic operation (``prices_per_event``): per
    (pass, segment) row its busiest physical core's events by the model ``synaptic_events`` uses (the value rule's one
    event per committed cell, or cells x timesteps x the declared activity_factor), for the latency and, through the
    write law's inputs, the overlapped compute. The context unchanged on every other chip, or without the census."""
    if passes is None or not prices_per_event(pcfg.get("platform_physics_resolved")):
        return context
    value_rule = context.mac_events is not None
    if not value_rule and (context.timesteps is None or context.activity_factor is None):
        return context
    steps, activity = float(context.timesteps or 1), float(context.activity_factor or 1.0)

    def per_cell(cells: int) -> float:  # the order of from_candidate's product: cells x timesteps x activity_factor
        return float(cells) if value_rule else float(cells) * steps * activity

    events = busiest_core_events(passes, per_cell)
    writes = context.writes
    if writes is not None and writes.census is not None and len(writes.census.census.passes) == len(events):
        writes = replace(writes, pass_sops=events)
    return replace(context, pass_core_events=events, writes=writes)


#: [W08] What the candidate plane's output stage cannot see: it prices the lane-free shape's layout and host census.
CANDIDATE_STAGE_NOTE = ("candidate plane: lane work estimated from the lane plan on the candidate's model; its "
                        "layout, pass structure and host census are the lane-free shape's (no pool reduction cores, "
                        "the absorbed host ops still counted on the host)")
#: [U-E2 H-1] What a candidate plane whose walk applied the lane plan (``lanes.shape_walk``) prices.
CANDIDATE_LANE_WALK_NOTE = ("candidate plane: the lane plan applied in the shape-only walk as the build applies it "
                            "(absorbed host ops off the host census and the segmentation, operand rows of fused "
                            "adds, reduction cores of pool lanes, wires through absorbed ops carried); lane work "
                            "counted as the fusion counts it")


def _lane_context(context: CandidateQuantityContext, pcfg: Mapping[str, Any],
                  lane_ops: Optional[Mapping[str, int]], walked: bool = False) -> CandidateQuantityContext:
    """The lane block of a chip declaring ``onchip_arithmetic``, by the record plane's rule
    (``lane_block_quantities``); the context itself on every other chip."""
    declaration = lane_declaration(pcfg)
    if declaration is None:
        return context
    executed = lanes_executed(pcfg) is not None
    work = None if lane_ops is None else {kind: int(n) for kind, n in sorted(dict(lane_ops).items())}
    values, withheld = lane_block_quantities(
        classes=tuple(sorted(declaration.classes)), lanes_per_core=declaration.lanes_per_core,
        cores_per_tile=int(pcfg.get("cores_per_tile_resolved") or 1),
        lane_ops=sum((work or {}).values()) if executed else 0,
        provenance="modeled" if executed else "static")
    if work is None and executed:
        values.pop("lane_ops")  # no model, no estimate: the lane energy term names the census absent
    stage = {"label": "estimated" if executed else "no lanes executed", "classes": sorted(declaration.classes),
             "lanes_per_core": declaration.lanes_per_core, "lane_block": declaration.lane_block, "deployed": None,
             "lane_ops": work if executed and work is not None else {},
             "notes": [CANDIDATE_LANE_WALK_NOTE if walked and executed else CANDIDATE_STAGE_NOTE]}
    return replace(context, lane_quantities=tuple(sorted(values.items())),
                   lane_withheld=tuple(sorted(withheld.items())), output_stage=stage)


def _spiking_rule_context(
    pcfg: Mapping[str, Any], *, host_macs, onchip_macs, host_params, onchip_params,
    latency_steps, compute_op_count, programming, carry,
) -> CandidateQuantityContext:
    cores = [dict(ct) for ct in (pcfg.get("cores") or ())]

    def _capacity(field: str) -> Optional[int]:
        if not cores:
            return None
        return sum(int(ct.get("count", 1)) * int(ct.get(field, 0)) for ct in cores)

    rows = int(pcfg.get("tile_grid_rows_resolved", 0) or 0)
    cols = int(pcfg.get("tile_grid_cols_resolved", 0) or 0)
    steps = int(pcfg.get("simulation_steps", 0) or 0)
    activity = float(pcfg.get("activity_factor", 0.0) or 0.0)
    weight_bits = pcfg.get("weight_bits")

    cores_per_tile = int(pcfg.get("cores_per_tile_resolved", 0) or 0)
    return CandidateQuantityContext(
        timesteps=steps if steps > 0 else None,
        activity_factor=activity if activity > 0.0 else None,
        weight_bits=None if weight_bits is None else int(weight_bits),
        tiles=rows * cols if rows > 0 and cols > 0 else None,
        cores_per_tile=cores_per_tile if cores_per_tile > 0 else None,
        # The runner's tile placement is column-major over the mesh HEIGHT,
        # which the floorplan resolves as the row count.
        tile_mesh_height=rows if rows > 0 else None,
        cores_physical=(
            sum(int(ct.get("count", 1)) for ct in cores) if cores else None
        ),
        neurons_physical=_capacity("max_neurons"),
        axons_physical=_capacity("max_axons"),
        host_macs=host_macs,
        onchip_macs=onchip_macs,
        host_params=host_params,
        onchip_params=onchip_params,
        latency_steps=latency_steps,
        compute_op_count=compute_op_count,
        # [E2] Duck-typed: the search layer's programming census. Absent
        # census -> absent quantities -> the programming terms refuse.
        segment_cores=None if programming is None else int(programming.segment_cores),
        cells_committed=(
            None if programming is None else int(programming.committed_cells)
        ),
        reprogrammed_cores=(
            None if programming is None else int(programming.reprogrammed_cores)
        ),
        reprogrammed_bytes=(
            None if programming is None or programming.reprogrammed_bytes is None
            else int(programming.reprogrammed_bytes)
        ),
        reprogram_passes=(
            None if programming is None else int(programming.reprogram_passes)
        ),
        # [H2] The planned pass structure's carry census, under the run's own
        # transfer discipline — same keys the record seals, so fidelity zips.
        carried_raster_bytes=(
            None if carry is None else int(carry["carried_bytes"])
        ),
        carry_peak_live_bytes=(
            None if carry is None else int(carry["peak_live_bytes"])
        ),
        carry_out_bytes=None if carry is None else int(carry["boundary_out_bytes"]),
        carry_in_bytes=None if carry is None else int(carry["boundary_in_bytes"]),
    )
