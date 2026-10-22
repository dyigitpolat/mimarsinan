"""Candidate quantity context from the candidate's own resolved platform."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

from mimarsinan.deployment_record.cost.absolute.validity import declared_geometry
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
    """
    context = _spiking_rule_context(
        pcfg, host_macs=host_macs, onchip_macs=onchip_macs, host_params=host_params,
        onchip_params=onchip_params, latency_steps=latency_steps,
        compute_op_count=compute_op_count, programming=programming, carry=carry,
    )
    context = replace(context, core_geometry=declared_geometry(list(pcfg.get("cores") or ())),
                      writes=plane_writes(passes, pcfg, weight_bits=pcfg.get("weight_bits"),
                                          timesteps=context.timesteps, label="candidate", carry=carry))
    if not names_value_domain(pcfg):
        return context
    refuse_spiking_activity(pcfg)
    sized, withheld = value_carry(carry, context.timesteps, value_bits(pcfg))
    rule = ValueDomainFacts(
        mac_events=context.cells_committed,
        latency_steps=None if passes is None else value_window(
            [latency for row in passes.rows for latency in row.stage_latencies or ()]),
        carry_out_bytes=sized.get("carry_out_bytes"), carry_in_bytes=sized.get("carry_in_bytes"),
        withheld=tuple(sorted(withheld.items())),
    )
    return replace(
        context, timesteps=1, activity_factor=None, carried_raster_bytes=None,
        latency_steps=rule.latency_steps, mac_events=rule.mac_events, carry_out_bytes=rule.carry_out_bytes,
        carry_in_bytes=rule.carry_in_bytes, value_bits=value_bits(pcfg), value_rule=rule,
    )


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
