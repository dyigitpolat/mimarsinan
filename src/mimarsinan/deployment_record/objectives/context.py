"""Candidate quantity context from the candidate's own resolved platform."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from mimarsinan.deployment_record.quantities.from_candidate import (
    CandidateQuantityContext,
)


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
) -> CandidateQuantityContext:
    """The declarations a candidate's platform carries, as quantity context.

    Every field comes off the RESOLVED platform dict the candidate decodes to
    (the same surface the deployment resolves), so the gate that admits an
    objective and the view that extracts it read one source. Absent or
    sentinel-zero declarations stay ``None`` — no quantity claims them.
    """
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
