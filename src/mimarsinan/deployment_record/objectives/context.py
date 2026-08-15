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

    return CandidateQuantityContext(
        timesteps=steps if steps > 0 else None,
        activity_factor=activity if activity > 0.0 else None,
        weight_bits=None if weight_bits is None else int(weight_bits),
        tiles=rows * cols if rows > 0 and cols > 0 else None,
        cores_physical=(
            sum(int(ct.get("count", 1)) for ct in cores) if cores else None
        ),
        neurons_physical=_capacity("max_neurons"),
        axons_physical=_capacity("max_axons"),
        host_macs=host_macs,
        onchip_macs=onchip_macs,
        host_params=host_params,
        onchip_params=onchip_params,
    )
