"""Per-segment initialization cost: the reset constant + the programming payload.

One scheduled ``(segment × pass)`` costs a per-core reset/init constant plus
the payload its programming port has to move: the exact ``params_bytes`` the
mapping wrote plus the exact span ``connectivity_entries`` at a MODELED byte
width. A **resident** segment moves no payload at all — it costs exactly the
reset constant — because nothing is reprogrammed into its cores.

The DMA energy is the existing per-phase model
(``weight_reuse_cost_model.phase_cost_model``) applied to those real bytes; the
programming time is the same bytes over the programming-bandwidth band (a
DIVIDING coefficient, so it reads the opposite corner and the term stays
monotone).
"""

from __future__ import annotations

from typing import Tuple

from mimarsinan.deployment_record.cost.coefficients import (
    E_DMA_PER_BYTE_MJ,
    CoreInitCoefficients,
    banded,
    corner_value,
    dma_energy_mj,
    opposite_corner,
)
from mimarsinan.deployment_record.cost.combine import modeled
from mimarsinan.deployment_record.cost.terms import SegmentInitCost
from mimarsinan.deployment_record.schema import Band, DeploymentRecord, SegmentRecord

RESIDENT_PAYLOAD_BASIS = (
    "a resident segment is not reprogrammed: no weight or connectivity payload "
    "crosses the programming port, so it costs exactly the per-core reset "
    "constant"
)


def payload_bytes_band(
    segment: SegmentRecord, path: str, bytes_per_connectivity_entry: Band
) -> Band:
    """The programmed payload in bytes (identically zero for a resident pass)."""
    if segment.programming == "resident":
        return Band(low=0.0, nominal=0.0, high=0.0, basis=RESIDENT_PAYLOAD_BASIS)
    return banded(
        lambda corner: float(segment.params_bytes)
        + float(segment.connectivity_entries)
        * corner_value(bytes_per_connectivity_entry, corner),
        basis=(
            f"{path}.params_bytes (exact) + {path}.connectivity_entries (exact) "
            f"x bytes_per_connectivity_entry: {bytes_per_connectivity_entry.basis}"
        ),
    )


def segment_cost(
    segment: SegmentRecord,
    position: int,
    *,
    core_init: CoreInitCoefficients,
    bytes_per_connectivity_entry: Band,
    programming_bandwidth_bytes_per_s: Band,
) -> SegmentInitCost:
    """One segment's initialization cost, every term banded and sourced."""
    path = f"schedule.stages[{position}]"
    cores = len(segment.cores)
    payload = payload_bytes_band(segment, path, bytes_per_connectivity_entry)
    terms = (
        modeled(
            "reset_energy_mj", "mJ",
            banded(
                lambda corner: cores * corner_value(core_init.energy_mj, corner),
                basis=core_init.energy_mj.basis,
            ),
            f"{path}.cores x CoreInitCoefficients.energy_mj",
        ),
        modeled(
            "reset_time_s", "s",
            banded(
                lambda corner: cores * corner_value(core_init.time_s, corner),
                basis=core_init.time_s.basis,
            ),
            f"{path}.cores x CoreInitCoefficients.time_s",
        ),
        modeled("payload_bytes", "B", payload, f"{path} programmed payload"),
        modeled(
            "payload_energy_mj", "mJ",
            banded(
                lambda corner: dma_energy_mj(corner_value(payload, corner), corner),
                basis=E_DMA_PER_BYTE_MJ.basis,
            ),
            f"weight_reuse_cost_model.phase_cost_model DMA channel on {path}",
        ),
        modeled(
            "programming_time_s", "s",
            banded(
                lambda corner: corner_value(payload, corner)
                / corner_value(
                    programming_bandwidth_bytes_per_s, opposite_corner(corner)
                ),
                basis=programming_bandwidth_bytes_per_s.basis,
            ),
            f"{path} payload / programming_bandwidth_bytes_per_s",
        ),
    )
    return SegmentInitCost(
        stage_index=segment.stage_index,
        segment_index=segment.segment_index,
        pass_index=segment.pass_index,
        programming=segment.programming,
        core_count=cores,
        terms=terms,
    )


def segment_costs(
    record: DeploymentRecord,
    *,
    core_init: CoreInitCoefficients,
    bytes_per_connectivity_entry: Band,
    programming_bandwidth_bytes_per_s: Band,
) -> Tuple[SegmentInitCost, ...]:
    """Every scheduled segment's initialization cost, in schedule order."""
    return tuple(
        segment_cost(
            stage,
            position,
            core_init=core_init,
            bytes_per_connectivity_entry=bytes_per_connectivity_entry,
            programming_bandwidth_bytes_per_s=programming_bandwidth_bytes_per_s,
        )
        for position, stage in enumerate(record.schedule.stages)
        if isinstance(stage, SegmentRecord)
    )
