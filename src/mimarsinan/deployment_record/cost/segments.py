"""Per-segment initialization cost: the reset constant + the programming payload.

One scheduled ``(segment × pass)`` costs a per-core reset/init constant plus
the payload its programming port has to move per round. [U-ENERGY] The payload is
what the record's PERIOD CENSUS writes in the segment (one rule with every load,
``quantities.planes.loads``): the payload bytes of the cores it rewrites plus their
exact span ``connectivity_entries`` at a MODELED byte width. A segment the census
rewrites nothing of (a resident one, or a lone written pass that stays loaded across
rounds) moves no payload: it costs exactly the reset constant.

The DMA energy is the existing per-phase model
(``weight_reuse_cost_model.phase_cost_model``) applied to those real bytes; the
programming time is the same bytes over the programming-bandwidth band (a
DIVIDING coefficient, so it reads the opposite corner and the term stays
monotone).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
from mimarsinan.deployment_record.quantities.planes.loads import period_programming, record_load_inputs
from mimarsinan.deployment_record.schema import Band, DeploymentRecord, SegmentRecord

RESIDENT_PAYLOAD_BASIS = (
    "the record's period census rewrites none of this segment's cores (a resident segment, or a lone written pass "
    "that stays loaded across rounds): no weight or connectivity payload crosses the programming port, so it costs "
    "exactly the per-core reset constant"
)


def payload_bytes_band(
    segment: SegmentRecord, path: str, bytes_per_connectivity_entry: Band,
    written: Tuple[int, Optional[int]] = (0, 0),
) -> Band:
    """The payload the period census writes in the segment, in bytes (``written``: its rewritten cores' payload bytes
    and connectivity entries; identically zero where it rewrites none)."""
    payload, entries = written
    if not payload and not entries:
        return Band(low=0.0, nominal=0.0, high=0.0, basis=RESIDENT_PAYLOAD_BASIS)
    unstated = "" if entries is not None else "; the record states no connectivity entries per core: unpriced"
    return banded(
        lambda corner: float(payload) + float(entries or 0) * corner_value(bytes_per_connectivity_entry, corner),
        basis=(
            f"{path}: the period census's rewritten cores, their payload bytes (exact) + their connectivity_entries "
            f"(exact) x bytes_per_connectivity_entry: {bytes_per_connectivity_entry.basis}{unstated}"
        ),
    )


def segment_cost(
    segment: SegmentRecord,
    position: int,
    *,
    core_init: CoreInitCoefficients,
    bytes_per_connectivity_entry: Band,
    programming_bandwidth_bytes_per_s: Band,
    written: Tuple[int, Optional[int]] = (0, 0),
) -> SegmentInitCost:
    """One segment's initialization cost, every term banded and sourced."""
    path = f"schedule.stages[{position}]"
    cores = len(segment.cores)
    payload = payload_bytes_band(segment, path, bytes_per_connectivity_entry, written)
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


def programming_refusal(record: DeploymentRecord) -> str:
    """Why the cost model cannot price a record's per-segment programming ("" when it can): it prices what the
    record's period census writes (one rule with every load), so a record whose census cannot be stated is refused by
    this name, as a precondition of the model, never mid-report."""
    written, why = period_programming(record_load_inputs(record))
    if written is not None:
        return ""
    return ("the cost model's per-segment programming prices what the record's period census writes (one rule with "
            f"every load), and that census cannot be stated: {why}")


def segment_costs(
    record: DeploymentRecord,
    *,
    core_init: CoreInitCoefficients,
    bytes_per_connectivity_entry: Band,
    programming_bandwidth_bytes_per_s: Band,
) -> Tuple[SegmentInitCost, ...]:
    """Every scheduled segment's initialization cost, in schedule order: each segment programs what the record's period
    census writes in it (one census pass per segment, in schedule order), refused by name where that census cannot be
    stated (``programming_refusal``, the model's precondition)."""
    written, _ = period_programming(record_load_inputs(record))
    if written is None:
        raise ValueError(programming_refusal(record))
    segments = [(position, stage) for position, stage in enumerate(record.schedule.stages)
                if isinstance(stage, SegmentRecord)]
    return tuple(
        segment_cost(
            stage,
            position,
            core_init=core_init,
            bytes_per_connectivity_entry=bytes_per_connectivity_entry,
            programming_bandwidth_bytes_per_s=programming_bandwidth_bytes_per_s,
            written=writes,
        )
        for (position, stage), writes in zip(segments, written, strict=True)
    )
