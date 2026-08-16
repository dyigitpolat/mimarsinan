"""Seal what crossed the intra-segment pass boundaries, and under which discipline."""

from __future__ import annotations

from typing import Any, Optional

from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_wire_spans,
    pass_carry_census,
)
from mimarsinan.deployment_record.schema import PassCarryRecord


def carry_record_from_mapping(
    hybrid_mapping: Any, timesteps: Any, transfer: Any
) -> Optional[PassCarryRecord]:
    """The intra-segment pass-boundary census, or None when nothing crosses one.

    A program that DOES carry but was handed no timestep count or discipline would
    silently drop the fact that the two disciplines are different computations, so
    that combination raises rather than sealing a None.
    """
    if not carried_wire_spans(hybrid_mapping.stages):
        return None
    if timesteps is None or transfer is None:
        raise ValueError(
            "this program cuts a neural segment into passes, so the record must "
            "state what crossed those boundaries and under which discipline; "
            f"got timesteps={timesteps!r}, pass_transfer={transfer!r}"
        )
    census = pass_carry_census(hybrid_mapping.stages, int(timesteps), str(transfer))
    return PassCarryRecord(
        transfer=str(transfer), timesteps=int(timesteps), **census,
    )


def enforce_declared_buffer_capacity(
    hybrid_mapping: Any, timesteps: Any, transfer: Any, capacity_bytes: Any,
) -> None:
    """[B] The declared pass-buffer ceiling, enforced at MAPPING time.

    ``pass_buffer_capacity_bytes`` <= 0 (or None) means undeclared — no gate.
    A program whose worst pass boundary needs more live raster bytes than the
    platform declares refuses BEFORE any simulation runs, naming both numbers.
    The required buffer itself stays a reported metric (the record axes);
    this ceiling is opt-in and never a search axis.
    """
    capacity = int(capacity_bytes or 0)
    if capacity <= 0:
        return
    record = carry_record_from_mapping(hybrid_mapping, timesteps, transfer)
    if record is None:
        return
    if int(record.peak_live_bytes) > capacity:
        raise ValueError(
            f"the scheduled program's worst pass boundary needs "
            f"{record.peak_live_bytes} live raster bytes but the platform "
            f"declares pass_buffer_capacity_bytes={capacity}; loosen the "
            f"schedule (fewer passes / smaller cuts) or declare the buffer "
            f"the chip actually has"
        )
