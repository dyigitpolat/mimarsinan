"""Quantities of a search candidate: static facts of the shape, assumptions marked."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


class _LayoutStatsView(Protocol):
    """The packing facts a candidate layout answers (duck-typed like LayoutStatsView)."""

    total_hw_cores: int
    schedule_pass_count: int
    schedule_sync_count: int


@dataclass(frozen=True)
class CandidateQuantityContext:
    """Run declarations and censuses the layout itself does not carry.

    Every field is optional: an absent declaration keeps its quantities absent, so a
    candidate never claims a number nobody produced. The host/onchip censuses are
    filled by the search's layout hook (C3 wires the flow census); ``activity_factor``
    is the operator's declared switching-activity assumption.
    """

    timesteps: Optional[int] = None
    activity_factor: Optional[float] = None
    weight_bits: Optional[int] = None
    tiles: Optional[int] = None
    cores_physical: Optional[int] = None
    neurons_physical: Optional[int] = None
    axons_physical: Optional[int] = None
    host_macs: Optional[int] = None
    onchip_macs: Optional[int] = None
    host_params: Optional[int] = None
    onchip_params: Optional[int] = None


def _put(values: Dict[str, QuantityValue], key: str, value: Optional[float],
         provenance: str = "static") -> None:
    if value is None:
        return
    values[key] = QuantityValue(value=float(value), provenance=provenance)


def from_candidate(
    *,
    layout: Optional[_LayoutStatsView],
    chip_param_capacity: Optional[float],
    total_params: Optional[float],
    host_side_segment_count: Optional[int],
    context: CandidateQuantityContext,
) -> Quantities:
    """The static quantities a candidate can honestly answer.

    Never claimed here: placement-dependent hop counts, execution spike counts,
    measured host walls, and the as-mapped occupancy censuses (``cells_used``,
    ``macs``) — those are facts of a deployment, not of a shape.
    """
    del host_side_segment_count  # a view fact today; no quantity reads it yet
    values: Dict[str, QuantityValue] = {}

    _put(values, "cells_physical", chip_param_capacity)
    _put(values, "total_params", total_params)

    if layout is not None:
        _put(values, "cores_allocated", layout.total_hw_cores)
        _put(values, "pass_count", layout.schedule_pass_count)
        _put(values, "sync_count", layout.schedule_sync_count)

    _put(values, "timesteps", context.timesteps)
    _put(values, "weight_bits", context.weight_bits)
    _put(values, "tiles", context.tiles)
    _put(values, "cores_physical", context.cores_physical)
    _put(values, "neurons_physical", context.neurons_physical)
    _put(values, "axons_physical", context.axons_physical)
    _put(values, "host_macs", context.host_macs)
    _put(values, "onchip_macs", context.onchip_macs)
    _put(values, "host_params", context.host_params)
    _put(values, "onchip_params", context.onchip_params)
    if context.host_macs is not None and context.onchip_macs is not None:
        _put(values, "total_macs", context.host_macs + context.onchip_macs)

    # The EDA switching-activity discipline: candidate spike-dependent energy rests
    # on a DECLARED activity factor over the logical on-chip MAC census, and the
    # provenance says so. (The as-mapped replication correction is fidelity-tracked.)
    if (
        context.onchip_macs is not None
        and context.timesteps is not None
        and context.activity_factor is not None
    ):
        values["synaptic_events"] = QuantityValue(
            value=float(context.onchip_macs)
            * float(context.timesteps)
            * float(context.activity_factor),
            provenance="modeled",
        )

    return Quantities(values)
