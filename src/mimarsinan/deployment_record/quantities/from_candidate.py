"""Quantities of a search candidate: static facts of the shape, assumptions marked."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


class _NocEstimateView(Protocol):
    """The modeled NoC census a candidate's fragments price to (duck-typed;
    produced by the SANA-FE estimator, which this package must not import)."""

    @property
    def total_packets(self) -> float: ...
    @property
    def inter_tile_packets(self) -> float: ...
    @property
    def intra_tile_packets(self) -> float: ...
    @property
    def total_hops(self) -> float: ...


class _LayoutStatsView(Protocol):
    """The packing facts a candidate layout answers.

    Read-only properties, so any object exposing them satisfies it — including the
    objectives layer's own layout protocol, which this package must not import
    (quantities sits below it).
    """

    @property
    def total_hw_cores(self) -> int: ...
    @property
    def schedule_pass_count(self) -> int: ...
    @property
    def schedule_sync_count(self) -> int: ...
    @property
    def neural_segment_count(self) -> int: ...


@dataclass(frozen=True)
class CandidateQuantityContext:
    """Run declarations and censuses the layout itself does not carry.

    Every field is optional: an absent declaration keeps its quantities absent, so a
    candidate never claims a number nobody produced. The host/onchip censuses are
    filled by the search's layout hook (C3 wires the flow census); ``activity_factor``
    is the operator's declared switching-activity assumption.
    """

    timesteps: Optional[int] = None
    #: [E1] Σ over the program's EXECUTION stages of their executed windows,
    #: computed upstream through ``chip_simulation.stage_timesteps`` — the same
    #: rule the runner sizes its simulation with. ``None`` when the candidate
    #: cannot determine the stage structure, which makes every latency-bearing
    #: axis refuse BY NAME instead of pricing a wall that is 3.75x short (the
    #: measured error of the retired ``timesteps x neural_segment_count``).
    latency_steps: Optional[int] = None
    #: [E2] The programming census of the candidate's pass structure, computed
    #: upstream under the DEPLOYED residency law: cores over every pass (core
    #: init is paid whether or not weights stayed), and cores/bytes over the
    #: passes that actually install weights. Absent without a pass structure,
    #: so the programming terms refuse instead of pricing zero.
    segment_cores: Optional[int] = None
    reprogrammed_cores: Optional[int] = None
    reprogrammed_bytes: Optional[int] = None
    reprogram_passes: Optional[int] = None
    activity_factor: Optional[float] = None
    weight_bits: Optional[int] = None
    tiles: Optional[int] = None
    cores_per_tile: Optional[int] = None
    tile_mesh_height: Optional[int] = None
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
    noc: Optional[_NocEstimateView] = None,
) -> Quantities:
    """The static quantities a candidate can honestly answer.

    Never claimed here: execution spike counts, measured host walls, and the
    as-mapped occupancy censuses (``cells_used``, ``macs``) — those are facts
    of a deployment, not of a shape. Hop counts ARE claimable once ``noc``
    carries a placement-backed estimate (the wireload model), with provenance
    ``modeled``.
    """
    del host_side_segment_count  # a view fact today; no quantity reads it yet
    values: Dict[str, QuantityValue] = {}

    _put(values, "cells_physical", chip_param_capacity)
    _put(values, "total_params", total_params)

    if layout is not None:
        _put(values, "pass_count", layout.schedule_pass_count)
        _put(values, "sync_count", layout.schedule_sync_count)

    # [E1] The executed wall comes from the stage structure through the SAME
    # rule the runner uses, or it is ABSENT. It is never re-derived here: the
    # retired local formula (timesteps x neural_segment_count) ignored both
    # depth levels and the input-delivery cycle.
    _put(values, "latency_steps", context.latency_steps)

    # [E2] Programming multiplicands under the residency law — a resident pass
    # pays core init and nothing else. ``cores_allocated`` is the SAME count
    # the record means (cores over every pass), not the declared chip's core
    # count the candidate used to report under that name.
    _put(values, "cores_allocated", context.segment_cores)
    _put(values, "segment_cores", context.segment_cores)
    _put(values, "reprogrammed_cores", context.reprogrammed_cores)
    _put(values, "reprogrammed_bytes", context.reprogrammed_bytes)
    _put(values, "reprogram_passes", context.reprogram_passes)

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

    if noc is not None:
        for key, value in (
            ("noc_total_packets", noc.total_packets),
            ("noc_intra_tile_packets", noc.intra_tile_packets),
            ("noc_inter_tile_packets", noc.inter_tile_packets),
            ("noc_total_hops", noc.total_hops),
        ):
            values[key] = QuantityValue(value=float(value), provenance="modeled")

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
