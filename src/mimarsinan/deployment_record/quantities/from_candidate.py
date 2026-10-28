"""Quantities of a search candidate: static facts of the shape, assumptions marked."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import BUFFER_CAPACITY, ONCHIP_CARRY
from mimarsinan.deployment_record.quantities.write_law import write_quantities


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
    #: [U-N] The allocated physical cores over every pass (the census's per-pass cores summed): a wavefront head's
    #: dark cores are allocated (loaded) but not initialized, so this is the record's ``cores_allocated`` and may
    #: exceed ``segment_cores``.
    cores_allocated: Optional[int] = None
    #: [R1] The as-mapped committed cells of the planned program — the
    #: catalog's ``macs``/``cells_used`` at candidate completeness, and the
    #: event model's multiplicand (replicas really fire).
    cells_committed: Optional[int] = None
    reprogrammed_cores: Optional[int] = None
    reprogrammed_bytes: Optional[int] = None
    reprogram_passes: Optional[int] = None
    #: [H2] The planned pass structure's carry census under the run's sealed
    #: transfer discipline. Absent without the wire census, the discipline,
    #: or any crossing wire — absence means what the record's absence means.
    carried_raster_bytes: Optional[int] = None
    carry_peak_live_bytes: Optional[int] = None
    carry_out_bytes: Optional[int] = None
    carry_in_bytes: Optional[int] = None
    #: [U-F4] The carried bytes a declared on-chip carry buffer holds, and why the program does not fit it.
    carry_onchip_bytes: Optional[int] = None
    carry_onchip_refusal: Optional[str] = None
    #: [U-F4] The declared on-chip carry buffer's capacity (``carry_buffer_bytes``), which its leakage and area price.
    carry_buffer_capacity_bytes: Optional[int] = None
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
    #: [R2] Host ComputeOp invocations of the flow — the per-invocation
    #: overhead multiplicand; from the same walk as the host MAC census.
    compute_op_count: Optional[int] = None
    #: The value-domain rule's event census (every committed cell computes once
    #: per inference): ``synaptic_events`` as a static fact, never the activity model.
    mac_events: Optional[int] = None
    #: The value grid of a value-domain run (``activation_bits``), for the value NoC bytes.
    value_bits: Optional[int] = None
    #: The value rule's answers (``value_domain.ValueDomainFacts``) when the run names the value
    #: domain, else None; its ``withheld`` pairs name the quantities the candidate cannot size, so
    #: the terms that multiply them are named unpriced (``Quantities(withheld=)``).
    value_rule: Optional[Any] = None
    #: The declared core types (``validity.CoreGeometry``), for the aggregates' validity.
    core_geometry: Optional[Any] = None
    #: The write law's inputs (``write_law.PlaneWrites``), present only when
    #: ``inference_batch_K`` is declared.
    writes: Optional[Any] = None
    #: [W08] The lane block of a chip declaring ``onchip_arithmetic`` (``lanes.lane_block_quantities``): the lane
    #: work estimated from the lane plan and lanes per tile, and the quantities withheld by name; empty otherwise.
    lane_quantities: Tuple[Tuple[str, QuantityValue], ...] = ()
    lane_withheld: Tuple[Tuple[str, str], ...] = ()
    #: [W08] The candidate's output stage, printed beside its priced point (``output_stage_note``).
    output_stage: Optional[Mapping[str, Any]] = None
    #: [CL2] The ``(name, reason)`` pair of a target that declares no physics
    #: (``target_bundles.absolute_pricing_refusal``): the candidate's report carries it alone.
    pricing_refusal: Optional[Any] = None
    #: [targeted M2] Per (pass, segment) row, the synaptic events of its busiest physical core by the same model as
    #: ``synaptic_events`` (``compute_time.busiest_core_events``); formed only where the resolved physics prices
    #: compute per synaptic operation, None elsewhere (so no other chip's quantities change).
    pass_core_events: Optional[Tuple[float, ...]] = None
    #: [E-6] The MAC events per core type (``compute_technology.compute_split``), present only when a core type names
    #: a compute physics: each type's MAC energy at its own e_mac.
    compute: Optional[Any] = None
    #: [E-5] The run's converter resolution (``readout_law.adc_bits``), None when undeclared.
    readout_bits: Optional[int] = None


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
    noc_value_bytes: Optional[float] = None,
    withheld: Optional[Mapping[str, str]] = None,
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
    # the record means (the allocated cores over every pass, a wavefront head's
    # dark cores included [U-N]), not the declared chip's core count the
    # candidate used to report under that name.
    _put(values, "cores_allocated", context.cores_allocated)
    _put(values, "segment_cores", context.segment_cores)
    # [R1] One figure, two catalog names at this completeness (stated): the
    # committed rectangle IS the as-mapped MAC-site census of the shape.
    _put(values, "macs", context.cells_committed)
    _put(values, "cells_used", context.cells_committed)
    _put(values, "reprogrammed_cores", context.reprogrammed_cores)
    _put(values, "reprogrammed_bytes", context.reprogrammed_bytes)
    _put(values, "reprogram_passes", context.reprogram_passes)

    # [H2] Carry under the run's own discipline — the cost the composed schedule moves.
    _put(values, "carried_raster_bytes", context.carried_raster_bytes)
    _put(values, "carry_peak_live_bytes", context.carry_peak_live_bytes)
    _put(values, "carry_out_bytes", context.carry_out_bytes)
    _put(values, "carry_in_bytes", context.carry_in_bytes)
    _put(values, ONCHIP_CARRY, None if context.carry_onchip_refusal else context.carry_onchip_bytes)
    _put(values, BUFFER_CAPACITY, context.carry_buffer_capacity_bytes)

    _put(values, "timesteps", context.timesteps)
    _put(values, "weight_bits", context.weight_bits)
    _put(values, "tiles", context.tiles)
    _put(values, "cores_physical", context.cores_physical)
    _put(values, "neurons_physical", context.neurons_physical)
    _put(values, "axons_physical", context.axons_physical)
    _put(values, "host_macs", context.host_macs)
    _put(values, "onchip_macs", context.onchip_macs)
    _put(values, "host_params", context.host_params)
    _put(values, "compute_op_count", context.compute_op_count)
    _put(values, "onchip_params", context.onchip_params)
    if context.host_macs is not None and context.onchip_macs is not None:
        _put(values, "total_macs", context.host_macs + context.onchip_macs)

    _put(values, "noc_value_bytes", noc_value_bytes)
    writes = context.writes
    if writes is not None and writes.census is not None:
        values.update(write_quantities(
            writes.census.census, writes.policy, written_extents=writes.census.written_extents,
            most_written=writes.census.most_written_core, producer_slots=writes.census.producer_slots,
        ))

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
    # [R1] Events are modeled over the AS-MAPPED cells, never the logical
    # census: a replicated bank's copies really fire (measured ~108x apart on
    # the offload LeNet5). Absent without a pass structure — an energy claim
    # about an unmapped shape would be the silent-zero's mirror image.
    _put(values, "synaptic_events", context.mac_events)
    if (
        context.mac_events is None
        and context.cells_committed is not None
        and context.timesteps is not None
        and context.activity_factor is not None
    ):
        values["synaptic_events"] = QuantityValue(
            value=float(context.cells_committed)
            * float(context.timesteps)
            * float(context.activity_factor),
            provenance="modeled",
        )

    # [targeted M2] The per-core event census beside the total it splits, by the same model, where the context forms it.
    if "synaptic_events" in values and context.pass_core_events is not None:
        values["synaptic_events_busiest_core"] = QuantityValue(
            value=float(sum(context.pass_core_events)), provenance=values["synaptic_events"].provenance)
    values.update(dict(context.lane_quantities))
    held = {**dict(getattr(context.value_rule, "withheld", ()) or ()), **dict(context.lane_withheld),
            **dict(withheld or {}),
            **({ONCHIP_CARRY: context.carry_onchip_refusal} if context.carry_onchip_refusal else {})}
    return Quantities(values, withheld={key: reason for key, reason in held.items() if key not in values})
