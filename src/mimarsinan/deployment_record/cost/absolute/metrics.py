"""price_absolute: the target's physics × a view's quantities → absolute cost terms.

One pure engine for both completenesses (a candidate's static quantities | a sealed
record's measured ones). Every term = constant band × quantity with the evidence in
its source; a headline that cannot be answered honestly is REFUSED with a written
reason, never approximated silently.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.context import (
    ABSOLUTE_PRICING as ABSOLUTE_PRICING,
    AbsolutePricing,
    PricingContext,
    external_refusals as external_refusals,
)
from mimarsinan.deployment_record.cost.absolute.energy import price_energy
from mimarsinan.deployment_record.cost.absolute.formulas import (
    AREA_COMPONENTS,
    CHIP_AREA_ADDERS,
    LATENCY_COMPONENTS,
    add_bands,
    invert_band,
    programming_payload_bytes,
    scale_band,
)
from mimarsinan.deployment_record.cost.absolute.validity import (
    aggregate_scales,
    footprint_cores,
    judged_geometry,
    mean_core_note,
)
from mimarsinan.deployment_record.cost.absolute.writes import (
    WriteInputs,
    WritePricing,
    price_write_inputs,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band

_M2_TO_MM2 = 1e6

AREA_HEADLINE = "chip_area_mm2"
LATENCY_HEADLINE = "e2e_latency_s"
THROUGHPUT_HEADLINE = "throughput_inferences_s"


def _state_area_band(ctx: PricingContext) -> Optional[Band]:
    """State storage = area_per_state_bit × membrane_bits × neurons_physical."""
    if not (ctx.priced("area_per_state_bit") and ctx.priced("membrane_bits")
            and ctx.quantities.has("neurons_physical")):
        return None
    per_bit = ctx.physics.band("area_per_state_bit")
    bits = ctx.physics.band("membrane_bits")
    return scale_band(
        Band(per_bit.low * bits.low, per_bit.nominal * bits.nominal,
             per_bit.high * bits.high, "area_per_state_bit x membrane_bits"),
        ctx.quantities.get("neurons_physical").value * _M2_TO_MM2,
        "area_per_state_bit x membrane_bits x neurons_physical",
    )


def _aggregate_footprint(ctx: PricingContext) -> Optional[Tuple[Band, bool]]:
    """[E190] (the per-core footprint over the declared cores, whether it was scaled), or None
    with the area refused by name outside the footprint's validity."""
    value = ctx.physics.band("area_per_core_total")
    cores = ctx.quantities.get("cores_physical").value
    inside = (scale_band(value, cores * _M2_TO_MM2, "area_per_core_total x cores_physical"), False)
    if ctx.geometry is None:
        return inside
    bits = ctx.quantities.get("weight_bits").value if ctx.quantities.has("weight_bits") else None
    scales, reason = aggregate_scales(ctx.physics, ctx.geometry, None if bits is None else int(bits))
    if scales is None:
        ctx.refuse(AREA_HEADLINE, f"area_per_core_total lies outside its validity: {reason}")
        return None
    if all(scale == 1.0 for scale in scales):
        return inside
    cells = footprint_cores(ctx.geometry, scales)
    return scale_band(value, cells * _M2_TO_MM2,
                      "area_per_core_total x cores_physical, scaled by core cells (declared scaling rule)"), True


def price_area(ctx: PricingContext) -> None:
    """Chip area: the published per-core footprint if there is one, else components."""
    components: List[Band] = []
    evidence: List[str] = []
    unpriced: List[str] = []
    if ctx.priced("area_per_core_total"):
        if not ctx.quantities.has("cores_physical"):
            ctx.refuse(AREA_HEADLINE, "area_per_core_total is declared but the "
                                      "cores_physical census is absent")
            return
        footprint = _aggregate_footprint(ctx)
        if footprint is None:
            return
        band, scaled = footprint
        components.append(band)
        evidence.append(f"cores: area_per_core_total [{ctx.physics.band('area_per_core_total').basis}]"
                        + (" scaled by core cells (declared scaling rule)" if scaled else "")
                        + mean_core_note(ctx.physics, ctx.geometry))
    elif ctx.priced("area_per_cell"):
        bands, notes, missing = ctx.component_bands(AREA_COMPONENTS, _M2_TO_MM2)
        if not bands:
            ctx.refuse(AREA_HEADLINE, "area_per_cell is declared but the "
                                      "cells_physical census is absent")
            return
        state = _state_area_band(ctx)
        if state is not None:
            bands.append(state)
        components.extend(bands)
        evidence.extend(notes)
        unpriced.extend(missing)
    else:
        ctx.refuse(AREA_HEADLINE, "no compute-area source is declared: the target "
                                  "declares neither area_per_core_total nor "
                                  "area_per_cell")
        return
    # Beside the footprint on either path, and named unpriced where a census shows the work.
    bands, notes, missing = ctx.component_bands(CHIP_AREA_ADDERS, _M2_TO_MM2)
    components.extend(bands)
    evidence.extend(notes)
    unpriced.extend(missing)
    if ctx.physics.has("area_global_fixed"):
        value = ctx.physics.band("area_global_fixed")
        components.append(scale_band(value, _M2_TO_MM2, "area_global_fixed"))
        evidence.append(f"global: area_global_fixed [{value.basis}]")
    source = "; ".join(evidence)
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(AREA_HEADLINE, "mm^2", add_bands(components, source), source)


def _programming_latency(ctx: PricingContext) -> None:
    """The per-program-load wall — an overhead term, never the steady-state headline."""
    if not ctx.priced("t_program_per_byte"):
        return
    payload = programming_payload_bytes(ctx.quantities, ctx.physics)
    if payload is None:
        return
    payload_bytes, note = payload
    value = ctx.physics.band("t_program_per_byte")
    source = f"t_program_per_byte [{value.basis}] x {note}"
    ctx.term("latency_programming_s", "s",
             scale_band(value, payload_bytes, source), source)


def _host_unpriced(host) -> List[str]:
    return list(getattr(host, "unpriced", ()))


def price_latency(ctx: PricingContext, writes: Optional[WritePricing] = None) -> Optional[Band]:
    """Steady-state end-to-end latency per sample; returns the band energy reuses.

    ``t_hop`` is deliberately NOT added: execution is timestep-synchronous, so hop
    time is already inside the timestep the cycle constant prices — the same
    no-double-count discipline the measured ``sim_time_s`` carries.
    """
    if not ctx.priced("t_cycle"):
        ctx.refuse(LATENCY_HEADLINE, "t_cycle is undeclared — the one constant that "
                                     "converts latency_steps to seconds")
        _programming_latency(ctx)
        return None
    if not ctx.quantities.has("latency_steps"):
        ctx.refuse(LATENCY_HEADLINE, "t_cycle is declared but the latency_steps "
                                     "census is absent")
        _programming_latency(ctx)
        return None
    t_cycle = ctx.physics.band("t_cycle")
    steps = ctx.quantities.get("latency_steps").value
    components = [scale_band(t_cycle, steps, "t_cycle x latency_steps")]
    evidence = [f"compute: t_cycle [{t_cycle.basis}]"]

    bands, notes, unpriced = ctx.component_bands(LATENCY_COMPONENTS, 1.0)
    components.extend(bands)
    evidence.extend(notes)

    host = ctx.host_time()
    if host.unpriceable:
        ctx.refuse(LATENCY_HEADLINE, ctx.host_refusal(host))
        _programming_latency(ctx)
        return None
    if host.band is not None:
        ctx.term("latency_host_s", "s", host.band, host.band.basis)
        components.append(host.band)
        evidence.append("host: " + host.band.basis)
    unpriced.extend(_host_unpriced(host))
    if writes is not None:
        if writes.exposed is None:
            reason = next(r.reason for r in writes.refusals if r.name == "latency_write_exposed_s")
            ctx.refuse(LATENCY_HEADLINE, f"the exposed write time is refused: {reason}")
            _programming_latency(ctx)
            return None
        components.append(writes.exposed)
        evidence.append("writes: " + writes.exposed.basis)

    source = "; ".join(evidence) + (
        "; steady-state per sample: resident weights, programming amortized in "
        "latency_programming_s"
    )
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    band = add_bands(components, source)
    ctx.term(LATENCY_HEADLINE, "s", band, source)
    _programming_latency(ctx)
    return band


def price_throughput(ctx: PricingContext, e2e: Optional[Band]) -> None:
    """Inverse steady-state latency (v1; pipelined overlap is a stated follow-up)."""
    if e2e is None:
        ctx.refuse(THROUGHPUT_HEADLINE, "throughput is the inverse of e2e_latency_s, "
                                        "which is unavailable")
        return
    source = "1 / e2e_latency_s (steady-state; corners flip on inversion)"
    ctx.term(THROUGHPUT_HEADLINE, "inferences/s", invert_band(e2e, source), source)


def price_absolute(
    quantities: Quantities, physics: PlatformPhysics, *,
    writes: Optional[WriteInputs] = None, geometry: Any = None,
) -> AbsolutePricing:
    """Every absolute term the declared physics can honestly back, plus refusals.

    ``writes`` (a period census with the declared write policy, present only when the run declares
    ``inference_batch_K``) adds the write law's terms to the steady-state headlines; ``geometry``
    (the declared core types) judges the aggregate constants against the profile's validity. A
    caller that states none is judged by what the census proves (``validity.census_geometry``): a single
    core's type, or its cell count, which a declared 'cells' scaling rule reads exactly; at a point
    validity a mean core at the point is judged there with that basis stated in the term, and refused
    by name under ``validity.CENSUS_PROVED``. ``validity.UNJUDGED`` reads the aggregates as declared. A profile
    whose external component is undeclared prices nothing and refuses by name, the write law's terms each by its own
    name too (the write law's own gate, [fix_c]), so every plane states the same refusal.
    """
    if refusals := external_refusals(physics):
        named = () if writes is None else tuple(price_write_inputs(writes, physics).refusals)
        return AbsolutePricing(terms=(), refusals=refusals + named)
    ctx = PricingContext.create(quantities, physics, judged_geometry(geometry, quantities))
    priced_writes = None if writes is None else price_write_inputs(writes, physics)
    price_area(ctx)
    e2e = price_latency(ctx, priced_writes)
    price_energy(ctx, e2e, priced_writes)
    price_throughput(ctx, e2e)
    terms, refusals = list(ctx.terms), list(ctx.refusals)
    if priced_writes is not None:
        terms += priced_writes.terms
        refusals += priced_writes.refusals
    return AbsolutePricing(terms=tuple(terms), refusals=tuple(refusals),
                           verdicts=() if priced_writes is None else tuple(priced_writes.verdicts))
