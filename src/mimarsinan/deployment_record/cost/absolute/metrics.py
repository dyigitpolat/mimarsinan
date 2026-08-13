"""price_absolute: the target's physics × a view's quantities → absolute cost terms.

One pure engine for both completenesses (a candidate's static quantities | a sealed
record's measured ones). Every term = constant band × quantity with the evidence in
its source; a headline that cannot be answered honestly is REFUSED with a written
reason, never approximated silently.
"""

from __future__ import annotations

from typing import List, Optional

from mimarsinan.deployment_record.cost.absolute.context import (
    AbsolutePricing,
    PricingContext,
)
from mimarsinan.deployment_record.cost.absolute.energy import price_energy
from mimarsinan.deployment_record.cost.absolute.formulas import (
    AREA_COMPONENTS,
    LATENCY_COMPONENTS,
    add_bands,
    invert_band,
    programming_payload_bytes,
    scale_band,
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
        value = ctx.physics.band("area_per_core_total")
        cores = ctx.quantities.get("cores_physical").value
        components.append(scale_band(
            value, cores * _M2_TO_MM2, "area_per_core_total x cores_physical"))
        evidence.append(f"cores: area_per_core_total [{value.basis}]")
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


def price_latency(ctx: PricingContext) -> Optional[Band]:
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


def price_absolute(quantities: Quantities, physics: PlatformPhysics) -> AbsolutePricing:
    """Every absolute term the declared physics can honestly back, plus refusals."""
    ctx = PricingContext.create(quantities, physics)
    price_area(ctx)
    e2e = price_latency(ctx)
    price_energy(ctx, e2e)
    price_throughput(ctx, e2e)
    return AbsolutePricing(terms=tuple(ctx.terms), refusals=tuple(ctx.refusals))
