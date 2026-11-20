"""price_absolute: the target's physics × a view's quantities → absolute cost terms.

One pure engine for both completenesses (a candidate's static quantities | a sealed record's measured ones). Every
term = constant band × quantity with the evidence in its source; a headline that cannot be answered honestly is
REFUSED with a written reason, never approximated silently.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.context import (
    ABSOLUTE_PRICING as ABSOLUTE_PRICING, AbsolutePricing, PricingContext, PricingRefusal,
    external_refusals as external_refusals)
from mimarsinan.deployment_record.cost.absolute.energy import ENERGY_HEADLINE, price_energy
from mimarsinan.deployment_record.cost.absolute.formulas import (
    AREA_COMPONENTS, BUFFER_CAPACITY, CHIP_AREA_ADDERS, LATENCY_COMPONENTS, add_bands, carry_buffer_refusals,
    invert_band, scale_band)
from mimarsinan.deployment_record.cost.absolute.walls.loads import load_refusals, price_loads
from mimarsinan.deployment_record.cost.absolute.walls.static_power import static_sum
from mimarsinan.deployment_record.cost.absolute.validity import (
    aggregate_scales, footprint_cores, judged_geometry, mean_core_note)
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, WritePricing, price_write_inputs
from mimarsinan.deployment_record.cost.compute_time import STEP_LAW, compute_law, compute_time
from mimarsinan.deployment_record.cost.host_terms import (
    LATENCY_LINK, host_link_time, price_system_energy, store_refusal, write_link_bytes)
from mimarsinan.deployment_record.cost.terms import PricingFlag
from mimarsinan.deployment_record.platform_physics.area_components import fixed_area_refusal
from mimarsinan.deployment_record.platform_physics.area_rule.pricing import rule_area
from mimarsinan.deployment_record.platform_physics.converter import resolve as resolve_converter
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.batch_rounds import SPAN_CORES, SPAN_KEYS, SPAN_STEPS, SPAN_SYNCS
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.schema.provenance import Band

_M2_TO_MM2 = 1e6

AREA_HEADLINE = "chip_area_mm2"
LATENCY_HEADLINE = "e2e_latency_s"
THROUGHPUT_HEADLINE = "throughput_inferences_s"
GATED_HEADLINES = (AREA_HEADLINE, ENERGY_HEADLINE, LATENCY_HEADLINE, THROUGHPUT_HEADLINE)
#: [U-ENERGY] The priced times that are summands of ``e2e_latency_s``, each with the label its source gives it: the
#: static leg over e2e covers them, so none takes a leg of its own (the runtime ratchet reads this table).
E2E_SUMMANDS: Tuple[Tuple[str, str], ...] = (("latency_host_s", "host: "), (LATENCY_LINK, "host link: "),
                                              ("latency_write_exposed_s", "writes: "))
_E2E_LABEL = dict(E2E_SUMMANDS)


def _state_area_band(ctx: PricingContext) -> Optional[Band]:
    """State storage = area_per_state_bit × membrane_bits × neurons_physical."""
    if not (ctx.priced("area_per_state_bit") and ctx.priced("membrane_bits")
            and ctx.quantities.has("neurons_physical")):
        return None
    per_bit = ctx.physics.band("area_per_state_bit")
    bits = ctx.physics.band("membrane_bits")
    return scale_band(Band(per_bit.low * bits.low, per_bit.nominal * bits.nominal, per_bit.high * bits.high,
                           "area_per_state_bit x membrane_bits"), ctx.quantities.get("neurons_physical").value
                      * _M2_TO_MM2, "area_per_state_bit x membrane_bits x neurons_physical")


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


def price_area(ctx: PricingContext, readout_bits: Optional[int] = None) -> Tuple[PricingFlag, ...]:
    """Chip area, with the flags it names: [U-AREA] a description's ``area_rule`` block prices its core grid and outside
    part by the rule (``area_rule.pricing``, its kinds at ``readout_bits``); else the published per-core footprint if
    there is one, else components; the adders and, for declared parts, R5's outside part on every path."""
    components: List[Band] = []
    evidence: List[str] = []
    unpriced: List[str] = []
    rule = rule_area(ctx.physics, ctx.quantities, ctx.geometry, readout_bits)
    flags = () if rule is None else tuple(PricingFlag(AREA_HEADLINE, f.kind, f.reason, f.value) for f in rule.flags)
    if rule is not None and rule.refusal is not None:
        ctx.refuse(AREA_HEADLINE, rule.refusal)
        return flags
    if rule is not None and rule.band is not None:
        components.append(Band(*rule.band, rule.basis))
        evidence.append(rule.basis)
    elif ctx.priced("area_per_core_total"):
        if not ctx.quantities.has("cores_physical"):
            ctx.refuse(AREA_HEADLINE, "area_per_core_total is declared but the "
                                      "cores_physical census is absent")
            return flags
        footprint = _aggregate_footprint(ctx)
        if footprint is None:
            return flags
        band, scaled = footprint
        components.append(band)
        evidence.append(f"cores: area_per_core_total [{ctx.physics.band('area_per_core_total').basis}]"
                        + (" scaled by core cells (declared scaling rule)" if scaled else "")
                        + mean_core_note(ctx.physics, ctx.geometry))
    elif ctx.priced("area_per_cell"):
        bands, notes, missing = ctx.component_bands(AREA_COMPONENTS, _M2_TO_MM2)
        blocked = [ctx.blocked[c] for c, _f, _l in AREA_COMPONENTS if c in ctx.blocked]
        if blocked:  # [E-5] a converter the run's resolution leaves unstated
            ctx.refuse(AREA_HEADLINE, blocked[0])
            return flags
        if not bands:
            ctx.refuse(AREA_HEADLINE, "area_per_cell is declared but the "
                                      "cells_physical census is absent")
            return flags
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
        return flags
    # Beside the footprint on either path, and named unpriced where a census shows the work.
    bands, notes, missing = ctx.component_bands(CHIP_AREA_ADDERS, _M2_TO_MM2)
    blocked = [ctx.blocked[c] for c, _f, _l in CHIP_AREA_ADDERS if c in ctx.blocked]
    if blocked:  # [U-F4] a declared component whose area the target never stated
        ctx.refuse(AREA_HEADLINE, blocked[0])
        return flags
    components.extend(bands)
    evidence.extend(notes)
    unpriced.extend(missing)
    if rule is None or rule.declared_parts:  # [U-A] the outside part beside the grid: R5's, for declared parts
        if refused := fixed_area_refusal(ctx.physics):  # [U-A] the chip components' sum, else a sourced scalar
            ctx.refuse(AREA_HEADLINE, refused)
            return flags
        value = ctx.physics.band("area_global_fixed")
        components.append(scale_band(value, _M2_TO_MM2, "area_global_fixed"))
        evidence.append(f"global: area_global_fixed [{value.basis}]")
    source = "; ".join(evidence) + (rule.basis if rule is not None and rule.band is None else "")
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(AREA_HEADLINE, "mm^2", add_bands(components, source), source)
    return flags


def _host_unpriced(host) -> List[str]:
    return list(getattr(host, "unpriced", ()))


def price_latency(ctx: PricingContext, writes: Optional[WritePricing] = None) -> Optional[Band]:
    """Steady-state end-to-end latency per sample; returns the band energy reuses.

    ``t_hop`` is deliberately NOT added: execution is timestep-synchronous, so hop
    time is already inside the timestep the cycle constant prices — the same
    no-double-count discipline the measured ``sim_time_s`` carries. [U-C2 G-2] The
    compute is ``cost.compute_time``'s: per timestep (``t_cycle``) or, on a chip
    without a fixed timestep, per synaptic operation on the event census.
    """
    compute = compute_time(ctx.physics, ctx.quantities, ctx.priced)  # [U-C2 G-2] per timestep or per event
    if compute.band is None:
        ctx.refuse(LATENCY_HEADLINE, compute.refusal or "the compute time is not priced")
        return None
    components = [compute.band]
    evidence = [compute.evidence]

    bands, notes, unpriced = ctx.component_bands(LATENCY_COMPONENTS, 1.0)
    components.extend(bands)
    evidence.extend(notes)

    host = ctx.host_time()
    if host.unpriceable:
        ctx.refuse(LATENCY_HEADLINE, ctx.host_refusal(host))
        return None
    chip_side = list(components)  # [U-J2] every part but the host's own time: the wait it idles over
    if host.band is not None:
        ctx.term("latency_host_s", "s", host.band, host.band.basis)
        components.append(host.band)
        evidence.append(_E2E_LABEL["latency_host_s"] + host.band.basis)
    unpriced.extend(_host_unpriced(host))
    link, link_unpriced = host_link_time(ctx)  # [U-J] the host-chip link: every crossing byte, every transfer
    if refused := store_refusal(ctx):  # [U-J3] store traffic with no store to price it on, refused by name
        ctx.refuse(LATENCY_HEADLINE, refused)
        return None
    if link is not None:
        ctx.term(LATENCY_LINK, "s", link, link.basis)
        components.append(link)
        chip_side.append(link)
        evidence.append(_E2E_LABEL[LATENCY_LINK] + link.basis)
    unpriced.extend(link_unpriced)
    if writes is not None:
        if writes.exposed is None:
            reason = next(r.reason for r in writes.refusals if r.name == "latency_write_exposed_s")
            ctx.refuse(LATENCY_HEADLINE, f"the exposed write time is refused: {reason}")
            return None
        components.append(writes.exposed)
        chip_side.append(writes.exposed)
        evidence.append(_E2E_LABEL["latency_write_exposed_s"] + writes.exposed.basis)

    source = "; ".join(evidence) + (
        "; steady-state per sample: resident weights, programming amortized in "
        "latency_programming_s"
    )
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    band = add_bands(components, source)
    ctx.term(LATENCY_HEADLINE, "s", band, source)
    ctx.host_wait = add_bands(chip_side, "e2e_latency_s less the host's own time: the chip, the link, exposed writes")
    return band


#: [thru:w3] An input's span under the batch pipeline: the barriers and core initializations of its passes.
SPAN_HEADLINE = "latency_input_span_s"
#: [thru:w3] The priced times that read passes ``e2e_latency_s`` prices per inference: e2e's static leg covers them, so
#: none takes a leg of its own and none is its summand (the runtime ratchet reads this table).
E2E_READINGS: Tuple[str, ...] = (SPAN_HEADLINE,)
SPAN_COMPONENTS = (("t_sync_barrier", (SPAN_SYNCS,), SPAN_SYNCS), ("t_core_init", (SPAN_CORES,), SPAN_CORES))


def price_input_span(ctx: PricingContext) -> None:
    """[thru:w3] Under the batch pipeline, an input's span from its first pass to its last (``batch_rounds``): the
    compute of the span's passes at ``t_cycle``, their barriers and core initializations, beside the headline (which
    stays the time per inference that the throughput inverts). Priced by the step law only; on a chip priced per
    synaptic operation, or with the span's steps withheld, it is refused by name. Absent without the batch pipeline."""
    q = ctx.quantities
    if not any(q.has(key) or q.withheld_reason(key) for key in SPAN_KEYS):
        return
    if compute_law(ctx.physics, ctx.priced) != STEP_LAW:
        ctx.refuse(SPAN_HEADLINE, "an input's span is priced per step (t_cycle x input_span_steps); this chip prices "
                                  "compute otherwise, and no plane states a span's events")
        return
    if not q.has(SPAN_STEPS):
        ctx.refuse(SPAN_HEADLINE, f"the span's steps are withheld: {q.withheld_reason(SPAN_STEPS)}")
        return
    cycle = ctx.physics.band(STEP_LAW)
    components = [scale_band(cycle, q.get(SPAN_STEPS).value, "t_cycle x input_span_steps")]
    bands, notes, unpriced = ctx.component_bands(SPAN_COMPONENTS, 1.0)
    source = "; ".join([f"compute: t_cycle [{cycle.basis}]"] + notes) + (
        "; an input's span under the batch pipeline: the passes from its first to its last, per input")
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(SPAN_HEADLINE, "s", add_bands(components + bands, source), source)


def price_throughput(ctx: PricingContext, e2e: Optional[Band]) -> None:
    """Inverse steady-state latency (v1; pipelined overlap is a stated follow-up)."""
    if e2e is None:
        ctx.refuse(THROUGHPUT_HEADLINE, "throughput is the inverse of e2e_latency_s, "
                                        "which is unavailable")
        return
    source = "1 / e2e_latency_s (steady-state; corners flip on inversion)"
    ctx.term(THROUGHPUT_HEADLINE, "inferences/s", invert_band(e2e, source), source)


def resolved_physics(quantities: Quantities, physics: PlatformPhysics,
                     readout_bits: Optional[int]) -> Tuple[PlatformPhysics, Mapping[str, str], Mapping[str, str]]:
    """(the physics a pricing reads, the converter's refused constants, the carry buffer's unstated ones): the
    converter at the run's bits and the carry buffer's table at the view's declared capacity, one rule for every
    pricer that reads the description (``price_absolute``, the measured model's static leg)."""
    physics, converter_refused = resolve_converter(physics, readout_bits)
    physics, buffer_unstated = physics.at_carry_buffer(
        int(quantities.get(BUFFER_CAPACITY).value) if quantities.has(BUFFER_CAPACITY) else None)
    return physics, converter_refused, buffer_unstated


def price_absolute(
    quantities: Quantities, physics: PlatformPhysics, *,
    writes: Optional[WriteInputs] = None, geometry: Any = None, compute: Any = None,
    readout_bits: Optional[int] = None, loads: Optional[Any] = None,
) -> AbsolutePricing:
    """Every absolute term the declared physics can honestly back, plus refusals.

    ``writes`` (a period census with the declared write policy, present only when the run declares
    ``inference_batch_K``) adds the write law's terms to the steady-state headlines; ``geometry`` (the declared core
    types) judges the aggregate constants against the profile's validity. A caller that states none is judged by what
    the census proves (``validity.census_geometry``): a single core's type, or its cell count, which a declared 'cells'
    scaling rule reads exactly; at a point validity a mean core at the point is judged there with that basis stated in
    the term, and refused by name under ``validity.CENSUS_PROVED``. ``validity.UNJUDGED`` reads the aggregates as
    declared. A profile whose external component is undeclared prices nothing and refuses by name, [U-CR C-2] each
    headline it suppresses too (``GATED_HEADLINES``, by the gate's reason) and the write law's terms each by its own
    name (the write law's own gate, [fix_c]), so every plane states the same refusal. [E-6] ``compute``
    (``quantities.compute_technology.compute_split``: the plane's MAC events per core type, present only when a core
    type names a compute physics) prices each core type's MAC events at its own technology's e_mac. [E-5]
    ``readout_bits`` (the run's ``readout_law.adc_bits``) prices a profile's converter at that resolution
    (``platform_physics.converter.resolve``); a converter constant its table does not state there refuses, by name, each
    headline whose converter work exists. A profile without a converter, or no declared bits, is unchanged. [U-G] A
    declared carry buffer (``carry_buffer_capacity_bytes``) prices a profile's carry-buffer table at that capacity
    (``PlatformPhysics.at_carry_buffer``); a buffer constant its table leaves unstated there refuses, by name, each
    headline whose buffer work exists. A profile without the table, or no declared buffer, is unchanged.
    [U-ENERGY] ``loads`` (``quantities.planes.loads.LoadInputs``: the plane's every-core and period census) is priced by
    the one load law (``walls.loads``): the deployment's first load and the reprogramming per round, each with the
    chip's static power over its own wall; without it every load figure refuses by name. The chip's one static sum is
    computed once here, on the resolved physics, and returned (``AbsolutePricing.static_power``)."""
    if refusals := external_refusals(physics):
        reason = f"{ABSOLUTE_PRICING}: " + "; ".join(r.reason for r in refusals)
        named = tuple(PricingRefusal(h, reason) for h in GATED_HEADLINES) + (
            () if writes is None else tuple(price_write_inputs(writes, physics).refusals))
        gated = tuple(PricingRefusal(name, why) for name, why in load_refusals(loads, physics))  # the law's own gate
        return AbsolutePricing(terms=(), refusals=refusals + named + gated)
    physics, converter_refused, buffer_unstated = resolved_physics(quantities, physics, readout_bits)
    ctx = PricingContext.create(quantities, physics, judged_geometry(geometry, quantities), refused_constants={
        **converter_refused, **carry_buffer_refusals(quantities, physics, unstated=buffer_unstated)})
    ctx.static = static_sum(ctx.quantities, ctx.physics, ctx.priced, ctx.refused_constants)  # [U-ENERGY] once
    priced_writes = None if writes is None else price_write_inputs(writes, physics)
    ctx.write_link_bytes = None if writes is None else write_link_bytes(writes)  # [U-J] priced by an edge host only
    flags = price_area(ctx, readout_bits)  # [U-AREA] the area rule's flags, computed at read
    e2e = price_latency(ctx, priced_writes)
    price_input_span(ctx)  # [thru:w3] beside the headline, under the batch pipeline only
    price_energy(ctx, e2e, priced_writes, compute)
    price_system_energy(ctx)  # [U-J2] the whole-system line beside the headline, where the host states one
    price_throughput(ctx, e2e)
    price_loads(ctx, loads)  # [U-ENERGY] the first load and the reprogramming per round, one law
    terms, refusals = list(ctx.terms), list(ctx.refusals)
    if priced_writes is not None:
        terms += priced_writes.terms
        refusals += priced_writes.refusals
    return AbsolutePricing(terms=tuple(terms), refusals=tuple(refusals), flags=flags,
                           verdicts=() if priced_writes is None else tuple(priced_writes.verdicts),
                           static_power=ctx.static)
