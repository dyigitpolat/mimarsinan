"""The energy metric: compute + periphery + static + host, or a written refusal."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.context import (
    HostTime,
    PricingContext,
)
from mimarsinan.deployment_record.cost.absolute.formulas import (
    BUFFER_STATIC,
    ENERGY_COMPONENTS,
    PROGRAMMING_BYTE_ENERGIES,
    add_bands,
    carry_buffer_static_watts,
    per_core_type_compute,
    programming_payload_bytes,
    scale_band,
)
from mimarsinan.deployment_record.cost.host_terms import host_side_energy
from mimarsinan.deployment_record.cost.absolute.validity import (
    CensusCells,
    aggregate_scales,
    mean_core_note,
)
from mimarsinan.deployment_record.schema.provenance import Band

_J_TO_MJ = 1e3

ENERGY_HEADLINE = "energy_per_inference_mj"
ENERGY_DYNAMIC = "energy_dynamic_mj"
#: [W08] The chip's own energy (dynamic, static and write terms, no host), priced beside the headline wherever a
#: point prices a lane block (``lanes_per_tile`` present), so a host-bound verdict shows what the chip spends.
ENERGY_CHIP_ONLY = "energy_chip_only_mj"


def _prices_lane_block(ctx: PricingContext) -> bool:
    return ctx.quantities.has("lanes_per_tile") or "lanes_per_tile" in ctx.quantities.withheld


def _refuse(ctx: PricingContext, reason: str) -> None:
    """The headline refused, and with it (for a lane-block point) the chip-only term, by the same reason."""
    ctx.refuse(ENERGY_HEADLINE, reason)
    if _prices_lane_block(ctx):
        ctx.refuse(ENERGY_CHIP_ONLY, reason)


def _chip_only(ctx: PricingContext, components: List[Band], evidence: List[str], writes: Optional[Any],
               unpriced: List[str]) -> None:
    """The chip's own sum, naming every chip component of it the target cannot price (``unpriced``: the dynamic
    sum's, never a host item), as the dynamic and headline terms do. A declared write whose energy is refused refuses
    the term by that reason on every branch (the host-priced branch refuses it in ``_refuse``)."""
    if not _prices_lane_block(ctx):
        return
    if writes is not None and writes.energy is None:
        reason = next(r.reason for r in writes.refusals if r.name == "energy_write_mj")
        ctx.refuse(ENERGY_CHIP_ONLY, f"the write energy is refused: {reason}")
        return
    parts, sources = list(components), list(evidence)
    if writes is not None and writes.energy is not None:
        parts.append(writes.energy)
        sources.append("writes: " + writes.energy.basis)
    source = "chip only (no host energy): " + "; ".join(sources)
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(ENERGY_CHIP_ONLY, "mJ", add_bands(parts, source), source)


def _compute_constant(ctx: PricingContext) -> Optional[str]:
    """The aggregate if the target published one, else the per-MAC array constant."""
    for constant in ("e_synaptic_event_total", "e_mac"):
        if ctx.priced(constant):
            return constant
    return None


def _static_band(
    ctx: PricingContext, e2e: Optional[Band]
) -> Tuple[Optional[Band], bool]:
    """(static energy over the priced latency, declared-but-unpriceable). [U-F4] A declared on-chip carry buffer
    leaks beside the chip's own static power, on either path (no aggregate contains a component the run adds)."""
    per_core = ctx.priced("p_static_per_core") and ctx.quantities.has("cores_physical")
    global_ = ctx.priced("p_static_global")
    buffer = carry_buffer_static_watts(ctx.quantities, ctx.physics)
    if not per_core and not global_ and buffer is None:
        return None, False
    if e2e is None:
        return None, True
    watts: List[Band] = []
    if per_core:
        watts.append(scale_band(
            ctx.physics.band("p_static_per_core"),
            ctx.quantities.get("cores_physical").value,
            "p_static_per_core x cores_physical"))
    if global_:
        watts.append(ctx.physics.band("p_static_global"))
    if buffer is not None:
        watts.append(buffer)
    total = add_bands(watts, "static power")
    basis = "static power x e2e_latency_s" + ("" if buffer is None else f" (with {buffer.basis})")
    return Band(total.low * e2e.low * _J_TO_MJ,
                total.nominal * e2e.nominal * _J_TO_MJ,
                total.high * e2e.high * _J_TO_MJ, basis), False


def _aggregate_outside(ctx: PricingContext, compute: str) -> Optional[str]:
    """[E190] Why the per-event aggregate does not hold for this chip, or None. It is a point model: judged
    at each declared core type (or the census's mean core), never through a cells rule, whose scale can be
    1 off the point (two 128 x 512 cores hold two 256 x 256 cores' cells)."""
    if compute != "e_synaptic_event_total" or ctx.geometry is None:
        return None
    bits = ctx.quantities.get("weight_bits").value if ctx.quantities.has("weight_bits") else None
    scales, reason = aggregate_scales(ctx.physics, ctx.geometry, None if bits is None else int(bits))
    point = ctx.physics.validity.geometry
    if scales is None or point is None:
        return reason
    chip = ctx.geometry
    if isinstance(chip, CensusCells):
        if chip.mean is None or chip.strict:
            return (f"a per-event aggregate is a point model, and the census does not prove one core type "
                    f"({chip.reason}): pass the declared core types")
        cores = [chip.mean]
    else:
        cores = [(core.axons, core.neurons) for core in chip]
    if any(core != (point.axons, point.neurons) for core in cores):
        return "a per-event aggregate is a point model; no scaling rule reads it at another core"
    return None


def _host_band(ctx: PricingContext, e2e: Optional[Band]) -> Tuple[Optional[Band], Optional[str], List[str]]:
    """(host energy, refusal reason, unpriced) — p_host over the host wall; [U-J] on an edge host also its idle power
    over the wait and the link per crossing byte (``host_terms.host_side_energy``)."""
    host = ctx.host_time()
    missing = list(host.missing)
    if not ctx.physics.has("p_host") and (host.band is not None or missing):
        missing = ["p_host"] + missing
    if host.blocked is not None or missing:
        return None, ctx.host_refusal(HostTime(blocked=host.blocked, missing=tuple(missing))), []
    active = None
    if host.band is not None:
        p_host = ctx.physics.band("p_host")
        active = Band(p_host.low * host.band.low * _J_TO_MJ, p_host.nominal * host.band.nominal * _J_TO_MJ,
                      p_host.high * host.band.high * _J_TO_MJ,
                      f"p_host [{p_host.basis}] x host time ({host.band.basis})")
    band, unpriced = host_side_energy(ctx, active, host.band, e2e)
    return band, None, unpriced


def _programming_energy(ctx: PricingContext) -> None:
    """Per program LOAD, amortized separately from the per-inference headline. Like the per-inference
    terms it names every component of its own sum it cannot price: a row the target never declares
    while its census is positive (a payload, reprogrammed cores)."""
    components: List[Band] = []
    evidence: List[str] = []
    unpriced: List[str] = []
    payload = programming_payload_bytes(ctx.quantities, ctx.physics)
    if payload is not None:
        payload_bytes, note = payload
        for constant, label in PROGRAMMING_BYTE_ENERGIES:
            if not ctx.priced(constant):
                if payload_bytes and not ctx.physics.has(constant):
                    unpriced.append(f"{constant}({label})")
                continue
            value = ctx.physics.band(constant)
            components.append(scale_band(
                value, payload_bytes * _J_TO_MJ, f"{constant} x {note}"))
            evidence.append(f"{label}: {constant} [{value.basis}]")
    # [E4] Only the PROGRAMMING overhead is amortized here. Core INIT is not:
    # every pass resets its cores' neuron state, so it is paid per inference —
    # which is where the latency plane has always charged t_core_init.
    constant, quantity = "e_core_program", "reprogrammed_cores"
    if ctx.priced(constant) and ctx.quantities.has(quantity):
        value = ctx.physics.band(constant)
        components.append(scale_band(
            value, ctx.quantities.get(quantity).value * _J_TO_MJ,
            f"{constant} x {quantity}"))
        evidence.append(f"{constant} [{value.basis}]")
    elif not ctx.physics.has(constant) and ctx.quantities.has(quantity) and ctx.quantities.get(quantity).value:
        unpriced.append(f"{constant}(core_program)")
    elif ctx.priced(constant) and (withheld := ctx.quantities.withheld_reason(quantity)) is not None:
        unpriced.append(f"{constant}(core_program; {withheld})")  # [U-C C-6] the cores to charge are not determined
        if not components:  # the declared per-core charge is the term's only part: refused by name, never silent
            others = unpriced[:-1]
            ctx.refuse("energy_programming_mj", f"{constant} is declared but its multiplicand {quantity} is withheld: "
                       f"{withheld}" + (f"; also unpriced: {', '.join(others)}" if others else ""))
    if components:
        source = "; ".join(evidence) + (
            "; per program load, amortized separately from the per-inference headline"
        ) + (f"; unpriced: {', '.join(unpriced)}" if unpriced else "")
        ctx.term("energy_programming_mj", "mJ", add_bands(components, source), source)


def price_energy(ctx: PricingContext, e2e: Optional[Band], writes: Optional[Any] = None,
                 split: Optional[Any] = None) -> None:
    """Energy per inference, or a refusal naming the missing constant or census. [E-6] A per-core-type MAC split
    (``split``) prices the compute term per core type."""
    compute = _compute_constant(ctx)
    if compute is None:
        _refuse(ctx, "no compute-energy source is declared: the "
                     "target declares neither e_synaptic_event_total "
                     "nor e_mac")
        _programming_energy(ctx)
        return
    outside = _aggregate_outside(ctx, compute)
    if outside is not None:
        _refuse(ctx, f"e_synaptic_event_total lies outside its validity: {outside}")
        _programming_energy(ctx)
        return
    if not ctx.quantities.has("synaptic_events"):
        _refuse(ctx, f"{compute} is declared but the synaptic_events "
                     "census is absent (at candidate time it needs "
                     "the declared activity_factor; no record seals "
                     "an event census yet)")
        _programming_energy(ctx)
        return
    if split is not None:
        per_type, evidence, reason = per_core_type_compute(
            split, ctx.physics, compute, ctx.quantities.get("synaptic_events").value, _J_TO_MJ)
        if per_type is None:
            ctx.refuse(ENERGY_DYNAMIC, reason or "per-core-type compute physics refused")
            _refuse(ctx, reason or "per-core-type compute physics refused")
            _programming_energy(ctx)
            return
        components = list(per_type)
    else:
        value = ctx.physics.band(compute)
        events = ctx.quantities.get("synaptic_events").value
        components = [scale_band(value, events * _J_TO_MJ, f"{compute} x synaptic_events")]
        evidence = [f"compute: {compute} [{value.basis}]" + (
            mean_core_note(ctx.physics, ctx.geometry, point_model=True) if compute == "e_synaptic_event_total"
            else "")]

    bands, notes, unpriced = ctx.component_bands(ENERGY_COMPONENTS, _J_TO_MJ)
    blocked = [ctx.blocked[c] for c, _f, _l in ENERGY_COMPONENTS if c in ctx.blocked]
    if blocked:  # [E-5] a converter the run's resolution leaves unstated: no headline reads around it
        ctx.refuse(ENERGY_DYNAMIC, blocked[0])
        _refuse(ctx, blocked[0])
        _programming_energy(ctx)
        return
    components.extend(bands)
    evidence.extend(notes)

    # Switching energy stands on its own: it needs no latency, so a target that
    # cannot price static power still answers the number chip papers report. It
    # names every component of its own sum it cannot price, as the headline does.
    dynamic_source = "; ".join(evidence) + (
        "; DYNAMIC only — excludes static power and the host share"
    )
    if unpriced:
        dynamic_source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(ENERGY_DYNAMIC, "mJ", add_bands(components, dynamic_source),
             dynamic_source)

    if BUFFER_STATIC in ctx.refused_constants:  # [U-F4] the declared buffer leaks at a power the target never stated
        _refuse(ctx, ctx.refused_constants[BUFFER_STATIC])
        _programming_energy(ctx)
        return
    static, static_blocked = _static_band(ctx, e2e)
    if static_blocked:
        root = ctx.refusal_reason("e2e_latency_s") or "no reason was recorded"
        _refuse(ctx, "static power is declared but the priced latency "
                     f"it multiplies is unavailable, because {root}")
        _programming_energy(ctx)
        return
    if static is not None:
        ctx.term("energy_static_mj", "mJ", static, static.basis)
        components.append(static)
        evidence.append("static: " + static.basis)

    chip, chip_evidence, chip_unpriced = list(components), list(evidence), list(unpriced)
    host, host_reason, host_unpriced = _host_band(ctx, e2e)
    if host_reason is not None:
        _chip_only(ctx, chip, chip_evidence, writes, chip_unpriced)
        ctx.refuse(ENERGY_HEADLINE, host_reason)
        _programming_energy(ctx)
        return
    if host is not None:
        ctx.term("energy_host_mj", "mJ", host, host.basis)
        components.append(host)
        evidence.append("host: " + host.basis)
    unpriced.extend([f"p_host x {item}" for item in ctx.host_time().unpriced] + host_unpriced)
    if writes is not None:
        if writes.energy is None:
            reason = next(r.reason for r in writes.refusals if r.name == "energy_write_mj")
            _refuse(ctx, f"the write energy is refused: {reason}")
            _programming_energy(ctx)
            return
        components.append(writes.energy)
        evidence.append("writes: " + writes.energy.basis)
    _chip_only(ctx, chip, chip_evidence, writes, chip_unpriced)

    source = "; ".join(evidence)
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(ENERGY_HEADLINE, "mJ", add_bands(components, source), source)
    _programming_energy(ctx)
