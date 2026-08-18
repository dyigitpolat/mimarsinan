"""The energy metric: compute + periphery + static + host, or a written refusal."""

from __future__ import annotations

from typing import List, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.context import (
    HostTime,
    PricingContext,
)
from mimarsinan.deployment_record.cost.absolute.formulas import (
    ENERGY_COMPONENTS,
    PROGRAMMING_BYTE_ENERGIES,
    add_bands,
    programming_payload_bytes,
    scale_band,
)
from mimarsinan.deployment_record.schema.provenance import Band

_J_TO_MJ = 1e3

ENERGY_HEADLINE = "energy_per_inference_mj"
ENERGY_DYNAMIC = "energy_dynamic_mj"


def _compute_constant(ctx: PricingContext) -> Optional[str]:
    """The aggregate if the target published one, else the per-MAC array constant."""
    for constant in ("e_synaptic_event_total", "e_mac"):
        if ctx.priced(constant):
            return constant
    return None


def _static_band(
    ctx: PricingContext, e2e: Optional[Band]
) -> Tuple[Optional[Band], bool]:
    """(static energy over the priced latency, declared-but-unpriceable)."""
    per_core = ctx.priced("p_static_per_core") and ctx.quantities.has("cores_physical")
    global_ = ctx.priced("p_static_global")
    if not per_core and not global_:
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
    total = add_bands(watts, "static power")
    basis = "static power x e2e_latency_s"
    return Band(total.low * e2e.low * _J_TO_MJ,
                total.nominal * e2e.nominal * _J_TO_MJ,
                total.high * e2e.high * _J_TO_MJ, basis), False


def _host_band(ctx: PricingContext) -> Tuple[Optional[Band], Optional[str]]:
    """(host energy, refusal reason) — p_host over the host wall."""
    host = ctx.host_time()
    missing = list(host.missing)
    if not ctx.physics.has("p_host") and (host.band is not None or missing):
        missing = ["p_host"] + missing
    if host.blocked is not None or missing:
        return None, ctx.host_refusal(HostTime(blocked=host.blocked,
                                               missing=tuple(missing)))
    if host.band is None:
        return None, None
    p_host = ctx.physics.band("p_host")
    basis = f"p_host [{p_host.basis}] x host time ({host.band.basis})"
    return Band(p_host.low * host.band.low * _J_TO_MJ,
                p_host.nominal * host.band.nominal * _J_TO_MJ,
                p_host.high * host.band.high * _J_TO_MJ, basis), None


def _programming_energy(ctx: PricingContext) -> None:
    """Per program LOAD, amortized separately from the per-inference headline."""
    components: List[Band] = []
    evidence: List[str] = []
    payload = programming_payload_bytes(ctx.quantities, ctx.physics)
    if payload is not None:
        payload_bytes, note = payload
        for constant, label in PROGRAMMING_BYTE_ENERGIES:
            if not ctx.priced(constant):
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
    if components:
        source = "; ".join(evidence) + (
            "; per program load, amortized separately from the per-inference headline"
        )
        ctx.term("energy_programming_mj", "mJ", add_bands(components, source), source)


def price_energy(ctx: PricingContext, e2e: Optional[Band]) -> None:
    """Energy per inference, or a refusal naming the missing constant or census."""
    compute = _compute_constant(ctx)
    if compute is None:
        ctx.refuse(ENERGY_HEADLINE, "no compute-energy source is declared: the "
                                    "target declares neither e_synaptic_event_total "
                                    "nor e_mac")
        _programming_energy(ctx)
        return
    if not ctx.quantities.has("synaptic_events"):
        ctx.refuse(ENERGY_HEADLINE, f"{compute} is declared but the synaptic_events "
                                    "census is absent (at candidate time it needs "
                                    "the declared activity_factor; no record seals "
                                    "an event census yet)")
        _programming_energy(ctx)
        return
    value = ctx.physics.band(compute)
    events = ctx.quantities.get("synaptic_events").value
    components = [scale_band(value, events * _J_TO_MJ, f"{compute} x synaptic_events")]
    evidence = [f"compute: {compute} [{value.basis}]"]

    bands, notes, unpriced = ctx.component_bands(ENERGY_COMPONENTS, _J_TO_MJ)
    components.extend(bands)
    evidence.extend(notes)

    # Switching energy stands on its own: it needs no latency, so a target that
    # cannot price static power still answers the number chip papers report.
    dynamic_source = "; ".join(evidence) + (
        "; DYNAMIC only — excludes static power and the host share"
    )
    ctx.term(ENERGY_DYNAMIC, "mJ", add_bands(components, dynamic_source),
             dynamic_source)

    static, static_blocked = _static_band(ctx, e2e)
    if static_blocked:
        root = ctx.refusal_reason("e2e_latency_s") or "no reason was recorded"
        ctx.refuse(ENERGY_HEADLINE, "static power is declared but the priced latency "
                                    f"it multiplies is unavailable, because {root}")
        _programming_energy(ctx)
        return
    if static is not None:
        ctx.term("energy_static_mj", "mJ", static, static.basis)
        components.append(static)
        evidence.append("static: " + static.basis)

    host, host_reason = _host_band(ctx)
    if host_reason is not None:
        ctx.refuse(ENERGY_HEADLINE, host_reason)
        _programming_energy(ctx)
        return
    if host is not None:
        ctx.term("energy_host_mj", "mJ", host, host.basis)
        components.append(host)
        evidence.append("host: " + host.basis)

    source = "; ".join(evidence)
    if unpriced:
        source += f"; unpriced: {', '.join(unpriced)}"
    ctx.term(ENERGY_HEADLINE, "mJ", add_bands(components, source), source)
    _programming_energy(ctx)
