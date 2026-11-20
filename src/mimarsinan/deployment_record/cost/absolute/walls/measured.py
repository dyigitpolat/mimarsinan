"""[U-ENERGY class 1] The measured model's static leg: the chip's static power over the record's whole latency.

The sum is the record's own: its quantities, its declared physics resolved by the pricer's one rule
(``metrics.resolved_physics``) and summed by the pricer's one function (``static_power.static_sum``) with the
pricing's refused constants, so the measured model and the record's absolute pricing charge the same static power (a
carry buffer's tabled row included) and refuse the same parts. [V-M1] The leg is the one leg form's
(``static_power.static_leg``): where a declared part is refused, or the sum prices and the whole latency is refused,
the model's ``total_mj`` is withheld with the leg's reason (``measured_total``), never a number without its leg."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import mimarsinan.mapping.platform.target_bundles as target_bundles
from mimarsinan.deployment_record.cost.absolute.context import external_refusals
from mimarsinan.deployment_record.cost.absolute.formulas import carry_buffer_refusals
from mimarsinan.deployment_record.cost.absolute.metrics import resolved_physics
from mimarsinan.deployment_record.cost.absolute.walls.static_power import STATIC_PARTS, static_leg, static_sum
from mimarsinan.deployment_record.cost.combine import band_of, derived, flat_band, modeled, sum_bands
from mimarsinan.deployment_record.cost.terms import CostTerm, find_term_or_none
from mimarsinan.deployment_record.platform_physics.constants import resolve_supersessions
from mimarsinan.deployment_record.platform_physics.converter import declared_adc_bits
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema.provenance import Band


def measured_static_leg(record: Any, wall: Callable[[], Band]) -> Tuple[Optional[Band], str, str]:
    """(static power x ``wall`` in mJ, or None with why, ``"; no static energy: ..."``, and why the total is withheld,
    "" where it is not): ``wall`` is the measured model's ``total_s``, read only where the record's sum prices."""
    platform = record.identity.platform
    if (refusal := target_bundles.absolute_pricing_refusal(platform)) is not None:
        return None, f"; no static energy: {refusal[1]}", ""
    payload = platform.get("platform_physics_resolved")
    if payload is None:
        return None, "; no static energy: the record declares no physics", ""
    physics = PlatformPhysics.from_dict(payload)
    if gate := "; ".join(r.reason for r in external_refusals(physics)):  # the pricing refuses every declared part
        declared = any(physics.has(part) for part in STATIC_PARTS)
        return None, f"; no static energy: {gate}", gate if declared else ""
    quantities = from_record(record)
    physics, converter, buffer = resolved_physics(quantities, physics, declared_adc_bits(platform))
    refused = {**converter, **carry_buffer_refusals(quantities, physics, unstated=buffer)}
    static = static_sum(quantities, physics, frozenset(resolve_supersessions(physics.constants)).__contains__, refused)
    total, why = None, ""
    if static.parts and not static.refused:
        try:
            total = wall()
        except ValueError as exc:  # the measured model's own refusal of its latency, by name
            why = str(exc)
    form = static_leg(static, total, "total_s", why)
    if form.band is not None:
        return form.band, "", ""
    if form.refused:
        return None, f"; no static energy: {form.refused}", form.refused
    return None, "; no static energy: the record's physics states no static power" + (
        f" ({'; '.join(form.named)})" if form.named else ""), ""


def total_wall(latency: Sequence[CostTerm]) -> Band:
    """[composition: U-HOST, V-M1] The measured model's whole latency (``total_s``), its withholding raised as the
    wall's refusal, so the leg and ``total_mj`` are refused by it."""
    term = find_term_or_none(latency, "total_s")
    if term is None:
        raise ValueError("the measured model withholds total_s (its host work is unpriced)")
    return band_of(term)


def measured_total(record: Any, wall: Callable[[], Band], parts: Sequence[CostTerm], measured_mj: float,
                   basis: str) -> Tuple[Tuple[CostTerm, ...], str]:
    """(the modeled parts, the static leg's term and ``total_mj``; or the parts alone with why ``total_mj`` is
    withheld): the measured total plus every modeled part plus the chip's static power over ``total_s``."""
    static, why, withheld = measured_static_leg(record, wall)
    if withheld:
        return tuple(parts), withheld
    terms = tuple(parts) + (() if static is None else (modeled("modeled_static_mj", "mJ", static,
                                                               "static power x total_s"),))
    total = sum_bands([band_of(term) for term in terms] + [flat_band(measured_mj, basis)], basis=basis + why)
    return terms + (derived("total_mj", "mJ", total.nominal, "measured_total_mj + modeled programming/core-init/sync"
                            + (" + static over total_s" if why == "" else why), band=total),), ""
