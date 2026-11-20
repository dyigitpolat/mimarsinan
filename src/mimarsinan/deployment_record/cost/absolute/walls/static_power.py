"""[U-ENERGY class 1] The chip's one static sum and its one leg form over a priced chip time.

The sum is computed once per pricing, from the physics the pricer resolved (the carry buffer's table at the declared
capacity, the converter, supersession): per core x the declared cores, global, and a declared carry buffer's. Every
part the pricing refuses is carried with its reason (``refused``: the carry buffer's static row its table leaves
unstated at the declared capacity, or a declared capacity over an undeclared row). A declared static row the view
cannot multiply is named with its state's reason (the carry buffer's row over a capacity the platform's declarations
make zero prints that zero's reason in the sum's basis); a chip that declares no static row names the leg over every
wall; a static power held inside the per-event aggregate is named over every wall but the per-inference latency, whose
events it covers.

[V-M1] ONE leg form multiplies the sum by a wall (``static_leg``), and it is the only reader of the sum's power: the
per-inference headline, the deployment's first load and the reprogramming per round (``walls.loads``), the measured
model (``walls.measured``) and the step price read their legs from it. Where a declared part is refused, or the sum
prices and the wall is refused, the leg is refused with that reason, and so is every figure over the wall; a ratio of
two costs reads what each leg holds (``StaticSum.legs_over``), never the power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.formulas import (
    BUFFER_CAPACITY, BUFFER_STATIC, add_bands, carry_buffer_static_watts, scale_band)
from mimarsinan.deployment_record.cost.absolute.tables.components import absent_reason
from mimarsinan.deployment_record.schema.provenance import Band

_J_TO_MJ = 1e3
#: The chip's own static rows (never the host's).
STATIC_ROWS = ("p_static_per_core", "p_static_global")
#: Every static part the sum may hold, each refusable by the pricing's table of refused constants.
STATIC_PARTS = STATIC_ROWS + (BUFFER_STATIC,)
#: The per-event aggregate that may hold a chip's static power over its events.
AGGREGATE = "e_synaptic_event_total"
#: The wall the per-event aggregate's events cover.
HEADLINE_WALL = "e2e_latency_s"


@dataclass(frozen=True)
class StaticSum:
    """The chip's static power (W, corner by corner) or None, with what it cannot price named and what it refuses."""

    #: The power, read only by the one leg form (``static_leg``).
    watts: Optional[Band] = None
    #: Declared static rows the view cannot multiply, each with its state's reason.
    named: Tuple[str, ...] = ()
    #: The chip declares a static row.
    declared: bool = False
    #: The chip's static power is held in the per-event aggregate (priced over its events only).
    in_aggregate: bool = False
    #: [V-M1] Declared static parts the pricing refuses, each (part, reason): every leg over every wall refuses.
    refused: Tuple[Tuple[str, str], ...] = ()
    #: The parts ``watts`` sums, by name.
    parts: Tuple[str, ...] = ()

    def unpriced_over(self, wall: str) -> List[str]:
        """What the static leg over ``wall`` leaves unpriced, by name."""
        out = list(self.named)
        if self.in_aggregate:
            if wall != HEADLINE_WALL:
                out.append(f"static power(held in {AGGREGATE} over its synaptic events only; not priced over {wall})")
        elif not self.declared:
            out.append(f"{', '.join(STATIC_ROWS)}(static power over {wall}; undeclared)")
        return out

    def leg_refusal(self, wall: str) -> str:
        """Why no static leg over ``wall`` is priced where the sum prices nothing: its unpriced parts, by name."""
        return "no static leg: " + "; ".join(self.unpriced_over(wall) or [f"no static power over {wall}"])

    def legs_over(self, wall: str) -> Tuple[str, ...]:
        """What a cost over ``wall`` holds of the chip's static power: the summed parts, and the per-event aggregate's
        static power over the per-inference wall, whose events it covers (two costs hold like legs where equal)."""
        return self.parts + ((f"static power held in {AGGREGATE}",) if self.in_aggregate and wall == HEADLINE_WALL
                             else ())


#: No static power and nothing named: a caller with no pricing context states it by name.
NO_STATIC = StaticSum()


@dataclass(frozen=True)
class StaticLeg:
    """The chip's static power over one wall (mJ, corner by corner), or why every figure over the wall refuses."""

    band: Optional[Band] = None
    #: Non-empty: the leg is refused (a declared part, or the wall under a priced sum), and so is every figure over it.
    refused: str = ""
    #: What the leg leaves unpriced, by name (the four states' reasons).
    named: Tuple[str, ...] = ()
    #: The sum's basis (the parts it holds, as printed).
    power_basis: str = ""


def static_sum(quantities: Any, physics: Any, priced: Callable[[str], bool],
               refused_constants: Optional[Mapping[str, str]] = None) -> StaticSum:
    """The one static sum on a RESOLVED physics (``price_absolute`` calls it once per pricing); ``refused_constants``:
    the pricing's table of constants its declarations refuse (``PricingContext.refused_constants``)."""
    watts: List[Band] = []
    named: List[str] = []
    parts: List[str] = []
    if priced("p_static_per_core"):
        if quantities.has("cores_physical"):
            watts.append(scale_band(physics.band("p_static_per_core"), quantities.get("cores_physical").value,
                                    "p_static_per_core x cores_physical"))
            parts.append("p_static_per_core x cores_physical")
        else:  # [U-ENERGY class 2] never dropped silently: the state's reason
            named.append(f"p_static_per_core(static; {absent_reason(quantities, 'cores_physical')})")
    if priced("p_static_global"):
        watts.append(physics.band("p_static_global"))
        parts.append("p_static_global")
    buffer = carry_buffer_static_watts(quantities, physics)
    zero = ""
    if buffer is not None:
        watts.append(buffer)
        parts.append(buffer.basis)
    elif priced(BUFFER_STATIC) and not quantities.has(BUFFER_CAPACITY):  # [U-ENERGY class 2] never a silent zero
        if (declared := quantities.known_zero_reason(BUFFER_CAPACITY)) is not None:
            zero = f" (zero by declaration: {BUFFER_STATIC} ({declared}))"
        else:
            named.append(f"{BUFFER_STATIC}(static; {absent_reason(quantities, BUFFER_CAPACITY)})")
    band = None
    if watts:
        band = add_bands(watts, "static power" + ("" if buffer is None else f" (with {buffer.basis})") + zero)
    table = refused_constants or {}
    refused = tuple((part, table[part]) for part in STATIC_PARTS if part in table)
    return StaticSum(watts=band, named=tuple(named), declared=any(physics.has(r) for r in STATIC_ROWS),
                     in_aggregate=priced(AGGREGATE), refused=refused, parts=tuple(parts))


def static_leg(static: StaticSum, wall: Optional[Band], wall_name: str, wall_refused: str = "") -> StaticLeg:
    """[V-M1] The one leg form: the sum over ``wall`` (None: refused, ``wall_refused`` its reason). A declared part
    the pricing refuses refuses the leg by its reason; a priced sum over a refused wall refuses it by the wall's; a
    sum that prices nothing leaves no leg and names what it leaves unpriced."""
    if static.refused:
        return StaticLeg(refused="; ".join(reason for _part, reason in static.refused))
    watts = static.watts
    if watts is None:
        return StaticLeg(named=tuple(static.unpriced_over(wall_name)))
    if wall is None:
        return StaticLeg(refused=(f"static power is declared but the wall it multiplies ({wall_name}) is refused, "
                                  f"because {wall_refused or 'no reason was recorded'}"), power_basis=watts.basis)
    band = Band(watts.low * wall.low * _J_TO_MJ, watts.nominal * wall.nominal * _J_TO_MJ,
                watts.high * wall.high * _J_TO_MJ, f"{watts.basis} x {wall_name}")
    return StaticLeg(band=band, named=tuple(static.unpriced_over(wall_name)), power_basis=watts.basis)
