"""[U-AREA] The area rule's equations (design revision 3, section 2.1), one rule for every technology type:

    a_cell = kappa(kind) x sigma(n)  (SRAM; compute cells k_kind x k_L x sigma)  |  f x n^2 (1T1R)  |  a_PCM
    A_mem  = (B x a_cell + B_state x sigma(n)) / eta(type)
    A_conv = N_conv x a_conv(kind, b)                 the kind declared, else the type's default (rule E)
    A_core = (A_mem + A_conv) / s(type)               a type rule; an anchor scales its published core (anchor.py)
    A_chip = N x A_core / gamma + A_fixed             the outside part by the block's ``outside`` field (M2)

Every parameter is a registry value (registry.py) read by path; a value the registry does not hold is a named
refusal. The band of a type rule is nominal x [L, H] of its type (review M4); a description that names a reading as
its low edge (ruling R2) takes that reading x L of the reading's type. A type rule's fixed-outside reading (the core
grid plus the largest published outside part of the dies its gamma averages, MA-3) stands beside the band, never as
an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import units as u
# a cell's area (core/cells.py, the third delta round: moved for the module budget), re-exported here
from mimarsinan.deployment_record.platform_physics.area_rule.core.cells import (  # noqa: F401
    cell_area as cell_area, cell_factor as cell_factor, logic_rule_factor as logic_rule_factor,
    reads_sigma as reads_sigma)
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.frozen import freeze
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import (
    NONNEGATIVE, POSITIVE, judged, judged_band, power)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut

#: The scaled-term forms that count stored bits (the others count cells and converters); the technology types are the
#: registry's (its type tables' own ``type`` fields, ``Registry.types``).
BITS_TERMS = ("bits",)
#: The axes the block prices: the wizard's head (gui/wizard/physics_panel.py) and the fidelity gate
#: (deployment_record/fidelity_build.py) read this one tuple (through declaration.py).
AREA_RULE_AXES = ("chip_area_mm2",)


class ConverterRefusal(AreaRuleRefusal):
    """A named refusal of a core's converter (its kind, bits or unit area), with the core's array part priced before
    it (``array_um2``): the held-out estimate carries the readings of a row refused by converter kind."""

    def __init__(self, message: str, array_um2: Optional[float] = None):
        super().__init__(message)
        self.array_um2 = array_um2


class CensusCore(NamedTuple):
    """One core type of a census: rows (device rows), columns, the weight bits (None: not stated), pooled (a pooled
    census's one mean core, whose rows and columns may be fractions), read as declared (a census that states no core
    geometry, read at the description's own core)."""

    rows: float
    cols: float
    weight_bits: Optional[float] = None
    pooled: bool = False
    read_as_declared: bool = False


@dataclass(frozen=True)
class AreaFlag:
    """A departure the price names (computed at read, never sealed): kind, reason and, for a reading, its value."""

    kind: str
    reason: str
    value: Optional[float] = None


@dataclass(frozen=True)
class AreaPrice:
    """chip_area_mm2's band (mm^2) with its flags; ``fixed_reading`` is an anchor's off-size edge or a type rule's
    reading beside the band."""

    nominal: float
    low: float
    high: float
    fixed_reading: Optional[float] = None
    flags: Tuple[AreaFlag, ...] = ()
    ratio: Optional[float] = None
    cells_per_weight: Optional[int] = None
    readings: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # mi-K1: the band judged where it is made, its order with it, and its readings
        judged_band(f"{AREA_RULE_AXES[u.FIRST]}'s band", self.low, self.nominal, self.high)
        for name, value in (("fixed_reading", self.fixed_reading), *self.readings.items()):
            if value is not None:
                judged("{}'s reading {}", value, POSITIVE, AREA_RULE_AXES[u.FIRST], name)

    def reading(self, name: str) -> float:
        return self.readings[name]


class ConverterUnit(NamedTuple):
    value: float
    kind: str
    bound: Optional[str]
    bits_stated: bool


@door
def type_file(registry: Any, type_name: str) -> str:
    """The type table's file, a named refusal for a type the registry's tables do not state."""
    return registry.type_file(type_name)


@door
def type_value(registry: Any, type_name: str, name: str, excl: Sequence[str] = ()) -> float:
    try:
        return registry.value(f"{type_file(registry, type_name)}:{name}", excl)
    except HeldOut as held:
        raise AreaRuleRefusal(f"{type_name} {name}: it rests on the held-out chip's entries ({held})") from None


@door
def type_gamma(registry: Any, type_name: str, excl: Sequence[str] = ()) -> float:
    return type_value(registry, type_name, "gamma", excl)


@door
def band_factors(registry: Any, type_name: str) -> Tuple[float, float]:
    type_file(registry, type_name)
    return registry.band_of_type(type_name)


# ------------------------------------------------------------------ converters (rule E)
@door(declared="name or absent")
def resolve_kind(registry: Any, type_name: str, declared: Optional[str]) -> Optional[str]:
    """The declared kind, else the type's default kind (None: the type converts nothing)."""
    return declared if declared is not None else registry.type_table(type_name).get("default_converter_kind")


@door
def check_bits(kind: str, bits: Optional[float], low: float, high: float) -> None:
    """A kind's area holds over its published bits only: a named refusal outside them, and where the run declares no
    bits (review N-6: kinds resolve at price_absolute's readout bits)."""
    if bits is None:
        raise AreaRuleRefusal(f"converter kind {kind}: its area depends on the converter's bits ({low:g} to {high:g}) "
                              "and the run declares none (readout_law.adc_bits)")
    if not low <= bits <= high:
        if low == high:
            raise AreaRuleRefusal(f"converter kind {kind} publishes {low:g}-bit areas only; {bits} requested")
        raise AreaRuleRefusal(f"converter kind {kind}: its law holds over {low:g} to {high:g} bits; {bits} requested")


@door
def converter_area(registry: Any, kind: Optional[str], bits: Optional[float], excl: Sequence[str] = (),
                   check: bool = True) -> ConverterUnit:
    """One converter's unit area (um^2) of a kind at the requested bits (``check`` False: a reading that lifts the
    kind's bits range, never a price; a law's unit judged where it is made, a size above 0)."""
    if kind is None:
        raise AreaRuleRefusal("converter kind: the description converts but declares no kind and its type names none")
    spec = registry.converter_kind(kind)
    if "law" in spec:
        a_ref, alpha, (low, high) = registry.law(spec["law"], excl, f"converter kind {kind}")
        if check or bits is None:  # a law reads the bits: none stated is a named refusal, a reading's lifted too
            check_bits(kind, bits, low, high)
        reference = spec["law"]["recompute"]["reference_bits"]
        unit = a_ref * power(u.BINARY, alpha * (bits - reference))
        return ConverterUnit(judged("converter kind {}'s unit at {!r} bits", unit, POSITIVE, kind, bits), kind, None,
                             True)
    if "resolution" in spec:
        for name in spec["resolution"]:
            try:
                return ConverterUnit(registry.object_value(spec[name], tuple(excl), f"{kind}.{name}"), kind, None,
                                     False)
            except HeldOut:
                continue
        raise AreaRuleRefusal(f"converter kind {kind}: no published unit area remains (outside the held-out chip)")
    remains = f"converter kind {kind}: no published unit area remains (outside the held-out chip)"
    try:
        unit = registry.object_value(spec["unit_area"], tuple(excl), f"{kind}.unit_area")
    except (HeldOut, AreaRuleRefusal):
        raise AreaRuleRefusal(remains) from None
    stated = registry.object_value(spec["bits"], tuple(excl), f"{kind}.bits")
    if check and bits is not None and bits != stated and spec["unit_area"].get("bound"):
        raise AreaRuleRefusal(f"converter kind {kind}: its bound holds at {stated:g} bits; {bits} requested")
    if check:  # mi-3: a kind at one published width, a bound's included, needs the run's declared bits
        check_bits(kind, bits, stated, stated)
    return ConverterUnit(unit, kind, spec["unit_area"].get("bound"), True)


# ------------------------------------------------------------------ cores (a cell's area in core/cells.py)
class CoreTerms(NamedTuple):
    core_um2: float
    array_um2: float
    state_um2: float
    converters_um2: float


@door
def core_area(n_cells: float, cell_um2: float, eta: float, share: float, n_state: float = u.NONE,
              sigma_um2: float = u.NONE, n_conv: float = u.NONE, conv_um2: float = u.NONE) -> CoreTerms:
    """A_core = (A_mem + A_conv) / s with A_mem = (N_cells a_cell + N_state sigma) / eta; each term judged where it is
    made (a product beyond the floats is infinity, which the judge refuses by the term's name)."""
    cells = n_cells * cell_um2 / eta
    state = n_state * sigma_um2 / eta
    converters = n_conv * conv_um2
    terms = CoreTerms((cells + state + converters) / share, cells, state, converters)
    for name, value in terms._asdict().items():
        judged("the core's {}", value, POSITIVE if name == "core_um2" else NONNEGATIVE, name)
    return terms


# ------------------------------------------------------------------ the outside part (M2)
class OutsidePart(NamedTuple):
    gamma: float
    fixed_mm2: float
    fixed_from: Optional[str]


@door
def stated_fixed_parts(fixed: Mapping[str, Optional[float]]) -> Mapping[str, float]:
    return {place: value for place, value in fixed.items() if value is not None}


@door
def outside_rule(outside: str, gamma: Optional[float], fixed: Mapping[str, Optional[float]]) -> OutsidePart:
    """The outside part by the block's field: declared_parts take gamma 1 and the stated part (an override replaces
    the profile's row; a store restatement re-derives it); own_die and type take gamma and no fixed part, and an
    area_global_fixed stated beside gamma (profile, override or store) is a named refusal."""
    stated = stated_fixed_parts(fixed)
    if outside == "declared_parts":
        for place in ("override", "store", "profile"):
            if place in stated:
                return OutsidePart(u.UNITY, float(stated[place]), place)
        raise AreaRuleRefusal("the chip declares neither chip_components nor area_global_fixed: its chip-level area "
                              "is unknown (the fixed-area refusal, kept for declared_parts)")
    if outside in ("own_die", "type"):
        if stated:
            raise AreaRuleRefusal(f"the outside part is stated twice: gamma ({outside}) and area_global_fixed "
                                  f"({', '.join(sorted(stated))})")
        if gamma is None:
            raise AreaRuleRefusal(f"outside '{outside}' needs gamma; neither the block nor its type declares it")
        return OutsidePart(float(gamma), u.NONE, None)
    raise AreaRuleRefusal(f"outside {outside!r} is not one of declared_parts, own_die, type")


@door(gamma="share")
def outside_gamma(outside: str, gamma: float) -> float:
    """The core grid's share of the chip the outside field takes: declared parts take gamma 1 (their part is stated
    beside the grid), own_die and type take ``gamma``."""
    return u.UNITY if outside == "declared_parts" else gamma


@door(block="block or absent")
def wizard_also(block: Any) -> str:
    """What the chip-area axis needs beside its head constant, named by the block's outside field."""
    if block is None or block.outside == "declared_parts":
        return "area_global_fixed" + ("" if block is None else " (the chip components' sum or a sourced scalar)")
    return {"own_die": "area_rule.anchor.die_mm2 (the published die and its core count)",
            "type": "area_rule.type (the type table's gamma)"}[block.outside]


class HeldPart(NamedTuple):
    value: float
    parts: Mapping[str, float]


@door
def held_outside_part(registry: Any, type_name: str) -> HeldPart:
    """MA-3: the published outside parts of the dies a type's gamma averages, and the largest, which a type rule's
    fixed-outside reading holds; a named refusal where the type's table states none (the fix round's MA-1). Made once
    per type and registry (the fourth delta round: core/door.remembered), its parts frozen."""
    def made() -> HeldPart:
        table = registry.type_table(type_name)
        stated = table.get("outside_parts")
        owner = stated if isinstance(stated, str) else type_file(registry, type_name)
        obj = registry.files[owner].get("outside_parts")
        if not isinstance(obj, Mapping) or "evidence_kind" not in obj:
            raise AreaRuleRefusal(f"type {type_name} states no published outside part ({owner}: outside_parts names "
                                  "none): the fixed-outside reading has no published part to hold")
        parts = {p["label"]: registry.evaluate(p["expression"]) for p in obj["recompute"]["points"]}
        return HeldPart(registry.value(f"{owner}:outside_parts"), freeze(parts))
    return remembered(registry, "held outside part", (), made, type_name)
