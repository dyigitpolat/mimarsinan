"""[U-AREA] The held-out estimate of the published points (design revision 3, sections 3.3, 4 and 9.1; T-6, T-7,
T-22, T-26, T-27): every published chip, macro and model in ``data/evidence/published_points.json`` priced with the
registry viewed without the chip's own entries (``Registry.held_out``).

One core-area rule (the build's verify, MA-3): every core is priced by the type rule's own equation
(``typerule.core_terms``) from the point's declaration and counts, and a row's eta is that equation's own
(``core.equation.type_eta``, the fix round's re-verify mi-R3); this module holds no equation of a core. It reads a
point's organization as its counts (its stored bits, cells or arrays of cells, its converters, a memory beside its
core), holds the published scopes (a die over its type's gamma, a block below the core without the share, a scope's
published fixed parts), the readings of a row refused by converter kind and the statistics.

Modes: ``held_out`` (the chip's entries excluded by id prefix; its organization and own printed units kept as
inputs), ``strict`` (its own printed units excluded too, so its declaration gives way to its type's defaults) and
``in_sample``. A row the rule cannot price is a named refusal; a row refused by converter kind carries its readings
(the type's default kind with its bits range lifted, the bound's open low edge, each law kind at the row's bits),
never a price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from mimarsinan.deployment_record.platform_physics.area_rule import typerule, units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core import fields
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import door
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import POSITIVE, judged
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal, HeldOut
from mimarsinan.deployment_record.platform_physics.area_rule.rule import (
    ConverterRefusal, CoreTerms, converter_area, type_gamma)

MODES = ("held_out", "strict", "in_sample")
#: The organization fields that state a core's scaled count, one per point.
COUNT_FORMS = ("memory_bits", "cells", "arrays")


class Reading(NamedTuple):
    label: str
    converter_um2: float
    ratio: float
    type_default: bool


@dataclass(frozen=True)
class Row:
    chip: str
    type: str
    scope: str
    primary: bool
    cls: str
    model: bool
    scope_mismatch: bool
    published_mm2: float
    ratio: Optional[float] = None
    refused: Optional[str] = None
    refused_by: Optional[str] = None
    readings: Tuple[Reading, ...] = ()
    terms: Optional[CoreTerms] = None
    array_um2: Optional[float] = None
    eta: Optional[float] = None
    gamma: Optional[float] = None
    count: float = u.UNITY

    def __post_init__(self) -> None:  # a row judged where it is made, by the judge's table (its ratio a size above 0)
        fields.judged_holder(None, f"the row of {self.chip!r} at {self.scope!r}", self, "row")


@door
def die_gamma(registry: Any, type_name: str, excl: Sequence[str]) -> float:
    """The core-grid share a die scope divides by (the type's gamma under the exclusions)."""
    return type_gamma(registry, type_name, excl)


@door
def exclusions(point: Mapping[str, Any], mode: str) -> Tuple[str, ...]:
    if mode not in MODES:
        raise AreaRuleRefusal(f"parity mode {mode!r} is not one of {list(MODES)}")
    if mode == "in_sample":
        return ()
    return tuple(point["held_out"]) + (tuple(point.get("own_units") or ()) if mode == "strict" else ())


@door
def view(registry: Any, point: Mapping[str, Any], mode: str) -> Any:
    """The registry without the point's entries; held out, its own printed units stay inputs of its declaration."""
    own = tuple(point.get("own_units") or ()) if mode == "held_out" else ()
    return registry.held_out(exclusions(point, mode), own)


@door
def declared(point: Mapping[str, Any]) -> typerule.Declared:
    conv = point.get("converter") or {}
    return typerule.Declared(point["node_nm"], point.get("cell_kind"), None, conv.get("kind"), conv.get("own_unit"))


@door
def counts(registry: Any, point: Mapping[str, Any]) -> typerule.Counts:
    """The point's organization as its core's counts (its own inputs, read in-sample)."""
    org, conv, own = point.get("organization") or {}, point.get("converter") or {}, registry.evaluate
    forms = [f for f in COUNT_FORMS if f in org]
    if len(forms) != int(u.UNITY):
        raise AreaRuleRefusal(f"{point['name']}: its organization states {forms}, one of {list(COUNT_FORMS)}")
    array = own(org["arrays"]) * own(org["array_cells"]) if "arrays" in org else own(org[forms[u.FIRST]])
    n_conv = u.NONE
    if conv:
        n_conv = own(org["arrays"]) * own(conv["per_array"]) if "per_array" in conv else own(conv["count"])
    beside = org.get("memory_beside")
    return typerule.Counts(array, n_conv, None if beside is None else (own(beside["bits"]), beside["type"]))


def _core(reg: Any, base: Any, point: Mapping[str, Any], unit: Optional[float] = None) -> CoreTerms:
    bits = base.evaluate(point["converter"]["bits"]) if (point.get("converter") or {}).get("bits") else None
    return typerule.core_terms(reg, point["type"], declared(point), counts(base, point), bits, unit)


@door
def scope_mm2(reg: Any, base: Any, point: Mapping[str, Any], scope: Mapping[str, Any], terms: CoreTerms) -> Tuple[
        float, Optional[float], float]:
    """(the estimate at a scope in mm^2, its gamma, its core count)."""
    own = base.evaluate
    count = own(scope["count"]) if scope.get("count") else u.UNITY
    if scope.get("sub_arrays"):  # a block below the core: its arrays and converters without the share
        k = own(scope["sub_arrays"])
        sub = typerule.Counts(k * own(point["organization"]["array_cells"]),
                              k * own(point["converter"]["per_array"]))
        bits = own(point["converter"]["bits"]) if point["converter"].get("bits") else None
        block = typerule.core_terms(reg, point["type"], declared(point), sub, bits)
        est = (block.array_um2 + block.state_um2 + block.converters_um2) / u.UM2_PER_MM2
        return judged("the estimate at {} of {}", est, POSITIVE, scope["scope"], point.get("name")), None, u.UNITY
    est = count * terms.core_um2 / u.UM2_PER_MM2
    gamma = None
    if scope.get("gamma"):
        gamma = die_gamma(reg, point["type"], ())
        est = est / gamma
    for part in scope.get("fixed_mm2") or ():
        for expression in (part["expression"], part.get("fallback")):
            if expression is None:
                continue
            try:
                est += reg.evaluate(expression)
                break
            except HeldOut:
                continue
    return judged("the estimate at {} of {}", est, POSITIVE, scope["scope"], point.get("name")), gamma, count


@door
def law_kinds(registry: Any) -> Tuple[str, ...]:
    """The converter kinds whose unit area is a law over bits (each states its reading's label)."""
    return tuple(k for k, spec in registry.files["converter_kinds"]["kinds"].items() if "law" in spec)


def _readings(reg: Any, base: Any, point: Mapping[str, Any]) -> List[Reading]:
    """The readings beside a row refused by converter kind: its converter at other published units."""
    conv = point["converter"]
    bits = base.evaluate(conv["bits"]) if conv.get("bits") else None
    default = base.type_table(point["type"]).get("default_converter_kind")
    alternatives = []
    if base.converter_kind(conv["kind"]).get("unit_area", {}).get("bound"):
        alternatives.append(("no converter area (the bound's open low edge)", u.NONE, False))
    alternatives.append((f"the type's default kind {default} at {bits} bits (its range lifted)",
                         converter_area(reg, default, bits, check=False).value, True))
    for kind in (k for k in law_kinds(base) if k != default):
        alternatives.append((f"{base.converter_kind(kind)['reading']} at {bits} bits (extrapolated)",
                             converter_area(reg, kind, bits, check=False).value, False))
    primary = next(s for s in point["scopes"] if s["primary"])
    out = []
    for label, unit, type_default in alternatives:
        est, _g, _c = scope_mm2(reg, base, point, primary, _core(reg, base, point, unit))
        out.append(Reading(label, unit, est / base.evaluate(primary["published_mm2"]), type_default))
    return out


def _point_rows(registry: Any, point: Mapping[str, Any], mode: str) -> List[Row]:
    reg = view(registry, point, mode)
    base = dict(chip=point["name"], type=point["type"], model=bool(point.get("model")),
                scope_mismatch=bool(point.get("scope_mismatch")))
    rows = []
    try:
        terms = _core(reg, registry, point)
        eta = core_rule.type_eta(reg, point["type"])  # mi-R3: the eta its core equation divides by
        for scope in point["scopes"]:
            published = registry.evaluate(scope["published_mm2"])
            est, gamma, count = scope_mm2(reg, registry, point, scope, terms)
            rows.append(Row(scope=scope["scope"], primary=scope["primary"], cls=scope["class"],
                            published_mm2=published, ratio=est / published, terms=terms, array_um2=terms.array_um2,
                            eta=eta, gamma=gamma, count=count, **base))
    except AreaRuleRefusal as refused:
        by = "converter" if isinstance(refused, ConverterRefusal) else "rule"
        readings = tuple(_readings(reg, registry, point)) if by == "converter" else ()
        array = refused.array_um2 if isinstance(refused, ConverterRefusal) else None
        for scope in point["scopes"]:
            rows.append(Row(scope=scope["scope"], primary=scope["primary"], cls=scope["class"],
                            published_mm2=registry.evaluate(scope["published_mm2"]), refused=str(refused),
                            refused_by=by, readings=readings if scope["primary"] else (), array_um2=array,
                            count=registry.evaluate(scope["count"]) if scope.get("count") else u.UNITY, **base))
    return rows


@door(registry="registry with points")
def estimate(registry: Any, mode: str) -> List[Row]:
    """Every published point's rows (one per published scope) under a mode."""
    return [row for point in registry.published_points() for row in _point_rows(registry, point, mode)]


@door(rows="parity rows")
def headline(rows: Sequence[Row]) -> List[Row]:
    """The primary silicon rows without a scope mismatch (the headline set)."""
    return [r for r in rows if r.primary and not r.model and not r.scope_mismatch]


@door
def factor(ratio: float) -> float:
    return judged("the factor of a ratio", max(ratio, u.UNITY / ratio), POSITIVE)  # judged where it is made


@door
def stats(ratios: Sequence[float]) -> Dict[str, float]:
    logs = [math.log(r) for r in ratios]
    worst = max(ratios, key=lambda r: abs(math.log(r)))
    return {"n": len(ratios), "geomean": math.exp(sum(logs) / len(logs)), "min": min(ratios), "max": max(ratios),
            "worst": worst, "worst_factor": factor(worst),
            "within_2x": sum(u.UNITY / u.BINARY <= r <= u.BINARY for r in ratios),
            "within_3x": sum(u.UNITY / (u.BINARY + u.UNITY) <= r <= u.BINARY + u.UNITY for r in ratios)}


@door(rows="spread rows")
def strict_spread(rows: Sequence[Row], readings: str = "type_default") -> Dict[str, Dict[str, Any]]:
    """Form (ii)'s per-chip values: a priced row its ratio; a row refused by converter kind its type-default
    reading (``readings='all'``: every reading, the mutant of T-27); a row refused for a missing cell nothing."""
    out = {}
    for r in headline(rows):
        if r.refused is None:
            out[r.chip] = {"type": r.type, "values": [r.ratio], "how": "priced"}
        elif r.refused_by == "converter":
            out[r.chip] = {"type": r.type, "how": "refused by converter kind",
                           "values": [reading.ratio for reading in r.readings
                                      if readings == "all" or reading.type_default]}
    return out


@door(rows="strict rows")
def strict_form(rows: Sequence[Row]) -> Dict[str, Any]:
    spread = strict_spread(rows)
    priced = {c: x["values"][u.FIRST] for c, x in spread.items() if x["how"] == "priced"}
    kind = {c: x["values"][u.FIRST] for c, x in spread.items() if x["how"] != "priced"}
    worst = max(priced, key=lambda c: factor(priced[c]))
    values = list(priced.values()) + list(kind.values())
    out = {"n_priced": len(priced), "worst_factor": factor(priced[worst]), "worst_chip": worst,
           "n_refused_by_kind": len(kind), "refused_by_kind": sorted(kind),
           "readings_max_factor": max(factor(v) for v in kind.values()) if kind else None, "spread_min": min(values),
           "spread_max": max(values)}
    if not kind:  # the fifth check's MA-C3: no row refused by converter kind, no reading to bound
        out["readings_max_factor_note"] = "no headline row is refused by converter kind: its readings' factor is None"
    return out


from mimarsinan.deployment_record.platform_physics.area_rule.published.decomposition import (  # noqa: E402
    Stack, array_decomposition, sram_array_fact)

__all__ = ["MODES", "Reading", "Row", "Stack", "array_decomposition", "counts", "declared", "die_gamma", "estimate",
           "exclusions", "factor", "headline", "law_kinds", "sram_array_fact", "stats", "strict_form",
           "strict_spread", "view"]
