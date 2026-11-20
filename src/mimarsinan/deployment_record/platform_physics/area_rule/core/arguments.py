"""[U-AREA] The door's table (core/door.py, the third delta round): each argument of a door by its name (``ARGUMENTS``)
and each argument's judge (``JUDGES``), a declared object's fields by the judge's table (core/fields.py). The block, the
validity geometry and the rows beside the block are judged by the same judges at load (declaration.judge_block) and at
the price (a physics: the fourth delta round, MA-C2), once per frozen input (core/door.remembered, keyed without
pickling); the published points' judges are in core/points.py."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from mimarsinan.deployment_record.platform_physics.area_rule import declaration, units as u
from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule, fields, points
from mimarsinan.deployment_record.platform_physics.area_rule.core.door import holder_of, remembered
from mimarsinan.deployment_record.platform_physics.area_rule.core.fields import CENSUS
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import (
    CELL, COUNT, NONNEGATIVE, POSITIVE, SHARE, Kind, judged)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import parity
from mimarsinan.deployment_record.platform_physics.area_rule.registry import Registry, View, load_registry

Judge = Callable[[Any, str, Any, Mapping[str, Any], bool], None]


def registry_of(arguments: Mapping[str, Any], registry: Any) -> Any:
    """The registry a door's judges read: its own argument, else the method's registry, else the shipped one."""
    for name in ("registry", "reg", "base"):
        if isinstance(arguments.get(name), (Registry, View)):
            return arguments[name]
    return registry if isinstance(registry, (Registry, View)) else load_registry()


def _kind(kind: Kind, absent: bool = False) -> Judge:
    def judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
        if not (absent and value is None):
            judged(name, value, kind)
    return judge


def _spec(spec: fields.Spec, absent: bool = False) -> Judge:
    def judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
        if not (absent and value is None):
            spec(registry, name, value, arguments)
    return judge


def _holder(holder: str, absent: bool = False) -> Judge:
    def judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
        if value is None and not absent:
            raise AreaRuleRefusal(f"{name} None is not a {holder}")
        fields.judged_holder(registry, name, value, holder)
    return judge


def _sequence(test: Callable[[Any], bool], words: str) -> Judge:
    def judge(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
        if not isinstance(value, (list, tuple)) or not value or not all(test(x) for x in value):
            raise AreaRuleRefusal(f"{name} {value!r:.120} is not {words}")
    return judge


def _block(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """A block with every holder (declaration.judged_block), at a door once per frozen block (by its identity)."""
    if not isinstance(value, declaration.AreaRuleBlock):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not an area_rule block")
    if memo:
        remembered(registry, "block", (value,), lambda: declaration.judged_block(value, registry))
    else:
        declaration.judged_block(value, registry)


def _anchor_block(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    _block(registry, name, value, arguments, memo)
    if value.anchor is None:
        raise AreaRuleRefusal(f"{name}: an anchor's law reads the block's anchor, and the block states none")


def _validity(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """The description's validity geometry the count form reads (MA-K1), by its description's own field names."""
    fields.judged_holder(registry, "validity.geometry", value, "validity geometry")


def _beside(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """What the description states beside its block (judge_beside, at load, at the price and at every entry those
    rows have)."""
    (fixed, table), block = value, arguments.get("block")
    if isinstance(block, declaration.AreaRuleBlock):  # one that is no block is the block judge's named refusal
        declaration.judge_beside(block, fixed_stated=fixed, converter_table=table)


def _physics(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """A physics with a block (one without is R5's): its block, its validity geometry and the rows beside the block by
    the loader's judges, in the loader's order, once per frozen input: the identities of its block and geometry, the
    rows beside the block (declaration.beside_of, read at every call) as its digest (MA-C2)."""
    physics = value
    block = getattr(physics, "area_rule", None)
    if block is None:
        return
    validity = physics.validity
    geometry = getattr(validity, "geometry", None)  # a validity of no geometry is the physics' judge's refusal
    beside = declaration.beside_of(physics)
    local = {"block": block, "validity": geometry, "beside": beside}

    def judge() -> None:
        fields.judged_holder(registry, name, physics, "physics")
        for argument, table in (("block", "block"), ("validity", "validity geometry"), ("beside", "beside")):
            JUDGES[table](registry, argument, local[argument], local, False)
    if memo and isinstance(block, declaration.AreaRuleBlock):
        remembered(registry, "physics", (block, validity, geometry), judge, beside[u.FIRST]["profile"],
                   beside[u.SECOND])
    else:
        judge()


def _physics_with_block(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if getattr(value, "area_rule", None) is None:
        raise AreaRuleRefusal(f"{name} {value!r:.80} declares no area_rule block: R5's own path prices it")
    _physics(registry, name, value, arguments, memo)


def _quantities(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """The census's quantities the rule reads, each of its kind (the census an object of its declared class)."""
    if holder_of(type(value)) != "census quantities":
        raise AreaRuleRefusal(f"{CENSUS} {value!r:.80} is not the census's quantities (has, get, withheld_reason)")
    present = set(value.keys())
    for key, spec, kind, absent in fields.JUDGED["census quantities"]:  # a quantity's words made only on a refusal
        if key in present:
            item = value.get(key).value
            if kind is None:
                spec(registry, f"{CENSUS}: {key}", item, value)
            elif not (absent and item is None):
                judged("{}: {}", item, kind, CENSUS, key)


def _core_types(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """The caller's core types (a list of core types), what its census proves (its ``strict``), or none."""
    if isinstance(value, (tuple, list)):
        if not value:
            raise AreaRuleRefusal(f"{CENSUS}'s core types {value!r} state no core type")
        for i, core in enumerate(value):
            fields.judged_holder(registry, f"{CENSUS}'s core types[{i}]", core, "core type")
    elif value is not None:  # what a census proves: read only by its ``strict`` (CensusCells)
        fields.FIELDS["census proof"]["strict"](registry, f"{CENSUS}'s core types.strict",
                                                getattr(value, "strict", False), value)


def _types(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if not isinstance(value, (tuple, list)) or not value:
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not a list of core types (count, census core)")
    for i, pair in enumerate(value):
        if not isinstance(pair, (tuple, list)) or len(pair) != len(("count", "core")):
            raise AreaRuleRefusal(f"{name}[{i}] {pair!r:.80} is not a pair (count, census core)")
        judged("{}[{}] count", pair[u.FIRST], COUNT, name, i)
        fields.judged_holder(registry, f"{name}[{i}] core", pair[u.SECOND], "census core")


def _organization(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    place = arguments.get("place")
    core_rule.judged_fields(value, place if isinstance(place, str) else core_rule.ORGANIZATION)


def _anchor_organization(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    """An anchor's organization read without its type: its fields, its terms the registry's, its own core on them."""
    org = core_rule.judged_fields(value, core_rule.ORGANIZATION)
    if missing := [k for k in ("rows", "cols", "weight_bits_per_cell", "terms", "term_weights") if k not in org]:
        raise AreaRuleRefusal(f"{core_rule.ORGANIZATION} states no {missing}")
    fields.terms_form(registry, f"{core_rule.ORGANIZATION}.terms", org["terms"], org)
    core_rule.weighted_counts(org["terms"], org)


def _reading(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if value is not None:
        core_rule.judged_fields(value, core_rule.READING)


def _extent(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    judged(name, value, POSITIVE if arguments.get("pooled") is True else COUNT)


def _type(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    registry.type_file(value)


def _type_field(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    table = registry.type_table(arguments.get("type_name"))
    if not isinstance(value, str) or not isinstance(table.get(value), Mapping):
        raise AreaRuleRefusal(f"type {arguments.get('type_name')!r} states no value {value!r}")


def _fixed(registry: Any, name: str, value: Any, arguments: Mapping[str, Any], memo: bool) -> None:
    if not isinstance(value, Mapping) or not all(isinstance(place, str) for place in value):
        raise AreaRuleRefusal(f"{name} {value!r:.80} is not a mapping of places to areas")
    for place, area in value.items():
        if area is not None:
            judged("{}.{}", area, NONNEGATIVE, name, place)


#: Each argument's judge by its name; a door names its own where a name means another thing there.
JUDGES: Dict[str, Judge] = {
    "registry": points.registry_judge, "registry with points": points.points_judge,
    "physics with a block": _physics_with_block, "block": _block, "anchor block": _anchor_block,
    "block or absent": _spec(lambda r, n, v, a: _block(r, n, v, a, True), True), "physics": _physics,
    "validity geometry": _validity, "beside": _beside, "census quantities": _quantities, "core types": _core_types,
    "census types": _types, "census core": _holder("census core"),
    "census core or absent": _holder("census core", True), "count": _kind(COUNT), "bits": _kind(COUNT, True),
    "size": _kind(POSITIVE), "size or absent": _kind(POSITIVE, True),
    "share": _kind(SHARE), "share or absent": _kind(SHARE, True), "non-negative": _kind(NONNEGATIVE),
    "cell or absent": _kind(CELL, True), "cell": _kind(CELL), "type": _type, "type field": _type_field,
    "name or absent": _spec(fields.text(True)), "text": _spec(fields.text()), "flag": _spec(fields.flag()),
    "flag or absent": _spec(fields.flag(True)), "terms form": _spec(fields.terms_form),
    "organization": _organization, "anchor organization": _anchor_organization, "reading or absent": _reading,
    "census extent": _extent, "declaration": _holder("declaration"), "counts": _holder("counts"),
    "law": _holder("law"), "core terms": _holder("core terms"), "exclusions": _spec(fields.texts()),
    "fixed parts": _fixed, "outside": _spec(fields.one_of(fields.OUTSIDE)), "point": points.point_judge,
    "scope": points.scope_judge, "mode": _spec(fields.one_of(parity.MODES)), "parity rows": points.rows(),
    "spread rows": points.rows(spread=True), "strict rows": points.rows(spread=True, priced=True),
    "figure rows": points.rows(figures=True), "ratios": _sequence(lambda x: judged("a ratio", x, POSITIVE) is x,
                                                                   "a list of ratios"),
    "readings": _spec(fields.one_of(("type_default", "all"))),
    "scope rule": _spec(fields.one_of(("published", "macro"))),
    "evidence ids": _sequence(lambda x: isinstance(x, str), "a list of evidence ids"), "stacks": points.stacks,
}
#: Each argument of a door, by its name, and its judge.
ARGUMENTS: Mapping[str, str] = {
    "registry": "registry", "reg": "registry", "base": "registry", "physics": "physics", "block": "block",
    "validity": "validity geometry", "own_core": "validity geometry", "beside": "beside",
    "quantities": "census quantities", "geometry": "core types", "types": "census types", "census": "census core",
    "cores": "count", "n": "count", "n_pub": "count", "weight_rows": "count", "low": "count", "high": "count",
    "weight_bits": "bits", "readout_bits": "bits", "cell_bits": "bits", "bits": "bits", "type_name": "type",
    "kind": "name or absent", "declared": "declaration", "reading": "flag", "organization": "organization",
    "form": "terms form", "rows": "census extent", "cols": "census extent", "pooled": "flag", "converts": "flag",
    "logic_held": "flag", "check": "flag", "place": "text", "counts": "counts", "unit": "size or absent",
    "core_mm2": "size", "die_mm2": "size", "ratio": "size", "wider": "size or absent", "own_share": "share or absent",
    "law": "law", "excl": "exclusions", "name": "type field", "node_nm": "size", "declared_um2": "cell or absent",
    "n_cells": "size", "cell_um2": "cell", "eta": "share", "share": "share", "n_state": "non-negative",
    "sigma_um2": "non-negative", "n_conv": "non-negative", "conv_um2": "non-negative", "fixed": "fixed parts",
    "outside": "outside", "gamma": "share or absent", "point": "point", "mode": "mode", "scope": "scope",
    "terms": "core terms", "ratios": "ratios", "readings": "readings", "ids": "evidence ids",
    "scope_rule": "scope rule", "stacks": "stacks", "expression": "text",
}

__all__ = ["ARGUMENTS", "JUDGES", "registry_of"]
