"""[U-AREA] The judge's table of declared fields (the third delta round): every field the rule reads off every holder
(a description's block and its holders, the physics, its validity geometry and the rows beside the block, a census
core, a caller's core type, a declaration, a core's counts, an anchor's law, a published point and its scopes, a parity
row and its readings), each with its judge (core/specs.py, re-exported here). ``judged_holder`` judges a holder by its
table; a structural test holds every read of the rule's code a field of this table.

A holder read by key (a block's holder, a published point) may leave a field out (what it requires, the judge's own
rules state by name); a holder read by attribute (a census core, a law) states every field, absent as None where its
judge allows. ``CLASSES`` (the fourth delta round) declares each holder's class, the only place a class is named:
the declared input domain (core/door.py) admits an object of these classes, by exact type, and walks the fields this
table declares (``via``: a field judged under another holder's name, or a property the walk covers).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from mimarsinan.deployment_record.platform_physics.area_rule.core import equation as core_rule
from mimarsinan.deployment_record.platform_physics.area_rule.core.judge import (
    CELL, COUNT, COUNT_OR_ZERO, FINITE, NONNEGATIVE, POSITIVE, SHARE, judged)
from mimarsinan.deployment_record.platform_physics.area_rule.core.specs import (  # noqa: F401 - the table's judges
    CENSUS, Spec, area_flags, beside_constants, beside_rows, census_extent, citation, expression, flag, held_logic,
    law_organization, memory_beside, name, one_of, quantity, row_ratio, terms_form, text, texts, via)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal


def holder(holder_name: str, absent: bool = True) -> Spec:
    def judge(registry: Any, field: str, value: Any, held: Any) -> None:
        if value is None and not absent:
            raise AreaRuleRefusal(f"{field} None is not {holder_name}")
        judged_holder(registry, field, value, holder_name)
    setattr(judge, "of", holder_name)  # the declared domain: absent or an object of its class
    return judge


def holders(holder_name: str, empty: bool = False, form: Any = (list, tuple)) -> Spec:
    def judge(registry: Any, field: str, value: Any, held: Any) -> None:
        if not isinstance(value, form) or not (value or empty):
            raise AreaRuleRefusal(f"{field} {value!r:.120} is not a {'tuple' if form is tuple else 'list'} of "
                                  f"{holder_name} objects")
        for i, item in enumerate(value):
            judged_holder(registry, f"{field}[{i}]", item, holder_name)
    return judge


def _org(extra: Mapping[str, Spec]) -> Dict[str, Spec]:
    quantities = {k: quantity(kind) for k, kind in core_rule.QUANTITIES.items()}
    return {**quantities, "terms": terms_form, **extra}


TERMS = ("type_rule", "complete")
OUTSIDE = ("declared_parts", "own_die", "type")
SCOPES = ("die", "core", "macro")
OWN_SHARE_BASES = ("published_split", "own_memory_share")
#: The holders read by attribute; every other holder is read by key (``BY_METHOD``: by the census's has and get).
BY_ATTRIBUTE = frozenset({"block", "physics", "physics validity", "validity geometry", "census core", "core type",
                          "census proof", "declaration", "counts", "law", "own share answer", "core terms", "row",
                          "stack", "parity reading", "area flag", "census value", "physics constant", "converter table",
                          "registry"})
BY_METHOD = frozenset({"census quantities"})
#: The judge's table: each holder's fields, each with its judge.
FIELDS: Mapping[str, Mapping[str, Spec]] = {
    "block": {"type": name("type"), "terms": one_of(TERMS), "outside": one_of(OUTSIDE), "node": holder("node"),
              "anchor": holder("anchor"), "organization": holder("organization"), "cell_kind": name("cell kind"),
              "converter": holder("converter"), "reading_low_edge": holder("reading"),
              "published_size": holder("published size"), "drops": texts(), "note": text(),
              "node_nm": via("node", False), "bitcell_um2": via("node", False)},
    "node": {"node_nm": quantity(POSITIVE), "bitcell_um2": expression(CELL, "a declared cell")},
    "anchor": {"scope": one_of(SCOPES), "core_mm2": expression(POSITIVE, "a published figure"),
               "die_mm2": expression(POSITIVE, "a published figure"), "n_pub": expression(COUNT, "a published figure"),
               "own_share": holder("own share"), "logic_held_per_core": held_logic},
    "own share": {"basis": one_of(OWN_SHARE_BASES), "expression": expression(FINITE)},  # its share: N-3's path
    "organization": _org({"citation": citation}),
    "reading": _org({"type": name("type"), "ruling": text()}),
    "converter": {"kind": name("converter kind"), "own_unit": name("evidence id")},
    "published size": {"n_pub": expression(COUNT, "a published figure")},
    # the rows beside the block (constants, converter) are judged by judge_beside at the physics' judge (MA-C2)
    "physics": {"area_rule": via("block", of="block"), "validity": holder("physics validity", absent=False),
                "constants": via("beside", beside_constants, every=True),
                "converter": via("beside", of="converter table", every=True)},
    "physics validity": {"geometry": via("validity geometry", of="validity geometry")},  # under its own name
    "validity geometry": {"axons": quantity(COUNT), "neurons": quantity(COUNT), "cell_bits": quantity(COUNT, True)},
    "physics constant": {"nominal": via("beside")},
    "converter table": {"rows": via("beside", beside_rows)},
    # the rows beside the block as converter_area_stated and beside_of read them, by key (declaration.py)
    "physics constants": {"area_global_fixed": via("beside"), "area_per_adc": via("beside")},
    "converter row": {"area_per_adc": via("beside")},
    "census core": {"rows": census_extent, "cols": census_extent, "weight_bits": quantity(COUNT, True),
                    "pooled": flag(), "read_as_declared": flag()},
    "core type": {"axons": quantity(COUNT), "neurons": quantity(COUNT), "count": quantity(COUNT)},
    "census proof": {"strict": flag()},
    "census quantities": {"cores_physical": quantity(COUNT), "cells_physical": quantity(COUNT_OR_ZERO),
                          "neurons_physical": quantity(COUNT), "weight_bits": quantity(COUNT)},
    "census value": {"value": via("census quantities")},
    "declaration": {"node_nm": quantity(POSITIVE), "cell_kind": name("cell kind"), "bitcell_um2": quantity(CELL, True),
                    "converter_kind": name("converter kind"), "own_unit": name("evidence id")},
    "counts": {"array": quantity(POSITIVE), "converters": quantity(NONNEGATIVE), "beside": memory_beside},
    "law": {"type": name("type"), "scope": one_of(SCOPES), "outside": one_of(OUTSIDE), "node_nm": quantity(POSITIVE),
            "core_mm2": quantity(POSITIVE), "n_pub": quantity(COUNT), "gamma": quantity(SHARE),
            "organization": law_organization, "own_share": holder("own share answer", absent=False),
            "logic_held_per_core": flag(), "die_mm2": quantity(POSITIVE, True), "gamma_from": text(),
            "cell_flags": area_flags},
    "own share answer": {"value": quantity(SHARE, True), "basis": text(), "refusal": text(True)},
    "area flag": {"kind": text(), "reason": text(), "value": quantity(FINITE, True)},
    "row": {"chip": text(), "type": text(), "scope": text(), "primary": flag(), "cls": text(), "model": flag(),
            "scope_mismatch": flag(), "published_mm2": quantity(POSITIVE), "ratio": row_ratio,
            "refused": text(True), "refused_by": text(True), "array_um2": quantity(NONNEGATIVE, True),
            "eta": quantity(POSITIVE, True), "gamma": quantity(SHARE, True), "count": quantity(POSITIVE),
            "terms": holder("core terms"), "readings": holders("parity reading", empty=True, form=tuple)},
    "parity reading": {"label": text(), "converter_um2": quantity(NONNEGATIVE), "ratio": quantity(POSITIVE),
                       "type_default": flag()},
    "stack": {"chip": text(), "array_ratio": quantity(NONNEGATIVE), "family": text(True)},
    "core terms": {"core_um2": quantity(POSITIVE), "array_um2": quantity(NONNEGATIVE),
                   "state_um2": quantity(NONNEGATIVE), "converters_um2": quantity(NONNEGATIVE)},
    "point": {"name": text(), "type": name("type"), "node_nm": quantity(POSITIVE), "held_out": texts(),
              "own_units": texts(True), "cell_kind": name("cell kind"), "converter": holder("point converter"),
              "organization": holder("point organization", absent=False), "scopes": holders("scope"),
              "model": flag(True), "scope_mismatch": flag(True), "array_figure": holder("array figure")},
    "point organization": {"memory_bits": expression(POSITIVE), "cells": expression(POSITIVE),
                           "arrays": expression(POSITIVE), "array_cells": expression(POSITIVE),
                           "memory_beside": holder("memory beside")},
    "memory beside": {"bits": expression(POSITIVE), "type": name("type")},
    "point converter": {"kind": name("converter kind"), "own_unit": name("evidence id"), "bits": expression(COUNT),
                        "count": expression(NONNEGATIVE), "per_array": expression(NONNEGATIVE)},
    "scope": {"scope": text(), "published_mm2": expression(POSITIVE), "primary": flag(), "class": text(),
              "count": expression(COUNT), "gamma": flag(True), "sub_arrays": expression(COUNT),
              "fixed_mm2": holders("fixed part")},
    "fixed part": {"expression": expression(NONNEGATIVE), "fallback": expression(NONNEGATIVE)},
    "array figure": {"scope": text(), "um2": expression(POSITIVE), "ids": texts()},
    "registry": {},  # the rule's own data, loaded and checked (T-9): no caller's value
}

_AR, _PP, _DR = ("mimarsinan.deployment_record.platform_physics.area_rule.",
                 "mimarsinan.deployment_record.platform_physics.", "mimarsinan.deployment_record.")
#: Each holder read by attribute (or by method) and its declared classes, by their module and qualified name.
CLASSES: Mapping[str, tuple] = {
    "block": (_AR + "declaration.AreaRuleBlock",), "physics": (_PP + "profile.PlatformPhysics",),
    "physics validity": (_PP + "profile.PlatformPhysicsValidity",),
    "validity geometry": (_PP + "profile.ValidityGeometry",), "census core": (_AR + "rule.CensusCore",),
    "core type": (_DR + "cost.absolute.validity.CoreGeometry",),
    "census proof": (_DR + "cost.absolute.validity.CensusCells",),
    "census quantities": (_DR + "quantities.spec.Quantities",),
    "census value": (_DR + "quantities.spec.QuantityValue",),
    "declaration": (_AR + "core.equation.Declared",), "counts": (_AR + "core.equation.Counts",),
    "law": (_AR + "anchor.AnchorLaw",), "own share answer": (_AR + "anchor.OwnShare",),
    "core terms": (_AR + "rule.CoreTerms",), "row": (_AR + "published.parity.Row",),
    "parity reading": (_AR + "published.parity.Reading",), "stack": (_AR + "published.decomposition.Stack",),
    "area flag": (_AR + "rule.AreaFlag",), "physics constant": (_PP + "constants.PhysicsConstantValue",),
    "converter table": (_PP + "converter.ConverterTable",),
    "registry": (_AR + "registry.Registry", _AR + "view.View")}
HOLDER_OF: Mapping[str, str] = {name: h for h, names in CLASSES.items() for name in names}
#: The holders whose objects are frozen (a block through and through by core/frozen.py, the frozen validity and its
#: geometry of leaves, the frozen physics): the domain remembers their pass by their identity, but for the fields the
#: table marks read at every call (the physics' rows beside the block: its constants and its converter's rows).
FROZEN = frozenset({"block", "physics", "physics validity", "validity geometry"})
#: A holder's place in the rule's words wherever it is found (a block is the description's area_rule).
PLACES: Mapping[str, tuple] = {"block": ("area_rule",), "physics": (), "validity geometry": ("validity", "geometry"),
                               "census quantities": (CENSUS,)}
#: The fields a holder read by key must state (a block's holders' own requirements are the block judge's, by name).
REQUIRED: Mapping[str, tuple] = {
    "point": ("name", "type", "node_nm", "held_out", "organization", "scopes"),
    "scope": ("scope", "published_mm2", "primary", "class"), "memory beside": ("bits", "type"),
    "fixed part": ("expression",), "array figure": ("scope", "um2", "ids")}


#: Each holder's judged fields, made once from the table: (field, its judge, its kind where the judge is a quantity's,
#: absent allowed); a field the table marks ``via`` is judged under its own holder's name.
JUDGED: Dict[str, tuple] = {name: tuple((f, s, getattr(s, "kind", None), getattr(s, "absent", False))
                                        for f, s in table.items() if getattr(s, "via", None) is None)
                            for name, table in FIELDS.items()}


def judged_holder(registry: Any, place: str, value: Any, holder_name: str) -> Any:
    """A holder at ``place`` judged by its table: an object (or absent), each field it states judged by its judge (a
    quantity by the one judge, its words made only on a refusal); a holder read by attribute states every field; one
    read by key states its required fields. A field the table marks ``via`` is judged under its own holder's name, and
    only such a field is skipped (MA-C4: never a class property for its own sake)."""
    if value is None:
        return value
    if holder_name in BY_ATTRIBUTE:
        for field, spec, kind, absent in JUDGED[holder_name]:
            try:
                item = getattr(value, field)
            except AttributeError:
                raise AreaRuleRefusal(f"{place} {value!r:.80} is not a {holder_name}: it states no {field}") from None
            if kind is None:
                spec(registry, f"{place}.{field}", item, value)
            elif not (absent and item is None):
                judged("{}.{}", item, kind, place, field)
        return value
    if not isinstance(value, Mapping):
        raise AreaRuleRefusal(f"{place} {value!r} is not an object")
    if missing := [f for f in REQUIRED.get(holder_name, ()) if value.get(f) is None]:
        raise AreaRuleRefusal(f"{place} states no {missing}")
    for field, spec, _kind, _absent in JUDGED[holder_name]:
        if field in value and value[field] is not None:
            spec(registry, f"{place}.{field}", value[field], value)
    return value


__all__ = ["BY_ATTRIBUTE", "BY_METHOD", "CENSUS", "CLASSES", "FIELDS", "FROZEN", "HOLDER_OF", "JUDGED", "OUTSIDE",
           "Spec", "OWN_SHARE_BASES", "PLACES", "SCOPES", "TERMS", "judged_holder"]
