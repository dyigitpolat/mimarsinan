"""[U-AREA] The fourth delta round (UB5/check.md: MA-C1 to MA-C6; the ruling 'U-AREA's Third Delta Round: the Declared
Input Domain, the Door's Cost and the Last Round under the Generator Gate'), tests first. Each test is red on UB5's tree
(UB5/uarea.patch on R5, applied digest d9a1649d) or on its named mutant, green on the round's tree; a guard (green on
UB5's tree too) is named as such.

- The declared input domain (core/door.py): a leaf of an exact type a description file holds, a container of them, a
  holder of a class the judge's table declares; anything else a named refusal naming its type and its field, before
  any other judge, at every door, at load and at the price.
- The six classes: MA-C1 (other number types), MA-C2 (the rows beside the block at the price), MA-C3 (the parity rows
  and their readings), MA-C4 (a holder's class properties), MA-C5 (an expression's length and depth), MA-C6 (four
  answers out of their kind), and the check's citation entries (a citation of no texts).
- The door's cost, by construction: one judgement per frozen input keyed without pickling, a frozen block that cannot
  change in place, the domain one pass over a frozen block's leaves, a door's call bound without inspect.
- My own property over the declared domain: every argument and every field of every door crossed with every leaf
  type and extreme value of the domain is a named refusal or a value of its kind; every type outside the domain is
  refused by name at every door; every judged field of every shipped description, by load and by replace, likewise.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import json
import math
import pickle
import sys
from collections import OrderedDict
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import PHYSICS_CONSTANTS, get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.area_rule import (
    anchor, declaration, evidence, pricing, registry, rule, typerule, units as u)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import decomposition, parity
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration
from mimarsinan.deployment_record.platform_physics.converter import ConverterTable
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule._physics import priced_physics
from unit.deployment_record.area_rule.test_area_rule_door import (
    AREA, GENERIC, PKG, PROFILES, READOUT_BITS, _base_calls, _crash, _price, _shipped, _valid)

CELL = "v('S-SRAM-10')"
DOOR, FIELDS, JUDGE = f"{PKG}.core.door", f"{PKG}.core.fields", f"{PKG}.core.judge"
OUTSIDE_WORDS = "outside the area rule's declared input domain"
LARGEST = sys.float_info.max


def _module(dotted):
    return importlib.import_module(dotted)


class _Duck:
    """An object of no declared class."""


class _PropHolder:
    """A holder whose every field the rule reads is a class property giving a text (MA-C4's holder)."""
    for _f in ("rows", "cols", "weight_bits", "axons", "neurons", "cell_bits", "count", "area_rule", "validity",
               "geometry", "node_nm", "core_mm2", "n_pub", "gamma", "organization", "own_share", "type", "scope",
               "outside", "array", "converters", "beside", "ratio", "eta", "published_mm2", "core_um2", "array_um2",
               "state_um2", "converters_um2", "cell_kind", "bitcell_um2", "converter_kind", "own_unit", "chip",
               "array_ratio", "family", "strict", "value", "label", "converter_um2", "type_default"):
        locals()[_f] = property(lambda self: "4")
    pooled = read_as_declared = logic_held_per_core = primary = model = scope_mismatch = False
    del _f


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


#: Values of no declared type: other number types, bytes, sets, mapping views, subclasses, duck and property holders.
OUTSIDE = (Fraction(4), Fraction(7, 2), Decimal("4"), complex(4, 0), np.int64(4), np.int64(2 ** 62), np.float64(4.0),
           np.float32(4.0), np.bool_(True), b"4", bytearray(b"4"), frozenset({4}), {4}, MappingProxyType({"x": 4}),
           OrderedDict(x=4), SimpleNamespace(), _Duck(), _PropHolder(), _ListSub([4]), _DictSub(x=4))
#: Every leaf type of the domain at its extremes, and small containers of them.
DOMAIN_VALUES = (None, True, False, 0, 1, -1, 2 ** 53, 2 ** 53 + 1, 10 ** 400, -(10 ** 400), 0.0, -0.0, 0.5, 5e-324,
                 sys.float_info.min, 1e-300, 1.5e157, 1e300, LARGEST, -LARGEST, math.inf, -math.inf, math.nan, "", "4",
                 "T1", [], [4], (), (4,), {}, {"x": 4})


def _type_name(value):
    return type(value).__qualname__


def _outside_refusal(fn):
    """The refusal ``fn`` raises, asserted a named refusal of the domain; its text."""
    with pytest.raises(AreaRuleRefusal) as caught:
        fn()
    return str(caught.value)


def _census(cores, rows, cols, bits, form="declared", cast=float):
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": bits, "tiles": cores}
    geometry = [CoreGeometry(rows, cols, cores)]
    withheld = None
    if form == "no_bits":
        values.pop("weight_bits")
        withheld = {"weight_bits": "core types of several widths (a test census)"}
    return Quantities({k: QuantityValue(cast(v), "static") for k, v in values.items()}, withheld), geometry


def _area(physics, quantities, geometry):
    """('priced', band) or ('refused', reason) of price_absolute's chip area."""
    priced = price_absolute(quantities, physics, geometry=geometry, readout_bits=READOUT_BITS)
    term = {t.name: t for t in priced.terms}.get(AREA)
    if term is None:
        return "refused", {r.name: r.reason for r in priced.refusals}.get(AREA)
    return "priced", (term.band.low, term.band.nominal, term.band.high)


def _read_fields(value):
    """The fields the rule reads off a holder: its table's (core/fields.py, by the holder's declared class)."""
    fields = _module(FIELDS)
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    holder = next((h for h, names in fields.CLASSES.items() if name in names), None)
    return set(fields.FIELDS[holder]) if holder is not None else None


def _field_changes(value, values, read_only=False):
    """(label, changed value) for each field of a structured value set to each of ``values``: a mapping's keys, a
    NamedTuple's and a dataclass's fields (``read_only``: those the judge's table says the rule reads), a block's
    holders' keys, a list's first element, a pair list's members."""
    out = []
    read = _read_fields(value) if read_only else None
    if isinstance(value, dict):
        for key in list(value)[:16]:
            out += [(f"[{key!r}]={v!r}"[:70], {**value, key: v}) for v in values]
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for name in value._fields:
            if read is not None and name not in read:
                continue
            out += [(f".{name}={v!r}"[:70], value._replace(**{name: v})) for v in values]
    elif isinstance(value, declaration.AreaRuleBlock):
        for holder in declaration.HOLDERS:
            held = getattr(value, holder)
            if isinstance(held, dict):
                for key in list(held):
                    out += [(f".{holder}[{key!r}]={v!r}"[:70], replace(value, **{holder: {**held, key: v}}))
                            for v in values]
        for name in ("type", "terms", "outside", "cell_kind"):
            out += [(f".{name}={v!r}"[:70], replace(value, **{name: v})) for v in values]
    elif dataclasses.is_dataclass(value) and not isinstance(value, type) and type(value).__name__ in (
            "AnchorLaw", "Row", "PlatformPhysics"):
        for f in dataclasses.fields(value):
            if f.name in ("source", "constants", "conversion_model", "external_components", "chip_components",
                          "system_energy_rows", "backing_store", "transfer_rows", "link_ends", "carry_buffer",
                          "format_version", "name", "display_name", "description_file"):
                continue
            if read is not None and f.name not in read:
                continue
            for v in values:
                try:
                    out.append((f".{f.name}={v!r}"[:70], replace(value, **{f.name: v})))
                except Exception:  # noqa: BLE001 - a value its own class refuses when made is no door's input
                    continue
    elif isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, tuple) and len(first) == 2 and not hasattr(first, "_fields"):
            count, core = first
            out += [(f"[0].count={v!r}"[:70], [(v, core), *value[1:]]) for v in values]
            out += [(f"[0].core{label}"[:70], [(count, c), *value[1:]])
                    for label, c in _field_changes(core, values, read_only)]
        elif dataclasses.is_dataclass(first) and type(first).__name__ == "CoreGeometry":
            for f in ("axons", "neurons", "count"):
                out += [(f"[0].{f}={v!r}"[:70], [replace(first, **{f: v}), *value[1:]]) for v in values]
        elif not dataclasses.is_dataclass(first):
            out += [(f"[0]={v!r}"[:70], [v, *value[1:]]) for v in values]
    return out


@pytest.fixture(scope="module")
def base_calls():
    """The door tests' captured base calls, and calls of my own that the shipped prices make too rarely to be
    captured (at most two per door): every cell kind at a cell's doors, every type at the type's doors."""
    calls = _base_calls()
    reg = registry.load_registry()
    core = _module(f"{PKG}.core.equation")
    kinds = sorted(reg.files["cell_kinds"]["kinds"])
    extra = {
        "core.cells.cell_area": [dict(registry=reg, kind=k, node_nm=22.0, excl=(), declared_um2=None, declared=False)
                                 for k in kinds],
        "core.cells.reads_sigma": [dict(registry=reg, kind=k) for k in kinds],
        "core.equation.resolved_cell": [dict(registry=reg, type_name=t, declared=core.Declared(28.0, k))
                                        for t in sorted(reg.types) for k in kinds[:1] + kinds[-3:]],
        "rule.type_value": [dict(registry=reg, type_name=t, name=n, excl=()) for t in sorted(reg.types)
                            for n in ("s", "gamma")],
    }
    for short, more in extra.items():
        calls.setdefault(f"{PKG}.{short}", []).extend(more)
    return calls


def _doors():
    return _module(DOOR).DOORS


def _call(qualname, args):
    fn = _doors()[qualname]
    args = dict(args)
    target = getattr(args.pop("self"), fn.__name__) if "self" in args else fn
    return target(**args)


# ---------------------------------------------------------------------------------------------------------- the domain
class TestTheDeclaredInputDomain:
    def test_the_domain_is_stated_once_in_the_doors_module(self):
        """Red on UB5's tree (no domain): the leaves' exact types, the containers and the holders' classes, the last
        read from the judge's table (one line per holder), every class named there a class of the toolchain."""
        door, fields = _module(DOOR), _module(FIELDS)
        assert door.LEAVES == frozenset({int, float, str, bool, type(None)})
        assert {list, tuple, dict} <= set(door.SEQUENCES) | set(door.MAPPINGS)
        for holder, names in fields.CLASSES.items():
            assert holder in fields.FIELDS, holder
            for name in names:
                module, _, qualname = name.rpartition(".")
                obj = importlib.import_module(module)
                for part in qualname.split("."):
                    obj = getattr(obj, part)
                assert isinstance(obj, type), name
        assert set(fields.BY_ATTRIBUTE) <= set(fields.CLASSES), set(fields.BY_ATTRIBUTE) - set(fields.CLASSES)

    @pytest.mark.parametrize("value", OUTSIDE, ids=[_type_name(v) for v in OUTSIDE])
    def test_a_value_outside_the_domain_is_refused_by_name_with_its_type_and_its_field(self, value):
        """Red on UB5's tree: a Fraction raises TypeError out of price_absolute (MA-C1), a numpy integer wraps, a
        property holder raises AttributeError (MA-C4); here each is a named refusal naming its type and its field."""
        physics = priced_physics(GENERIC)
        block = physics.area_rule
        changed = replace(physics, area_rule=replace(block, node={**block.node, "node_nm": value}))
        q, geometry = _census(100, 256, 256, 4)
        state, reason = _area(changed, q, geometry)
        assert state == "refused" and OUTSIDE_WORDS in reason and type(value).__qualname__ in reason, reason
        assert "area_rule.node.node_nm" in reason, reason

    def test_the_domain_is_the_doors_first_act_before_any_other_judge(self):
        """Red on UB5's tree: the block's judge refuses its type first; the domain refuses the cell bits first."""
        reg = registry.load_registry()
        block = replace(get_platform_physics(GENERIC).area_rule, type="no_such_type")
        text = _outside_refusal(lambda: typerule.price_type_rule(reg, block, 10, rule.CensusCore(256, 256, 4),
                                                                 Fraction(8), 8))
        assert OUTSIDE_WORDS in text and "cell_bits" in text and "Fraction" in text, text

    @pytest.mark.parametrize("cast", [np.int64, np.float64, np.float32, Fraction], ids=lambda c: c.__name__)
    def test_a_census_in_another_number_type_is_refused_by_name_at_the_price(self, cast):
        """Red on UB5's tree: a Fraction census raises TypeError (MA-C1), a numpy one prices (no caller of record
        hands one over: the domain refuses it by name)."""
        for name in ("loihi", GENERIC, "neurram"):
            physics = priced_physics(name)
            q, geometry = _census(4, 64, 512, 8, cast=cast)
            state, reason = _area(physics, q, geometry)
            assert state == "refused" and OUTSIDE_WORDS in reason and cast.__name__ in reason, (name, state, reason)
            q, _g = _census(4, 64, 512, 8)
            state, reason = _area(physics, q, [CoreGeometry(cast(64), cast(512), 4)])
            assert state == "refused" and OUTSIDE_WORDS in reason, (name, state, reason)

    def test_a_container_nested_beyond_the_domains_depth_is_refused_by_name(self):
        """Red on UB5's tree: a citation nested 5,000 lists deep is accepted (and a deeper one is a RecursionError
        waiting for the first walk); the domain bounds its nesting by a declared depth."""
        physics = priced_physics("loihi")
        deep = "a text"
        for _ in range(5000):
            deep = [deep]
        block = replace(physics.area_rule, organization={**physics.area_rule.organization, "citation": deep})
        q, geometry = _census(4, 64, 1024, 8)
        state, reason = _area(replace(physics, area_rule=block), q, geometry)
        assert state == "refused" and "nests deeper" in reason, (state, str(reason)[:200])

    @pytest.mark.parametrize("key", [("rows",), np.str_("extra"), Fraction(1), np.int64(3)], ids=_type_name)
    def test_a_mappings_key_outside_the_domain_is_refused_by_name(self, key):
        """The round's mutant h_map_keys (a mapping's keys not judged: an organization holding a key of another type is
        priced, the key never read): the domain judges a mapping's key as a leaf, by name and type."""
        core = _module(f"{PKG}.core.equation")
        organization = {"rows": 256, "cols": 256, "weight_bits_per_cell": 4, key: 1}
        text = _outside_refusal(lambda: core.cells_per_weight(organization, 8))
        assert "a key" in text and OUTSIDE_WORDS in text and type(key).__qualname__ in text, text


# ---------------------------------------------------------------------------------------------------------- MA-C1
class TestOtherNumberTypesMAC1:
    def test_the_judge_takes_exact_types_and_returns_them(self):
        """Red on UB5's tree: the judge admits every numbers.Real (a numpy integer and a Fraction pass a count)."""
        judge = _module(JUDGE)
        assert judge.judged("rows", 4, judge.COUNT) == 4 and type(judge.judged("rows", 4, judge.COUNT)) is int
        assert type(judge.judged("a size", 4.0, judge.POSITIVE)) is float
        for value in (np.int64(4), Fraction(4), Decimal("4"), np.float64(4.0), np.float32(4.0), np.bool_(True)):
            with pytest.raises(AreaRuleRefusal, match="rows"):
                judge.judged("rows", value, judge.COUNT)

    def test_the_census_cores_the_rule_makes_are_built_of_python_numbers(self):
        """A guard: the census cores census_types makes (declared core types, a pooled census, one read as declared)
        hold Python ints and floats only (built from the judge's returned values)."""
        q, geometry = _census(100, 256, 256, 4)
        own = get_platform_physics(GENERIC).validity.geometry
        pooled = Quantities({k: QuantityValue(float(v), "static") for k, v in {
            "cores_physical": 6, "cells_physical": 6 * 300, "neurons_physical": 6 * 7, "weight_bits": 4}.items()})
        bare = Quantities({k: QuantityValue(float(v), "static") for k, v in {"cores_physical": 8,
                                                                             "weight_bits": 4}.items()})
        for types in (pricing.census_types(q, geometry, 4.0), pricing.census_types(pooled, None, 4.0),
                      pricing.census_types(bare, None, 4.0, own)):
            for count, core in types:
                assert type(count) is float
                for value in core:
                    assert type(value) in (int, float, bool, type(None)), (core, type(value))

    @pytest.mark.parametrize("value", [Fraction(4), np.int64(2 ** 31 + 5), np.float32(4.0), Decimal("4")],
                             ids=lambda v: type(v).__name__)
    def test_every_judged_field_in_another_number_type_is_a_named_refusal_at_load_and_at_the_price(self, value):
        """Red on UB5's tree: TypeError at typerule.py:75, anchor.py:222 and :227, rule.py:137, core/equation.py:141
        (a Fraction's format), a numpy count that wraps; here a named refusal naming the type, by the same words at load
        and at the price."""
        defects = []
        for name, data in _shipped().items():
            if data["area_rule"]["terms"] == "complete":
                continue
            physics = priced_physics(name)
            block = data["area_rule"]
            paths = [("node", "node_nm")] + [("organization", k) for k in (block.get("organization") or {})
                                             if k in ("rows", "cols", "weight_bits_per_cell", "stored_bits_per_cell",
                                                      "bits_per_neuron", "columns_per_converter", "cells_per_weight")]
            for path in paths:
                mod = copy.deepcopy(data)
                mod["area_rule"][path[0]][path[1]] = value
                try:
                    profile_from_dict(mod)
                    load = "accepted"
                except Exception as caught:  # noqa: BLE001 - every exception is recorded
                    load = str(caught) if _crash(caught) is None else f"CRASH {_crash(caught)}"
                held = getattr(physics.area_rule, path[0])
                made = replace(physics, area_rule=replace(physics.area_rule, **{path[0]: {**held, path[1]: value}}))
                q, geometry = _census(4, 64, 256, 4)
                state, reason = _area(made, q, geometry)
                if OUTSIDE_WORDS not in load or state != "refused" or reason != load:
                    defects.append(f"{name} {'.'.join(path)}: load {load[:100]} | price {state} {str(reason)[:100]}")
            for key in ("axons", "neurons", "cell_bits"):
                geometry = physics.validity.geometry
                made = replace(physics, validity=replace(physics.validity, geometry=replace(geometry,
                                                                                            **{key: value})))
                q, _g = _census(8, 64, 64, 4, "no_bits")
                state, reason = _area(made, q, None)
                if state != "refused" or OUTSIDE_WORDS not in str(reason):
                    defects.append(f"{name} validity.geometry.{key}: price {state} {str(reason)[:120]}")
        assert not defects, defects[:12]

    @pytest.mark.parametrize("field, value, words", [
        ("cores_physical", 2.5, "is not a whole count above 0"), ("weight_bits", 0, "is not a whole count above 0"),
        ("neurons_physical", 0, "is not a whole count above 0"),
        ("cells_physical", -1, "is not a whole count of at least 0")], ids=lambda v: str(v))
    def test_the_census_quantities_are_judged_at_the_door_by_name(self, field, value, words):
        """The round's mutant h_census_quantity (the census's quantities not judged at the door: a census value out of
        its kind priced, or refused later in other words): the door's census judge names the field and its kind."""
        physics = priced_physics(GENERIC)
        q, geometry = _census(100, 256, 256, 4)
        values = {k: q.get(k).value for k in q.keys()}
        values[field] = float(value)
        changed = Quantities({k: QuantityValue(v, "static") for k, v in values.items()})
        state, reason = _area(physics, changed, geometry)
        assert state == "refused" and f"census: {field} " in str(reason) and words in str(reason), (state, reason)


# ---------------------------------------------------------------------------------------------------------- MA-C2
def _with_constant(physics, key):
    decl = {"nominal": 1.0, "unit": PHYSICS_CONSTANTS[key].display_unit, "evidence_kind": "estimated",
            "derivation": "a test row", "note": "a test row"}
    return replace(physics, constants={**physics.constants, key: constant_from_declaration(key, decl)})


class TestTheRowsBesideTheBlockMAC2:
    def test_the_rows_beside_the_block_are_judged_at_the_price_by_the_same_call_as_at_load(self):
        """Red on UB5's tree (the builder's mi-F1, the check's MA-C2): a description the loader refuses is priced
        when the same physics is made in code (loihi with area_global_fixed: (1.875, 1.875, 9.16) mm^2)."""
        checked, defects = 0, []
        for name, data in _shipped().items():
            physics = priced_physics(name)
            for key in ("area_global_fixed", "area_per_adc"):
                mod = copy.deepcopy(data)
                mod.setdefault("constants", {})[key] = {"nominal": 1.0, "unit": PHYSICS_CONSTANTS[key].display_unit,
                                                        "evidence_kind": "estimated", "citation": "a test row",
                                                        "derivation": "a test row", "note": "a test row"}
                try:
                    profile_from_dict(mod)
                    continue  # the loader accepts it: so may the price
                except AreaRuleRefusal as refused:  # the rule's beside judge: the price's words are the load's
                    load = str(refused)
                except Exception as caught:  # noqa: BLE001 - R5's own loader refuses first (a named refusal)
                    assert _crash(caught) is None, (name, key, _crash(caught))
                    load = None
                org = physics.area_rule.organization or {}
                cen = (4, org.get("rows", 256), org.get("cols", 256), org.get("weight_bits_per_cell", 4))
                state, reason = _price(_with_constant(physics, key), cen)
                checked += 1
                if state != "refused" or (load is not None and reason != load):
                    defects.append(f"{name} {key}: load '{str(load)[:90]}' | price {state} {str(reason)[:90]}")
        assert checked >= 9 and not defects, (checked, defects)

    def test_the_physics_fields_the_beside_judge_reads_are_in_the_judges_table(self):
        """Red on UB5's tree: the physics' constants and converter rows are read beside the block at load only."""
        fields = _module(FIELDS)
        assert {"area_rule", "validity", "constants", "converter"} <= set(fields.FIELDS["physics"])

    def test_the_beside_judge_at_the_price_follows_the_block_and_the_geometry_as_at_load(self):
        """A block the loader refuses by its own rule, beside a stated area_per_adc, is refused at the price by the
        block's words (the loader's order: the block, the geometry, then the rows beside it)."""
        physics, block = priced_physics(GENERIC), get_platform_physics(GENERIC).area_rule
        changed = replace(block, converter={"kind": "sar_bitcell_reusing"})
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(changed, registry.load_registry())
        state, reason = _price(_with_constant(replace(physics, area_rule=changed), "area_per_adc"), (100, 256, 256, 4))
        assert (state, reason) == ("refused", str(caught.value)), (state, reason)

    def test_the_rows_beside_the_block_are_in_the_declared_domain_at_every_price(self):
        """The round's mutants h_beside_constants, h_constants_every and h_beside_rows (the physics' constants not
        walked by the domain, or walked at its first price only, and its converter table's rows not walked): a physics
        constant whose nominal is a numpy float, also one set in place after a first price of the same physics, and a
        converter table row that is no object are refused by the domain, by name, at the price."""
        key = "area_global_fixed"
        stated = constant_from_declaration(key, {"nominal": 1.0, "unit": PHYSICS_CONSTANTS[key].display_unit,
                                                 "evidence_kind": "estimated", "derivation": "a test row",
                                                 "note": "a test row"})
        numpy_nominal = dataclasses.replace(stated, nominal=np.float64(stated.nominal))
        for name in ("loihi", GENERIC):
            physics = priced_physics(name)
            g = physics.validity.geometry
            q, geometry = _census(4, g.axons, g.neurons, 4)
            state, reason = _area(replace(physics, constants={**physics.constants, key: numpy_nominal}), q, geometry)
            assert state == "refused" and OUTSIDE_WORDS in str(reason) and f"constants.{key}.nominal" in str(reason), (
                name, state, reason)
            changed = replace(physics, constants={**physics.constants, key: stated})
            first = _area(changed, q, geometry)  # the outside part stated twice: its own rule's named refusal
            changed.constants[key] = numpy_nominal
            state, reason = _area(changed, q, geometry)
            assert state == "refused" and OUTSIDE_WORDS in str(reason), (name, first, state, reason)
            state, reason = _area(replace(physics, converter=ConverterTable(rows={8: [1, 2]})), q, geometry)
            assert state == "refused" and "converter.rows[8]" in str(reason) and "not an object" in str(reason), (
                name, state, reason)

    def test_a_converter_table_rows_area_is_read_beside_the_block_at_the_price(self):
        """The round's mutant h_area_stated_rows (a converter table row's area_per_adc not read beside the block):
        hermes_64's block names its converter kind; a converter table row stating an area_per_adc beside it is the
        converter's area stated twice, refused at the rule's price (R5's own converter path reads such a row first
        at price_absolute)."""
        physics = get_platform_physics("hermes_64")
        g = physics.validity.geometry
        q, geometry = _census(4, g.axons, g.neurons, 4)
        plain = pricing.rule_area(replace(physics, converter=ConverterTable(rows={8: {"bits": 8}})), q, geometry,
                                  READOUT_BITS)
        assert plain is not None and plain.refusal is None, plain
        stated = pricing.rule_area(replace(physics, converter=ConverterTable(rows={8: {"area_per_adc": 1.0}})), q,
                                   geometry, READOUT_BITS)
        assert stated is not None and "stated twice" in str(stated.refusal), stated


# ---------------------------------------------------------------------------------------------------------- MA-C3
def _row(**kw):
    base = dict(chip="x", type="T1", scope="die", primary=True, cls="c", model=False, scope_mismatch=False,
                published_mm2=1.0, ratio=1.5)
    return parity.Row(**{**base, **kw})


class TestTheParityRowsMAC3:
    def test_a_rows_readings_are_parity_readings_of_the_judges_table(self):
        """Red on UB5's tree: readings are judged only as 'a tuple' (AttributeError at parity.py:238; a reading of
        ratio NaN answers [nan]); the reading's fields are not in the judge's table."""
        fields = _module(FIELDS)
        assert set(fields.FIELDS["parity reading"]) == {"label", "converter_um2", "ratio", "type_default"}
        refused = dict(ratio=None, refused="r", refused_by="converter")
        for readings in ((1,), (parity.Reading("l", 1.0, math.nan, True),), (parity.Reading("l", 1.0, "1.0", True),),
                         (parity.Reading("l", -1.0, 1.0, True),), (parity.Reading(4, 1.0, 1.0, True),)):
            with pytest.raises(AreaRuleRefusal):
                _row(readings=readings, **refused)

    def test_a_priced_row_states_its_ratio(self):
        """Red on UB5's tree: a priced row without its ratio is made, and strict_form raises TypeError at :215."""
        with pytest.raises(AreaRuleRefusal, match="ratio"):
            _row(ratio=None)

    def test_strict_form_answers_its_readings_factor_as_none_where_no_row_is_refused_by_converter(self):
        """Red on UB5's tree: ValueError 'max() arg is an empty sequence' at parity.py:251."""
        got = parity.strict_form([_row()])
        assert got["readings_max_factor"] is None and got["n_refused_by_kind"] == 0
        assert "no headline row is refused by converter kind" in got["readings_max_factor_note"]

    def test_the_strict_rows_hold_one_headline_row_per_chip_and_a_type_default_reading_each(self):
        """Red on UB5's tree: ValueError at parity.py:247 (a priced and a converter-refused row of one chip), and an
        IndexError for a converter-refused row of no type-default reading."""
        refused = _row(ratio=None, refused="r", refused_by="converter",
                       readings=(parity.Reading("l", 1.0, 1.2, False),))
        with pytest.raises(AreaRuleRefusal, match="one headline row per chip"):
            parity.strict_form([_row(), refused])
        with pytest.raises(AreaRuleRefusal, match="type-default reading"):
            parity.strict_form([_row(), replace(refused, chip="y")])
        with pytest.raises(AreaRuleRefusal, match="one headline row per chip"):
            parity.strict_spread([_row(), _row(ratio=2.0)])

    def test_the_shipped_strict_form_is_unchanged(self):
        """A guard: the strict form of the shipped estimate (the parity capture's)."""
        got = parity.strict_form(parity.estimate(registry.load_registry(), "strict"))
        assert (got["n_priced"], got["n_refused_by_kind"]) == (8, 3)
        assert got["worst_factor"] == pytest.approx(2.7933384046867955, rel=1e-15)
        assert got["readings_max_factor"] == pytest.approx(2.3326377856973823, rel=1e-15)


# ---------------------------------------------------------------------------------------------------------- MA-C4
class _CoreWithProperty:
    cols, weight_bits, pooled, read_as_declared = 64, 4, False, False

    @property
    def rows(self):
        return math.nan


class TestTheTablesSkipMAC4:
    def test_the_tables_judge_skips_only_the_fields_it_marks_via(self):
        """Red on mutant d_C4_skip (the skip widened back to every class property) and on UB5's tree: a census core
        whose rows is a class property giving NaN is never judged (core/fields.py:230-231)."""
        fields = _module(FIELDS)
        with pytest.raises(AreaRuleRefusal, match=r"c\.rows nan"):
            fields.judged_holder(registry.load_registry(), "c", _CoreWithProperty(), "census core")

    def test_a_holder_of_no_declared_class_is_refused_by_name_at_every_door_that_takes_a_holder(self):
        """Red on UB5's tree: TypeError at core/equation.py:135 and rule.py:194, AttributeError at
        core/arguments.py:100 (MA-C4's 58 inputs)."""
        reg, gen = registry.load_registry(), priced_physics(GENERIC)
        q, geometry = _census(100, 256, 256, 4)
        holder = _PropHolder()
        calls = {
            "census core": lambda: typerule.price_type_rule(reg, gen.area_rule, 10, holder, 8, 8),
            "physics": lambda: pricing.rule_price(holder, q, geometry, 8),
            "core terms": lambda: parity.scope_mm2(reg, reg, dict(reg.published_points()[0]),
                                                   reg.published_points()[0]["scopes"][0], holder),
            "law": lambda: anchor.price_anchor(holder, 4),
            "declaration": lambda: typerule.core_terms(reg, "T1", holder, typerule.Counts(65536.0)),
            "counts": lambda: typerule.core_terms(reg, "T1", typerule.Declared(28.0), holder),
            "core type": lambda: pricing.census_types(q, [holder], 4),
        }
        for label, fn in calls.items():
            text = _outside_refusal(fn)
            assert OUTSIDE_WORDS in text and "_PropHolder" in text, (label, text)


# ---------------------------------------------------------------------------------------------------------- MA-C5
#: Beyond the bounds: deep ones, and a shallow one of 3,005 characters (the round's mutant h_expr_length: a length
#: unbounded, the shallow text reached the parser).
DEEP = ("-" * 20000 + CELL, "+".join([CELL] * 4000), "(" * 150 + CELL + ")" * 150, "-" * 100 + CELL,
        "v('" + "S" * 3000 + "')")
#: The keys under which the registry's data, the published points and a block state an expression over the evidence.
EXPRESSION_KEYS = ("expression", "fallback", "um2", "published_mm2", "count", "sub_arrays", "memory_bits", "cells",
                   "arrays", "array_cells", "bits", "per_array", "bitcell_um2", "core_mm2", "die_mm2", "n_pub")


class TestTheExpressionsBoundsMAC5:
    def test_the_bounds_are_named_limits_and_every_shipped_expression_lies_within_them(self):
        """Red on UB5's tree (no bound): the limits are named in units.py; every expression of the registry's data, of
        the published points and of the shipped descriptions lies within both."""
        assert u.EXPRESSION_LENGTH >= 1024 and u.EXPRESSION_DEPTH >= 32
        reg = registry.load_registry()
        texts = []

        def visit(obj):
            if isinstance(obj, dict):
                for key, v in obj.items():
                    if key in EXPRESSION_KEYS and isinstance(v, str):
                        texts.append(v)
                    elif key == "bits" and isinstance(v, list):
                        texts.extend(b for b in v if isinstance(b, str))
                    else:
                        visit(v)
            elif isinstance(obj, list):
                for v in obj:
                    visit(v)
        for doc in list(reg.files.values()) + list(reg.points):
            visit(doc)
        for data in _shipped().values():
            visit(data["area_rule"])
        assert len(texts) > 150
        for text in texts:
            length, depth = evidence.expression_size(text)
            assert length <= u.EXPRESSION_LENGTH and depth <= u.EXPRESSION_DEPTH, (text, length, depth)

    @pytest.mark.parametrize("expression", DEEP, ids=["unary_20000", "sum_4000", "parens_150", "unary_100",
                                                      "long_3005"])
    def test_an_expression_beyond_the_bounds_is_refused_by_name_before_it_is_parsed(self, expression, monkeypatch):
        """Red on UB5's tree: MemoryError at evidence.py:63 for 20,000 signs (at load and at the price); a
        RecursionError named for the others, after the parse. Here every entry refuses it by its declared bound and
        the parser never sees it."""
        real = ast.parse

        def guarded(source, *a, **k):
            assert source != expression, "the parser met an expression beyond the declared bounds"
            return real(source, *a, **k)
        monkeypatch.setattr(evidence.ast, "parse", guarded)
        reg, gen = registry.load_registry(), priced_physics(GENERIC)
        core = _module(f"{PKG}.core.equation")
        block = replace(gen.area_rule, node={**gen.area_rule.node, "bitcell_um2": expression})
        data = json.loads((PROFILES / f"{GENERIC}.json").read_text())
        data["area_rule"]["node"]["bitcell_um2"] = expression
        loihi = json.loads((PROFILES / "loihi.json").read_text())
        loihi["area_rule"]["anchor"]["core_mm2"] = expression
        entries = {"evaluate": lambda: reg.evaluate(expression),
                   "cited": lambda: core.cited(reg, "x", expression, core.CELL, "a declared cell"),
                   "judge_block": lambda: declaration.judge_block(block, reg),
                   "load": lambda: profile_from_dict(data), "load anchor": lambda: profile_from_dict(loihi)}
        for label, fn in entries.items():
            text = _outside_refusal(fn)
            assert "longer than" in text or "deeper than" in text, (label, text[-200:])
        state, reason = _price(replace(gen, area_rule=block), (100, 256, 256, 4))
        assert state == "refused" and ("longer than" in reason or "deeper than" in reason), (state, reason[-200:])


# ---------------------------------------------------------------------------------------------------------- MA-C6
class TestAnswersOfTheirKindMAC6:
    def test_a_cells_area_is_judged_where_cell_area_makes_it(self):
        """Red on UB5's tree: cell_area answers infinity or 0.0 (the check's 30 inputs): an F2 kind at the floats'
        ends of the node, a kind over sigma whose factor exceeds 1 at a declared cell of the largest float. Each is a
        named refusal naming the cell, or a finite cell above 0."""
        reg = registry.load_registry()
        kinds = reg.files["cell_kinds"]["kinds"]
        cases = [(k, n, None) for k, spec in kinds.items() if "F2" in spec
                 for n in (1.5e157, 1e300, LARGEST, 1e-310, 2.2e-308)]
        cases += [(k, 22.0, LARGEST) for k, spec in kinds.items() if "F2" not in spec and "um2" not in spec]
        refused = 0
        for kind, node, declared in cases:
            try:
                got = rule.cell_area(reg, kind, node, (), declared, False)
            except AreaRuleRefusal as caught:
                assert "cell" in str(caught), str(caught)
                refused += 1
                continue
            assert math.isfinite(got) and got > 0, (kind, node, declared, got)
        assert refused >= 14, refused

    def test_a_stacks_array_totals_and_ratio_are_judged_where_they_are_made(self):
        """Red on UB5's tree: a figure row of count the largest float makes an infinite stack."""
        reg = registry.load_registry()
        rows = parity.estimate(reg, "held_out")
        figure = next(p for p in reg.published_points() if p.get("array_figure"))
        changed = [replace(r, count=LARGEST) if (r.chip, r.scope) == (figure["name"], figure["array_figure"]["scope"])
                   else r for r in rows]
        with pytest.raises(AreaRuleRefusal, match="stack"):
            decomposition.array_decomposition(reg, changed)

    def test_a_scopes_estimate_is_judged_where_scope_mm2_makes_it(self):
        """Red on UB5's tree: terms.core_um2 at the largest float gives an infinite estimate."""
        reg = registry.load_registry()
        point = next(p for p in reg.published_points() if any(s.get("gamma") for s in p["scopes"]) and not any(
            s.get("sub_arrays") for s in p["scopes"]))
        scope = {**next(s for s in point["scopes"] if s.get("gamma")), "count": "4096 * 4096"}
        terms = rule.CoreTerms(LARGEST, 1.0, 0.0, 0.0)
        with pytest.raises(AreaRuleRefusal, match="estimate"):
            parity.scope_mm2(reg, reg, point, scope, terms)

    def test_a_sub_array_scopes_estimate_is_judged_where_scope_mm2_makes_it(self, monkeypatch):
        """The round's mutant h_scope_estimate (a sub-array scope's estimate not judged): a block below the core whose
        terms sum beyond the floats is a named refusal of its estimate, never an infinite size."""
        reg = registry.load_registry()
        point = next(p for p in reg.published_points() if any(s.get("sub_arrays") for s in p["scopes"]))
        scope = next(s for s in point["scopes"] if s.get("sub_arrays"))
        assert parity.scope_mm2(reg, reg, point, scope, rule.CoreTerms(1.0, 1.0, 0.0, 0.0))[0] > 0
        monkeypatch.setattr(parity.typerule, "core_terms", lambda *a, **k: rule.CoreTerms(LARGEST, LARGEST, LARGEST,
                                                                                          0.0))
        with pytest.raises(AreaRuleRefusal, match="estimate"):
            parity.scope_mm2(reg, reg, point, scope, rule.CoreTerms(1.0, 1.0, 0.0, 0.0))

    @pytest.mark.parametrize("expression", [f"{CELL} / 1e-310", f"{CELL} * 1e308 * 1e10", "1e308 * 10 - 1e308 * 10"])
    def test_the_registrys_arithmetic_names_a_value_beyond_the_floats(self, expression):
        """Red on UB5's tree: Registry.evaluate answers infinity (and NaN)."""
        with pytest.raises(AreaRuleRefusal, match="registry expression"):
            registry.load_registry().evaluate(expression)

    def test_a_citation_is_judged_as_texts(self):
        """Red on UB5's tree (the check's 54 citation entries): an organization's citation holding infinity is
        carried into the anchor's law."""
        reg = registry.load_registry()
        block = get_platform_physics("loihi").area_rule
        for citation in ({**block.organization["citation"], "rows": math.inf}, [4], {"rows": None}, ""):
            changed = replace(block, organization={**block.organization, "citation": citation})
            with pytest.raises(AreaRuleRefusal, match="cit"):
                anchor.anchor_law(reg, changed)
        assert anchor.anchor_law(reg, block).organization["citation"] == block.organization["citation"]


# ---------------------------------------------------------------------------------------------------------- the cost
class TestTheDoorsCostByConstruction:
    def test_a_price_pickles_nothing(self, monkeypatch):
        """Red on UB5's tree: the door's memo keys a block by its pickled text at every price."""
        physics = priced_physics("loihi")
        _price(physics, (4, 64, 1024, 8))
        calls, real = [], pickle.dumps
        monkeypatch.setattr(pickle, "dumps", lambda *a, **k: calls.append(1) or real(*a, **k))
        for _ in range(3):
            assert _price(physics, (4, 64, 1024, 8))[0] == "priced"
        assert not calls, len(calls)

    def test_a_frozen_block_cannot_change_in_place_and_a_replaced_one_is_judged_again(self):
        """Red on UB5's tree: a block's holder changes in place (the door then sees a new text)."""
        physics = priced_physics("odin")
        block = replace(physics.area_rule, **{h: copy.deepcopy(dict(getattr(physics.area_rule, h))) for h in
                                              declaration.HOLDERS if getattr(physics.area_rule, h) is not None})
        with pytest.raises(TypeError):
            block.organization["rows"] = 0
        with pytest.raises(TypeError):
            block.organization["citation"].update({"x": "y"})
        with pytest.raises(TypeError):
            block.anchor["n_pub"] = "1"
        changed = replace(physics, area_rule=replace(block, organization={**block.organization, "rows": 0}))
        assert _price(physics, (1, 256, 256, 4))[0] == "priced"
        assert _price(changed, (1, 256, 256, 4))[0] == "refused"

    def test_a_frozen_blocks_text_digest_and_copies_are_its_plain_forms(self):
        """A guard: a frozen block equals its plain form, writes the same text and digest, and pickles and copies."""
        for name, data in _shipped().items():
            block = get_platform_physics(name).area_rule
            plain = declaration.AreaRuleBlock.from_dict(json.loads(json.dumps(data["area_rule"])))
            assert block == plain, name
            assert json.dumps(block.to_dict(), sort_keys=True) == json.dumps(plain.to_dict(), sort_keys=True)
            assert json.loads(json.dumps(block.to_dict())) == json.loads(json.dumps(plain.to_dict()))
            assert declaration.block_digest(block) == declaration.block_digest(plain)
            for twin in (pickle.loads(pickle.dumps(block)), copy.deepcopy(block), copy.copy(block)):
                assert twin == block and declaration.block_digest(twin) == declaration.block_digest(block)

    def test_a_doors_call_is_bound_without_inspect(self, monkeypatch):
        """Red on UB5's tree: every outermost door call binds its arguments by inspect.Signature.bind."""
        import inspect
        physics = priced_physics(GENERIC)

        def refused(*_a, **_k):
            raise AssertionError("inspect.Signature.bind at a door")
        monkeypatch.setattr(inspect.Signature, "bind", refused)
        assert _price(physics, (100, 256, 256, 4))[0] == "priced"
        reg = registry.load_registry()
        with pytest.raises(TypeError):
            rule.type_value(reg, "T1")  # a call a door cannot bind is Python's own TypeError
        with pytest.raises(TypeError):
            rule.type_value(reg, "T1", "s", (), "one too many")

    def test_the_domain_walks_a_frozen_blocks_leaves_once(self, monkeypatch):
        """Red on UB5's tree (no domain): three prices of one physics walk its block's leaves once."""
        door = _module(DOOR)
        physics = priced_physics("loihi")
        walked, real = [], door.walk_holder
        monkeypatch.setattr(door, "walk_holder", lambda value, *a, **k: (walked.append(value) if isinstance(
            value, declaration.AreaRuleBlock) else None, real(value, *a, **k))[1])
        door.forget_domain()
        for _ in range(3):
            assert _price(physics, (4, 64, 1024, 8))[0] == "priced"
        assert len([v for v in walked if v is physics.area_rule]) == 1, len(walked)

    def test_a_frozen_holders_remembered_pass_holds_only_for_the_object_it_walked(self, monkeypatch):
        """The round's mutant h_frozen_alive (a frozen holder's pass trusted by its identity alone): where two blocks
        take one identity (an identity reused after its block is gone, made here by one identity for every object),
        the domain walks the second afresh and refuses it by name."""
        door = _module(DOOR)
        good = priced_physics("loihi").area_rule
        bad = replace(good, node={**good.node, "node_nm": np.float64(22.0)})
        monkeypatch.setattr(door, "id", lambda _obj: 0, raising=False)
        door.forget_domain()
        try:
            door.in_domain(good, ("area_rule",))
            text = _outside_refusal(lambda: door.in_domain(bad, ("area_rule",)))
            assert OUTSIDE_WORDS in text and "node_nm" in text and "float64" in text, text
        finally:
            door.forget_domain()

    def test_a_remembered_outcome_holds_only_for_the_objects_it_was_made_for(self, monkeypatch):
        """The round's mutant h_memo_alive (an outcome trusted by its objects' identities alone): where two blocks take
        one identity, the memo makes the second block's outcome afresh."""
        memo = _module(f"{PKG}.core.memo")
        reg = registry.load_registry()
        first, second = priced_physics("loihi").area_rule, priced_physics("odin").area_rule
        monkeypatch.setattr(memo, "id", lambda _obj: 0, raising=False)
        try:
            assert memo.remembered(reg, "a test's outcome", (first,), lambda: "first") == "first"
            assert memo.remembered(reg, "a test's outcome", (second,), lambda: "second") == "second"
            assert memo.remembered(reg, "a test's outcome", (second,), lambda: "made again") == "second"
        finally:
            memo.forget(reg)

    def test_a_doors_defaults_are_bound_and_judged(self, monkeypatch):
        """The round's mutant h_bind_default (a default left unbound, so never judged): the door judges every argument
        the call binds, a default with it."""
        door = _module(DOOR)
        seen, real = [], door.judged_input
        monkeypatch.setattr(door, "judged_input", lambda args, *a, **k: seen.append(dict(args)) or real(args, *a, **k))
        q, geometry = _census(100, 256, 256, 4)
        pricing.rule_price(priced_physics(GENERIC), q, geometry)
        assert seen and "readout_bits" in seen[0] and seen[0]["readout_bits"] is None, seen

    def test_a_call_a_door_cannot_bind_is_refused_before_any_judge(self, monkeypatch):
        """The round's mutant h_bind_positional (a call of one positional argument too many bound by its first ones and
        judged, Python's refusal coming only at the body): a call the door's signature cannot bind is Python's own
        TypeError before any judge runs."""
        door = _module(DOOR)
        seen, real = [], door.judged_input
        monkeypatch.setattr(door, "judged_input", lambda args, *a, **k: seen.append(set(args)) or real(args, *a, **k))
        reg = registry.load_registry()
        with pytest.raises(TypeError):
            rule.type_value(reg, "T1", "s", (), "one too many")
        assert not seen, seen

    def test_a_frozen_blocks_lists_cannot_change_in_place(self):
        """The round's mutant h_freeze_lists (a block's lists left plain): a list a block holds (truenorth's held-logic
        citation) is frozen with the block."""
        data = json.loads((PROFILES / "truenorth.json").read_text())
        block = declaration.AreaRuleBlock.from_dict(data["area_rule"])
        cited = block.anchor["logic_held_per_core"]["citation"]
        assert isinstance(cited, list) and cited, cited
        with pytest.raises(TypeError):
            cited.append("another id")

    def test_a_converters_unit_is_remembered_per_readout_width(self):
        """The round's mutant h_converter_memo (a converter's unit remembered for every readout width): a converter
        kind bound to 8 bits answers its unit at 8 bits and its bound's named refusal at 10, in either order."""
        core = _module(f"{PKG}.core.equation")
        reg = registry.load_registry()
        declared = core.Declared(22.0, None, None, "sar_bitcell_reusing", None)
        for widths in ((8, 10), (10, 8)):
            _module(DOOR).forget(reg)
            for bits in widths:
                if bits == 8:
                    assert core.converter_um2(reg, "T2_digital", declared, bits) > 0
                else:
                    with pytest.raises(AreaRuleRefusal, match="bound"):
                        core.converter_um2(reg, "T2_digital", declared, bits)


# ---------------------------------------------------------------------------------------------------------- property
class TestMyPropertyOverTheDeclaredDomain:
    @pytest.mark.parametrize("module", ["core.equation", "core.cells", "rule", "anchor", "typerule", "pricing",
                                        "declaration", "published.parity", "published.decomposition", "registry"])
    def test_every_argument_and_field_with_every_leaf_and_extreme_is_a_named_refusal_or_a_value_of_its_kind(
            self, module, base_calls):
        """Red on UB5's tree: answers of infinity or 0 (MA-C6), ValueError and AttributeError at the parity doors
        (MA-C3); here every argument and every field of every door, crossed with every leaf type of the domain at its
        extremes, is a named refusal or a value of its kind (every number finite and not negative, a band positive
        and ordered, a positive kind's answer above 0)."""
        doors = {q: fn for q, fn in _doors().items() if fn.__module__ == f"{PKG}.{module}"}
        assert doors and all(base_calls.get(q) for q in doors), [q for q in doors if not base_calls.get(q)]
        tried, defects = 0, []
        for qualname, _fn in doors.items():
            for base in base_calls[qualname]:
                for param, value in base.items():
                    if param in ("self", "cls"):
                        continue
                    variants = [(repr(v)[:40], v) for v in DOMAIN_VALUES] + _field_changes(value, DOMAIN_VALUES)
                    for label, variant in variants:
                        tried += 1
                        try:
                            got = _call(qualname, {**base, param: variant})
                        except Exception as caught:  # noqa: BLE001 - every exception is recorded
                            if _crash(caught) is not None:
                                defects.append(f"{qualname}({param} {label}): {_crash(caught)[:140]}")
                            continue
                        if not _valid(got) or not _positive_answer(qualname, got):
                            defects.append(f"{qualname}({param} {label}): an answer of no kind {repr(got)[:120]}")
        assert tried > 100 and not defects, (tried, len(defects), defects[:25])

    @pytest.mark.parametrize("module", ["core.equation", "core.cells", "rule", "anchor", "typerule", "pricing",
                                        "declaration", "published.parity", "published.decomposition", "registry"])
    def test_every_type_outside_the_domain_is_refused_by_name_at_every_door(self, module, base_calls):
        """Red on UB5's tree (no domain): TypeError, AttributeError, prices of numpy and Fraction values."""
        doors = {q: fn for q, fn in _doors().items() if fn.__module__ == f"{PKG}.{module}"}
        tried, defects = 0, []
        for qualname, _fn in doors.items():
            for base in base_calls[qualname]:
                for param, value in base.items():
                    if param in ("self", "cls"):
                        continue
                    variants = [(repr(v)[:40], v) for v in OUTSIDE] + _field_changes(value, OUTSIDE, True)
                    for label, variant in variants:
                        tried += 1
                        try:
                            _call(qualname, {**base, param: variant})
                            defects.append(f"{qualname}({param} {label}): answered")
                        except AreaRuleRefusal as refused:
                            if OUTSIDE_WORDS not in str(refused):
                                defects.append(f"{qualname}({param} {label}): another refusal {str(refused)[:100]}")
                        except Exception as caught:  # noqa: BLE001
                            defects.append(f"{qualname}({param} {label}): {type(caught).__name__} {caught}"[:160])
        assert tried > 100 and not defects, (tried, len(defects), defects[:25])

    @pytest.mark.parametrize("name", sorted(n for n, d in _shipped().items() if d["area_rule"]["terms"] != "complete"))
    def test_every_judged_field_of_a_description_by_load_and_by_replace_is_refused_or_priced_alike(self, name):
        """Red on UB5's tree: a description refused at load is priced when its physics is made in code (MA-C2), and
        a Fraction field raises TypeError (MA-C1). Here every judged field of the block, set to every leaf type of the
        domain at its extremes and to every type outside it, is a named refusal or a priced band, the same by load
        and by replace (never refused at load and priced by replace)."""
        data = _shipped()[name]
        physics = priced_physics(name)
        block = data["area_rule"]
        org = block.get("organization") or {}
        cens = [(4, org.get("rows", 256), org.get("cols", 256), org.get("weight_bits_per_cell", 4)),
                (100, 128, 128, 4, "no_bits")]
        paths = [(k,) for k in ("type", "terms", "outside", "cell_kind")] + [
            (h, k) for h in declaration.HOLDERS for k in (block.get(h) or {})]
        defects, tried = [], 0
        for path in paths:
            for value in DOMAIN_VALUES + OUTSIDE[:8]:
                mod = copy.deepcopy(data)
                holder = mod["area_rule"]
                for key in path[:-1]:
                    holder = holder[key]
                holder[path[-1]] = value
                try:
                    profile_from_dict(mod)
                    load = "accepted"
                except Exception as caught:  # noqa: BLE001
                    load = "refused" if _crash(caught) is None else f"CRASH {_crash(caught)}"
                b = physics.area_rule
                try:
                    made_block = replace(b, **{path[0]: value}) if len(path) == 1 else replace(
                        b, **{path[0]: {**getattr(b, path[0]), path[1]: value}})
                except AreaRuleRefusal:
                    continue
                states = [_price(replace(physics, area_rule=made_block), cen)[0] for cen in cens]
                tried += 1
                if load.startswith("CRASH") or any(s not in ("priced", "refused") for s in states) or (
                        load == "refused" and "priced" in states):
                    defects.append(f"{'.'.join(path)}={value!r}: load {load[:80]} | price {states}")
        for key in ("area_global_fixed", "area_per_adc"):  # the rows beside the block (MA-C2)
            for nominal in (1.0, 0.0, 1e300, LARGEST):
                mod = copy.deepcopy(data)
                mod.setdefault("constants", {})[key] = {"nominal": nominal, "unit": PHYSICS_CONSTANTS[key].display_unit,
                                                        "evidence_kind": "estimated", "citation": "a test row",
                                                        "derivation": "a test row", "note": "a test row"}
                try:
                    profile_from_dict(mod)
                    load = "accepted"
                except Exception as caught:  # noqa: BLE001
                    load = "refused" if _crash(caught) is None else f"CRASH {_crash(caught)}"
                made = _with_constant(physics, key)
                made = replace(made, constants={**made.constants, key: replace(made.constants[key], nominal=nominal,
                                                                               low=nominal, high=nominal)})
                states = [_price(made, cen)[0] for cen in cens]
                tried += 1
                if load.startswith("CRASH") or any(s not in ("priced", "refused") for s in states) or (
                        load == "refused" and "priced" in states):
                    defects.append(f"constants.{key}={nominal!r}: load {load[:80]} | price {states}")
        assert tried > 100 and not defects, (tried, defects[:15])


#: The doors whose answer is of a positive kind (a cell, a unit, a share, a factor, a sigma, a count of cells).
POSITIVE_ANSWERS = {"core.cells.logic_rule_factor", "core.cells.cell_factor", "core.cells.cell_area",
                    "core.equation.grid_share", "core.equation.core_cell_um2", "core.equation.converter_um2",
                    "core.equation.type_eta", "anchor.own_die_gamma", "registry.Registry.sigma", "rule.type_value",
                    "rule.type_gamma",
                    "published.parity.factor", "published.parity.die_gamma", "core.equation.cells_per_weight",
                    "anchor.device_rows"}


def _positive_answer(qualname, got):
    short = qualname[len(PKG) + 1:]
    if short not in POSITIVE_ANSWERS:
        return True
    scalar = got.um2 if hasattr(got, "um2") else got
    return isinstance(scalar, (int, float)) and not isinstance(scalar, bool) and math.isfinite(scalar) and scalar > 0
