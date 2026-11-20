"""[U-AREA] The second delta round (UB3/reverify.md: MA-D1 (a) to (d), mi-D1, mi-D2): one judge of the declared
quantities, tests first. Each test is red on UB3's tree (UB3/uarea.patch on R5, applied digest 0cd64c11) or on its
named mutant (UB4/scripts/mut_r4.py), green on the round's tree; a guard (green on UB3's tree too) is named as such.

- The class 'a quantity the area rule reads from a declaration is never judged as a quantity of its kind', closed by a
  property: over every shipped anchor and the generic type rule (and that rule under every type), every field the
  rule's code reads off a declaration (found by an AST sweep of the code's own reads, never a list typed here) and a
  fixed list of bad and edge values, and over every cell kind of the registry with every own-share basis and an own
  share below 1, at 1 and above 1, the load judge and price_absolute give a named refusal or a finite band with
  0 < low <= nominal <= high. Never another exception.
- The judge: its kinds, its refusal naming the field and the value, one judge at load and at the price, N-3's path for
  an own share outside (0, 1], the own die's grid share, a type rule's organization and reading at load, the rule's
  own band, the names the equation reads and the objects that hold the quantities.
- mi-D1: the declared-cell flag only where the resolved cell read the declared cell.
- mi-D2: an anchor priced without rows or columns; a cells type rule without columns_per_converter.
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mimarsinan.common.refusal import is_toolchain_refusal
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import (
    anchor, declaration, pricing, registry, rule, typerule)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule._physics import priced_physics

AREA = "chip_area_mm2"
AR = Path(registry.__file__).resolve().parent
#: The one judge's module (core/equation.py); its names are absent on UB3's tree, so the tests that read them are red
#: there by their absence.
CORE = "mimarsinan.deployment_record.platform_physics.area_rule.core.equation"
ANCHORS = ("truenorth", "loihi", "odin", "hermes_64", "neurram", "sram_cim_macro")
GENERIC = "generic_estimated_22nm"
READOUT_BITS = 8
#: The fixed list of bad and edge values (the ruling, MA-D1 (b)): 0, -1, a text, None, a bool, infinity, not-a-number,
#: a list.
BAD = (0, -1, "4", None, True, math.inf, math.nan, [4])
#: The same values stated as expressions over the evidence (a field the rule evaluates), each beside a field's own
#: expression ``e``: zero, negative, infinite, not-a-number, a division by zero, and a text that is no expression.
EXPRESSION_FORMS = ("0 * {e}", "-1 * {e}", "1e999 * {e}", "(1e999 - 1e999) * {e}", "{e} / 0", "{e} +",
                    "a text that is no expression")
#: A declared cell the evidence holds (the node series' 10 nm cell), the base of a cell expression and of the
#: re-verify's cited cell of eight node cells.
CELL = "v('S-SRAM-10')"

# ---------------------------------------------------------------------------------------------------------- the fields
#: Each holder of declared fields: the rule's modules that read it and the source texts the holder has there.
HOLDERS = {
    "organization": (("core/equation.py", "anchor.py"), {"org", "organization"}),
    "node": (("declaration.py",), {"self.node", "self.node or {}", "node", "block.node or {}"}),
    "anchor": (("anchor.py", "declaration.py"), {"a", "anchor", "block.anchor"}),
    "own_share": (("anchor.py", "declaration.py"), {"spec", "share"}),
    "converter": (("declaration.py", "pricing.py"), {"converter", "block.converter or {}"}),
    "reading_low_edge": (("typerule.py",), {"block.reading_low_edge or {}", "reading"}),
    "published_size": (("pricing.py", "declaration.py"), {"block.published_size", "size"}),
}


def _reads(holder):
    """The string keys the rule's code reads off ``holder`` (``X[key]``, ``X.get(key)``, ``key in X``): the fields are
    the code's own reads, never typed here (a field the code reads later enters the property with no edit)."""
    files, texts = HOLDERS[holder]
    keys = set()
    for name in files:
        text = (AR / name).read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and \
                    isinstance(node.slice.value, str):
                held, key = node.value, node.slice.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and \
                    node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                held, key = node.func.value, node.args[0].value
            elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and \
                    isinstance(node.left.value, str) and isinstance(node.ops[0], (ast.In, ast.NotIn)):
                held, key = node.comparators[0], node.left.value
            else:
                continue
            if ast.get_source_segment(text, held) in texts:
                keys.add(key)
    return keys


def _examples():
    """Each field's shipped expression (a text that reads the evidence), the base of its expression forms."""
    out = {("node", "bitcell_um2"): CELL}
    for name in ANCHORS:
        block = get_platform_physics(name).area_rule
        for holder, held in (("anchor", block.anchor), ("own_share", block.anchor.get("own_share"))):
            for key, value in (held or {}).items():
                if isinstance(value, str) and "(" in value:
                    out.setdefault((holder, key), value)
    return out


def _values(holder, key, examples):
    forms = tuple(f.format(e=examples[(holder, key)]) for f in EXPRESSION_FORMS) if (holder, key) in examples else ()
    return BAD + forms


# ---------------------------------------------------------------------------------------------------------- outcomes
def _census(cores, rows, cols, weight_bits):
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": weight_bits, "tiles": cores}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _crash(caught):
    return None if is_toolchain_refusal(caught) and str(caught).strip() else f"{type(caught).__name__}: {caught}"


def _load(source, reg):
    """The load judge (AreaRuleBlock.from_dict and judge_block, as the loader runs them): None, or the crash."""
    try:
        declaration.judge_block(declaration.AreaRuleBlock.from_dict(source), reg)
    except Exception as caught:  # noqa: BLE001 - the property records any other exception
        return _crash(caught)
    return None


def _price(physics, make_block, cores, rows, cols, weight_bits):
    """price_absolute on ``physics`` with the block ``make_block()`` (a block that cannot be made is refused by name):
    ('refused' | 'priced' | the defect, the band)."""
    try:
        p = replace(physics, area_rule=make_block())
        priced = price_absolute(_census(cores, rows, cols, weight_bits), p,
                                geometry=[CoreGeometry(rows, cols, cores)], readout_bits=READOUT_BITS)
    except Exception as caught:  # noqa: BLE001
        crash = _crash(caught)
        return ("refused", None) if crash is None else (f"CRASH {crash}", None)
    term = {t.name: t for t in priced.terms}.get(AREA)
    if term is None:
        refusal = {r.name: r.reason for r in priced.refusals}.get(AREA)
        return ("refused", None) if refusal and refusal.strip() else ("NO NAMED REFUSAL", None)
    band = (term.band.low, term.band.nominal, term.band.high)
    good = all(math.isfinite(x) for x in band) and 0.0 < band[0] <= band[1] <= band[2]
    return ("priced" if good else f"BAD BAND {band}"), band


def _changed(held, key, value):
    return {**(held or {}), key: value}


def _defects(cases):
    return [f"{label}: load {load}, price {price}" for label, load, price in cases
            if load is not None or price not in ("refused", "priced")]


# ---------------------------------------------------------------------------------------------------------- MA-D1 (b)
class TestTheClassPropertyMAD1:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_every_field_the_rule_reads_off_an_anchor_with_every_bad_value(self, name):
        """Red on UB3's tree: ZeroDivisionError, TypeError, AttributeError, ValueError out of the load judge and
        price_absolute, and negative or non-finite bands."""
        reg, examples = registry.load_registry(), _examples()
        physics = priced_physics(name)
        block = physics.area_rule
        org = block.organization
        source = dict(block.to_dict())
        cases = []
        for holder in ("organization", "node", "anchor", "own_share", "converter"):
            held = block.anchor.get("own_share") if holder == "own_share" else getattr(block, holder)
            # the anchor's figures are also read by name from the code's own table (declaration.ANCHOR_FIGURES)
            keys = _reads(holder) | (set(declaration.ANCHOR_FIGURES) if holder == "anchor" else set())
            for key in sorted(keys) + [None]:
                for value in _values(holder, key, examples) if key is not None else BAD:
                    new = value if key is None else _changed(held, key, value)
                    where = holder if holder != "own_share" else "anchor"
                    put = new if holder != "own_share" else {**block.anchor, "own_share": new}
                    load = _load({**source, where: put}, reg)
                    for rows, cols in ((org["rows"], org["cols"]), (org["rows"] // 2, org["cols"])):
                        price, _band = _price(physics, lambda: replace(block, **{where: put}), 4, rows, cols,
                                              org["weight_bits_per_cell"])
                        cases.append((f"{holder}.{key} = {value!r} at {rows} x {cols}", load, price))
        assert len(cases) > 400 and any(price == "priced" for _l, _d, price in cases)
        assert not _defects(cases), _defects(cases)[:30]

    @pytest.mark.parametrize("type_name", sorted(registry.load_registry().types))
    def test_every_field_the_rule_reads_off_the_generic_type_rule_with_every_bad_value(self, type_name):
        """Red on UB3's tree: the type rule's organization and reading are read unjudged (ZeroDivisionError,
        TypeError), the generic description under each type of the registry."""
        reg, examples = registry.load_registry(), _examples()
        physics = priced_physics(GENERIC)
        block = replace(physics.area_rule, type=type_name)
        source = {**block.to_dict(), "type": type_name}
        cases = []
        for holder in ("organization", "reading_low_edge", "node", "converter"):
            keys = _reads(holder) | (_reads("organization") if holder == "reading_low_edge" else set())
            held = getattr(block, holder)
            for key in sorted(keys) + [None]:
                for value in _values(holder, key, examples) if key is not None else BAD:
                    put = value if key is None else _changed(held, key, value)
                    load = _load({**source, holder: put}, reg)
                    price, _band = _price(physics, lambda: replace(block, **{holder: put}), 100, 256, 256, 4)
                    cases.append((f"{type_name} {holder}.{key} = {value!r}", load, price))
        assert len(cases) > 150 and any(price == "priced" for _l, _d, price in cases)
        assert not _defects(cases), _defects(cases)[:30]

    @pytest.mark.parametrize("name", ANCHORS)
    def test_every_cell_kind_with_every_own_share_basis_and_a_share_below_at_and_above_1(self, name):
        """Red on UB3's tree: a declared cell kind or cell puts an own share above 1, and the band's low edge is a
        negative chip area (ODIN with the logic-rule SRAM kind) or the price is not finite."""
        reg = registry.load_registry()
        physics = priced_physics(name)
        block = physics.area_rule
        org, s = block.organization, f"r('{reg.type_file(block.type)}:s')"
        shares = [{"basis": "own_memory_share"}] + [
            {"basis": "published_split", "expression": e} for e in (s, f"{s} / {s}", f"1.5 * {s} / {s}")]
        w = org["weight_bits_per_cell"]
        geometries = ((org["rows"], org["cols"], w), (max(1, org["rows"] // 16), max(1, org["cols"] // 16), w),
                      (org["rows"] * 2, org["cols"], 2 * w))
        cases, above = [], 0
        for kind in sorted(reg.files["cell_kinds"]["kinds"]) + [None]:
            for node in (block.node, {**block.node, "bitcell_um2": f"8 * {CELL}"}):
                for share in shares:
                    changed = replace(block, cell_kind=kind, node=node, anchor={**block.anchor, "own_share": share})
                    load = _load({**block.to_dict(), "cell_kind": kind, "node": node, "anchor": changed.anchor}, reg)
                    try:
                        value = anchor.anchor_law(reg, changed).own_share.value
                    except Exception:  # noqa: BLE001 - a law refused (or crashed: the price below records it)
                        value = None
                    above += value is None
                    for rows, cols, bits in geometries:
                        price, _band = _price(physics, lambda: changed, 4, rows, cols, bits)
                        cases.append((f"{kind} / {node.get('bitcell_um2')} / {share} at {rows} x {cols} x {bits}",
                                      load, price))
        assert len(cases) == 10 * 2 * 4 * 3
        assert not _defects(cases), _defects(cases)[:30]
        assert above, "no own share left (0, 1] for this anchor: the N-3 path is not exercised"

    @pytest.mark.parametrize("name", ANCHORS + (GENERIC,))
    def test_every_quantity_at_the_extremes_of_the_floats(self, name):
        """Red on UB3's tree: a divisor of 5e-324 puts an infinite ratio under math.ceil (OverflowError), a count of
        10**400 leaves the floats, and a tiny or huge figure prices a band of no meaning. Values of no fixed list: the
        floats' own extremes, through the organization, the node and the anchor's figures."""
        reg, physics = registry.load_registry(), priced_physics(name)
        block = physics.area_rule
        org = block.organization or {"rows": 256, "cols": 256, "weight_bits_per_cell": 4}
        extremes = (5e-324, sys.float_info.min, 1e-9, 0.5, 2 ** 53, 10 ** 20, 10 ** 400, 1e300, sys.float_info.max)
        cases, puts = [], []
        for key in sorted(_reads("organization") - {"terms", "term_weights"}):
            puts += [("organization", {**(block.organization or {}), key: v}) for v in extremes]
        puts += [("node", {**block.node, "node_nm": v}) for v in extremes]
        for key in sorted(core_key for core_key in (block.anchor or {}) if core_key in declaration.ANCHOR_FIGURES):
            e = block.anchor[key]
            puts += [("anchor", {**block.anchor, key: f"{v!r} * {e}"}) for v in (5e-324, 1e-300, 1e300, 1e308)]
        for holder, put in puts:
            load = _load({**block.to_dict(), holder: put}, reg)
            price, _band = _price(physics, lambda: replace(block, **{holder: put}), 4, org["rows"] // 2, org["cols"],
                                  org["weight_bits_per_cell"])
            cases.append((f"{holder} {put}", load, price))
        assert not _defects(cases), _defects(cases)[:20]

    def test_a_published_size_with_every_bad_value(self):
        """Red on UB3's tree: a typed or malformed published size crashes the pricer's size flags."""
        reg, physics = registry.load_registry(), priced_physics("isaac_like")
        block, cases = physics.area_rule, []
        e = block.published_size["n_pub"]
        for value in BAD + tuple(f.format(e=e) for f in EXPRESSION_FORMS):
            put = {**block.published_size, "n_pub": value}
            load = _load({**block.to_dict(), "published_size": put}, reg)
            price, _band = _price(physics, lambda: replace(block, published_size=put), 16128, 128, 128, 2)
            cases.append((f"published_size.n_pub = {value!r}", load, price))
        assert not _defects(cases), _defects(cases)


# ---------------------------------------------------------------------------------------------------------- MA-D1 (a)
class TestTheOneJudgeMAD1:
    def test_each_kind_takes_its_values_and_refuses_the_others_by_the_fields_name(self):
        """Red on UB3's tree (no judge)."""
        core = importlib.import_module(CORE)
        cases = {core.COUNT: ((1, 2.0, 256, 2 ** 53), (0, -1, 0.5, 2 ** 60, "4", None, True, math.inf, math.nan, [4],
                                                         10 ** 400, sys.float_info.min)),
                 core.COUNT_OR_ZERO: ((0, 0.0, 640), (-1, 7.5, "0", None, False, math.inf, -math.inf, math.nan)),
                 core.SHARE: ((1, 1.0, 0.5, 1e-9), (0, -0.5, 1.0000001, 1.5, "0.5", None, True, math.nan)),
                 core.POSITIVE: ((14, 0.0864), (0, -130, "28", None, True, math.inf, math.nan, [28])),
                 core.CELL: ((0.0312,), (0, -0.0312, math.inf, math.nan, "0.0312"))}
        for kind, (good, bad) in cases.items():
            for value in good:
                assert core.judged("area_rule.x", value, kind) == value
            for value in bad:
                with pytest.raises(AreaRuleRefusal) as caught:
                    core.judged("area_rule.organization.rows", value, kind)
                text = str(caught.value)
                assert "area_rule.organization.rows" in text and repr(value)[:12] in text and kind.words in text
        assert core.judged("w", {"cells": 36, "converters": 0}, core.WEIGHTS) == {"cells": 36, "converters": 0}
        for bad in ([1], None, 0, {"bits": -1}, {"bits": "1"}, {"bits": math.inf}, {"bits": True}):
            with pytest.raises(AreaRuleRefusal, match="term_weights"):
                core.judged("area_rule.organization.term_weights", bad, core.WEIGHTS)

    def test_the_judged_fields_are_the_fields_the_count_form_reads(self):
        """Red on UB3's tree (no table): the fields the code reads off an organization (core/equation.py and
        anchor.py, by AST) are the judge's table and the one name 'terms'; a field read later fails this test until it
        has its kind."""
        core = importlib.import_module(CORE)
        assert _reads("organization") - {"terms"} == set(core.QUANTITIES)

    def test_the_load_judge_and_the_pricer_call_the_one_judge(self, monkeypatch):
        """Red on UB3's tree (no judge): the organization's fields, the node and the anchor's figures and share are
        judged by core.judged at load and again at the price."""
        core = importlib.import_module(CORE)
        seen, real = [], core.judged
        # the fourth delta round: the judge makes its words from ``field.format(*words)`` only on a refusal
        monkeypatch.setattr(core, "judged", lambda field, value, kind, *words: seen.append(
            field.format(*words) if words else field) or real(field, value, kind, *words))
        physics = priced_physics("loihi")
        declaration.judge_block(physics.area_rule, registry.load_registry())
        at_load, seen[:] = set(seen), []
        # the fourth delta round: the price judges and derives once per frozen input (core/door.remembered); forgotten,
        # the price judges and derives again here
        importlib.import_module("mimarsinan.deployment_record.platform_physics.area_rule.core.door").forget(
            registry.load_registry())
        price_absolute(_census(64, 64, 512, 8), physics, geometry=[CoreGeometry(64, 512, 64)],
                       readout_bits=READOUT_BITS)
        for fields in (at_load, set(seen)):
            text = " ".join(fields)
            for name in ("organization.rows", "organization.stored_bits_per_cell", "organization.term_weights",
                         "node.node_nm", "anchor.core_mm2", "anchor.n_pub", "grid share"):
                assert name in text, (name, sorted(fields))
        assert any("own_share" in f for f in seen)

    def test_an_own_share_above_1_takes_n3s_path_the_held_logic_edge_refused_by_name(self):
        """Red on UB3's tree: ODIN with the logic-rule SRAM kind has an own share of 1.144 and a band of
        (-0.013, 0.001, 0.001) mm2 at one 16 x 16 core."""
        reg = registry.load_registry()
        odin = replace(get_platform_physics("odin").area_rule, cell_kind="sram_6t_logic_rule")
        law = anchor.anchor_law(reg, odin)
        assert law.own_share.value is None and "own_share" in law.own_share.refusal
        assert "a share above 0 and at most 1" in law.own_share.refusal
        price = anchor.price_anchor_types(law, ((1.0, rule.CensusCore(16, 16, 3)),))
        assert 0.0 < price.low <= price.nominal <= price.high
        assert any(f.kind == "band_edge_refused" and "own_share" in f.reason for f in price.flags)
        split = replace(get_platform_physics("loihi").area_rule, anchor={
            **get_platform_physics("loihi").area_rule.anchor,
            "own_share": {"basis": "published_split", "expression": "1.5 * r('types/t1:s') / r('types/t1:s')"}})
        assert anchor.anchor_law(reg, split).own_share.value is None

    def test_an_anchor_that_holds_its_logic_per_core_with_a_share_above_1_is_refused_off_its_point(self):
        """Red on UB3's tree: TrueNorth with the PCM unit cell has an own share of 15.35 and a nominal of -1,641 mm2 at
        4,096 cores of 128 x 256."""
        physics = priced_physics("truenorth")
        block = replace(physics.area_rule, cell_kind="pcm_analog_unit_cell")
        state, _band = _price(physics, lambda: block, 4096, 128, 256, 9)
        assert state == "refused", state
        own = pricing.rule_area(replace(physics, area_rule=block), _census(4096, 256, 256, 9),
                                [CoreGeometry(256, 256, 4096)], READOUT_BITS)
        assert own.band is not None and own.band[0] == own.band[1] == own.band[2], own  # at its own point: identity

    def test_an_own_die_smaller_than_its_core_grid_is_a_named_refusal_at_load_and_at_the_price(self):
        """Red on UB3's tree: TrueNorth with half its published die loads, and its band's low edge is a negative chip
        area off its size (the held outside part die - n_pub x core is negative)."""
        physics = priced_physics("truenorth")
        block = physics.area_rule
        half = {**block.anchor, "die_mm2": f"{block.anchor['die_mm2']} / 2"}
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(replace(block, anchor=half), registry.load_registry())
        assert "grid share" in str(caught.value)
        state, _band = _price(physics, lambda: replace(block, anchor=half), 16, 256, 256, 9)
        assert state == "refused", state

    def test_a_type_rules_organization_and_reading_are_judged_at_load_and_at_the_price(self):
        """Red on UB3's tree (MA-D1 (d)): both load, and the price raises ZeroDivisionError or TypeError."""
        reg, physics = registry.load_registry(), priced_physics(GENERIC)
        block = physics.area_rule
        cases = {"organization": ({"weight_bits_per_cell": 0, "stored_bits_per_cell": 1},
                                  "area_rule.organization.weight_bits_per_cell"),
                 "reading_low_edge": ({**block.reading_low_edge, "bits_per_neuron": None},
                                      "area_rule.reading_low_edge.bits_per_neuron")}
        for holder, (put, field) in cases.items():
            with pytest.raises(AreaRuleRefusal) as caught:
                declaration.judge_block(replace(block, **{holder: put}), reg)
            assert field in str(caught.value)
            got = pricing.rule_area(replace(physics, area_rule=replace(block, **{holder: put})),
                                    _census(100, 256, 256, 4), [CoreGeometry(256, 256, 100)], READOUT_BITS)
            assert got.refusal is not None and field in got.refusal, got

    def test_a_type_rule_whose_organization_stores_no_bit_is_a_named_refusal_of_its_count(self):
        """Red on UB3's tree: the core prices as 0 mm2. Red on the mutant that skips the count's judge (the band's
        judge then refuses, without naming the organization)."""
        physics = priced_physics(GENERIC)
        put = {"weight_bits_per_cell": 4, "stored_bits_per_cell": 0, "bits_per_neuron": 0}
        got = pricing.rule_area(replace(physics, area_rule=replace(physics.area_rule, organization=put)),
                                _census(100, 256, 256, 4), [CoreGeometry(256, 256, 100)], READOUT_BITS)
        assert got.refusal is not None and "area_rule.organization" in got.refusal, got

    def test_the_rules_own_band_is_judged_a_finite_area_above_0(self):
        """Red on UB3's tree: a pooled census whose mean core holds 1e306 rows prices an infinite band (a census-side
        value, which no declared field states, so the band's own judge refuses it)."""
        physics = priced_physics("loihi")
        got = pricing.rule_area(physics, _census(64, 1e306, 512, 8), None, READOUT_BITS)
        assert got.band is None and got.refusal is not None and AREA in got.refusal, got

    @pytest.mark.parametrize("name, field", [(GENERIC, "cell_kind"), ("neurram", "converter.kind"),
                                             ("neurram", "converter.own_unit"), (GENERIC, "reading_low_edge.type")])
    def test_a_name_the_equation_reads_that_is_not_a_text_is_a_named_refusal(self, name, field):
        """Red on UB3's tree: a list as a name raises TypeError (unhashable) out of its table's lookup at load (and at
        the price, where the price reads it)."""
        reg = registry.load_registry()
        physics = priced_physics(name)
        block = physics.area_rule
        holder, _dot, key = field.partition(".")
        put = ["sram_6t_push_rule"] if not key else {**(getattr(block, holder) or {}), key: ["T1"]}
        changed = replace(block, **{holder: put})
        with pytest.raises(AreaRuleRefusal):
            declaration.judge_block(changed, reg)
        org = block.organization or {"rows": 256, "cols": 256, "weight_bits_per_cell": 4}
        state, _band = _price(physics, lambda: changed, 4, org["rows"] // 2, org["cols"], org["weight_bits_per_cell"])
        # an own printed unit is read only where a converter is priced from it (never by an anchor's ratio)
        assert state == "refused" or (key == "own_unit" and state == "priced"), state

    @pytest.mark.parametrize("holder", ["node", "anchor", "organization", "converter", "reading_low_edge",
                                        "published_size"])
    def test_a_holder_of_quantities_that_is_not_an_object_is_a_named_refusal(self, holder):
        """Red on UB3's tree: the block is made, and its reads raise AttributeError or TypeError."""
        block = get_platform_physics("loihi").area_rule
        for value in ([1], "text", 4, True):
            with pytest.raises(AreaRuleRefusal) as caught:
                replace(block, **{holder: value})
            assert f"area_rule.{holder}" in str(caught.value)
            with pytest.raises(AreaRuleRefusal):
                declaration.AreaRuleBlock.from_dict({**block.to_dict(), holder: value})

    def test_a_value_outside_its_kind_is_refused_at_load_and_at_the_price_never_priced(self):
        """Red on UB3's tree (MA-D1 (c)): weight_bits_per_cell -4, rows 0 or -1 on a bits anchor and a bool count
        priced silently. Every value of the fixed list that its field's kind refuses is a named refusal at both."""
        core = importlib.import_module(CORE)
        reg = registry.load_registry()
        silent = []
        for name in ANCHORS:
            physics = priced_physics(name)
            block = physics.area_rule
            org = block.organization
            puts = [("organization", key, kind, value, {**org, key: value})
                    for key, kind in core.QUANTITIES.items() for value in BAD]
            puts += [("node", "node_nm", core.POSITIVE, value, {**block.node, "node_nm": value})
                     for value in BAD + (-130,) if value not in (0, None)]  # 0 and None: 'names its node' at load
            puts += [("anchor", key, core.ANCHOR_FIGURES[key], value, {**block.anchor, key: f"{value} * {e}"})
                     for key, e in block.anchor.items() if key in core.ANCHOR_FIGURES and e is not None
                     for value in (0, -1)]
            for holder, key, kind, value, put in puts:
                try:
                    core.judged(key, value, kind)
                    continue  # a value of its kind (an edge such as 0 bits per neuron)
                except AreaRuleRefusal:
                    pass
                if _try_accept({**block.to_dict(), holder: put}, reg):
                    silent.append(f"{name} load {holder}.{key} = {value!r}")
                state, _band = _price(physics, lambda: replace(block, **{holder: put}), 4, org["rows"] // 2,
                                      org["cols"], org["weight_bits_per_cell"])
                if state != "refused":
                    silent.append(f"{name} price {holder}.{key} = {value!r}: {state}")
        assert not silent, silent

    def test_an_organization_that_is_not_an_object_is_refused_by_the_judge_itself(self):
        """Red on UB3's tree (no judge); red on the mutant that reads a non-object as an empty organization."""
        core = importlib.import_module(CORE)
        for value in ([1], "text", 4, True):
            with pytest.raises(AreaRuleRefusal, match="area_rule.organization"):
                core.judged_fields(value, "area_rule.organization")

    def test_an_anchor_that_states_no_scope_is_a_named_refusal_at_the_price(self):
        """Red on UB3's tree: a KeyError 'scope' out of price_absolute (the load judge refuses it, the pricer not)."""
        physics = priced_physics("loihi")
        block = physics.area_rule
        anchor_ = {k: v for k, v in block.anchor.items() if k != "scope"}
        state, _band = _price(physics, lambda: replace(block, anchor=anchor_), 4, 64, 512, 8)
        assert state == "refused", state


def _try_accept(source, reg):
    """True where the load judge accepts ``source`` (no exception at all)."""
    try:
        declaration.judge_block(declaration.AreaRuleBlock.from_dict(source), reg)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------------------------------------- mi-D1
class TestDeclaredCellFlagWhereReadMiD1:
    def _flags_and_band(self, physics, cores=100, rows=256, cols=256, bits=4):
        priced = price_absolute(_census(cores, rows, cols, bits), physics, geometry=[CoreGeometry(rows, cols, cores)],
                                readout_bits=READOUT_BITS)
        term = {t.name: t for t in priced.terms}.get(AREA)
        band = None if term is None else (term.band.low, term.band.nominal, term.band.high)
        return {f.kind for f in priced.flags if f.name == AREA}, band

    @pytest.mark.parametrize("type_name", ["T3", "T4"])
    def test_a_cited_cell_the_resolved_cell_never_reads_moves_nothing_and_carries_no_flag(self, type_name):
        """Red on UB3's tree: a T3 or T4 type rule (an F2 or um2 cell) carries the declared_cell flag on a price the
        cell does not move. Red on the mutant 'the flag unconditional'."""
        physics = priced_physics(GENERIC)
        base = replace(physics.area_rule, type=type_name, reading_low_edge=None)
        cited = replace(base, node={**base.node, "bitcell_um2": CELL})
        _flags_base, band_base = self._flags_and_band(replace(physics, area_rule=base))
        flags, band = self._flags_and_band(replace(physics, area_rule=cited))
        assert band is not None and band == band_base, (band, band_base)
        assert "declared_cell" not in flags, flags

    @pytest.mark.parametrize("name", ["neurram", "hermes_64"])
    def test_an_own_memory_share_anchor_of_a_cell_that_never_reads_the_cited_cell_carries_no_flag(self, name):
        """Red on UB3's tree (the re-verify's probe P3: NeuRRAM and HERMES carry the flag, share and price unmoved)."""
        physics = priced_physics(name)
        block = replace(physics.area_rule, anchor={**physics.area_rule.anchor, "own_share": {"basis":
                                                                                              "own_memory_share"}})
        cited = replace(block, node={**block.node, "bitcell_um2": CELL})
        org = block.organization
        geometry = dict(cores=4, rows=org["rows"] // 2, cols=org["cols"], bits=org["weight_bits_per_cell"])
        _flags_base, band_base = self._flags_and_band(replace(physics, area_rule=block), **geometry)
        flags, band = self._flags_and_band(replace(physics, area_rule=cited), **geometry)
        assert band is not None and band == band_base and "declared_cell" not in flags, (flags, band, band_base)

    def test_a_cited_cell_the_resolved_cell_reads_moves_the_price_with_the_flag(self):
        """A guard on UB3's tree for the type rule; the anchor's own memory share and a T3 rule declaring an SRAM-form
        kind read the cited cell and carry the flag."""
        cases = []
        physics = priced_physics(GENERIC)
        t3 = replace(physics.area_rule, type="T3", reading_low_edge=None, cell_kind="sram_6t_push_rule")
        cases.append((physics, physics.area_rule, {}))
        cases.append((physics, t3, {}))
        loihi = priced_physics("loihi")
        org = loihi.area_rule.organization
        cases.append((loihi, loihi.area_rule, dict(cores=4, rows=org["rows"] // 2, cols=org["cols"], bits=8)))
        for phys, block, geometry in cases:
            cited = replace(block, node={**block.node, "bitcell_um2": CELL})
            _flags, band_base = self._flags_and_band(replace(phys, area_rule=block), **geometry)
            flags, band = self._flags_and_band(replace(phys, area_rule=cited), **geometry)
            assert band is not None and band != band_base and "declared_cell" in flags, (block.type, flags, band)

    def test_a_type_rule_whose_reading_reads_the_cited_cell_carries_the_flag(self):
        """A guard on UB3's tree; red on the mutant that asks the type rule's own cell only: a T3 rule's own cell never
        reads the cited cell, but its T1 reading (the push-rule cell on sigma) does, and the reading moves."""
        physics = priced_physics(GENERIC)
        base = replace(physics.area_rule, type="T3")
        cited = replace(base, node={**base.node, "bitcell_um2": CELL})
        readings = []
        for block in (base, cited):
            got = pricing.rule_area(replace(physics, area_rule=block), _census(100, 256, 256, 4),
                                    [CoreGeometry(256, 256, 100)], READOUT_BITS)
            readings.append({f.kind: f.value for f in got.flags})
        assert "declared_cell" in readings[1] and "declared_cell" not in readings[0]
        assert readings[1]["t1_reading"] != readings[0]["t1_reading"]

    def test_the_resolved_cell_reads_a_declared_cell_exactly_where_the_cell_area_does(self):
        """Red on UB3's tree (no resolution): for every cell kind and type, the resolution says it read the declared
        cell exactly where declaring it moves the cell's area (one branch, rule.reads_sigma)."""
        core = importlib.import_module(CORE)
        reg = registry.load_registry()
        for type_name in reg.types:
            for kind in sorted(reg.files["cell_kinds"]["kinds"]) + [None]:
                bare = core.Declared(28.0, kind)
                cited = bare._replace(bitcell_um2=8 * reg.evaluate(CELL))
                got = core.resolved_cell(reg, type_name, cited)
                assert got.um2 == core.core_cell_um2(reg, type_name, cited)
                assert got.reads_declared == (got.um2 != core.resolved_cell(reg, type_name, bare).um2), (type_name,
                                                                                                          kind)
                assert not core.resolved_cell(reg, type_name, bare).reads_declared


# ---------------------------------------------------------------------------------------------------------- mi-D2
class TestTheCountFormsGuardsMiD2:
    @pytest.mark.parametrize("key", ["rows", "cols"])
    def test_an_anchor_priced_without_rows_or_cols_is_a_named_refusal(self, key):
        """A guard on UB3's tree; red on the re-verify's mutant X_rows (own_counts' refusal dropped: KeyError)."""
        physics = priced_physics("loihi")
        block = physics.area_rule
        org = {k: v for k, v in block.organization.items() if k != key}
        state, _band = _price(physics, lambda: replace(block, organization=org), 4, 64, 512, 8)
        assert state == "refused", state
        got = pricing.rule_area(replace(physics, area_rule=replace(block, organization=org)), _census(4, 64, 512, 8),
                                [CoreGeometry(64, 512, 4)], READOUT_BITS)
        assert got.refusal is not None and key in got.refusal, got

    @pytest.mark.parametrize("type_name", ["T3", "T4", "T2_analog"])
    def test_a_cells_type_rule_without_columns_per_converter_counts_one_converter_per_column(self, type_name):
        """A guard on UB3's tree; red on the re-verify's mutant X_m_default2 (a missing m read as 2)."""
        reg = registry.load_registry()
        assert reg.type_table(type_name)["scaled_terms"] not in rule.BITS_TERMS
        declared = typerule.Declared(28.0)
        counts = typerule.census_counts(reg, type_name, declared, {"weight_bits_per_cell": 4},
                                        rule.CensusCore(256, 192, 4), 4)
        assert counts.converters == 192
        hermes = get_platform_physics("hermes_64").area_rule
        org = {k: v for k, v in hermes.organization.items() if k != "columns_per_converter"}
        assert anchor.anchor_ratio(org, rule.CensusCore(256, 192, 4)).anchor_counts["converters"] == org["cols"]


def test_the_shipped_descriptions_pass_the_judge():
    """A guard: every shipped description with a block loads and passes the load judge (no shipped value lies
    outside its kind)."""
    reg, loaded = registry.load_registry(), 0
    for path in sorted((registry.DATA.parent.parent / "profiles").glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("area_rule") is not None:
            declaration.judge_block(get_platform_physics(data["name"]).area_rule, reg)
            loaded += 1
    assert loaded == 11
