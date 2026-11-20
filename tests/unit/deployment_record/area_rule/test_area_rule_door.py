"""[U-AREA] The third delta round (UB4/check.md: MA-K1, MA-K2, mi-K1 to mi-K3; UB4/report.md: mi-E1): one door for the
area rule, tests first. Each test is red on UB4's tree (UB4/uarea.patch on R5, applied digest 43729df7) or on its named
mutant, green on the round's tree; a guard (green on UB4's tree too) is named as such.

- The door: every public function of the rule's pricing modules (and the registry's two entries a caller's quantity
  reaches) judges its whole input by the one function the loader also calls, before any arithmetic; a public function
  that is neither a door nor named with its reason fails a structural test, and so does an argument without its judge
  and a read of a declared field outside the judge's table (the AST sweep over every module of the rule).
- The check's sets: V (the validity geometry's bad values: a named refusal at load and at the price, by the same words)
  and the extremes (every node extreme x every cell kind x every type and anchor: a named refusal or a finite, positive,
  ordered band); mi-E1 and mi-K3 as door cases; mi-K1 (the band's order); mi-K2 (the held-logic statement).
- The family of my own: every argument of every door, with bad values and with each field of a structured argument
  changed, gives a named refusal or a finite value of its kind (a band positive and ordered), never another exception.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mimarsinan.common.refusal import is_toolchain_refusal
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.area_rule import (
    anchor, declaration, pricing, registry, rule, typerule)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import decomposition, parity
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule._physics import priced_physics, store_declared

AREA = "chip_area_mm2"
AR = Path(registry.__file__).resolve().parent
PKG = "mimarsinan.deployment_record.platform_physics.area_rule"
#: The round's modules (absent on UB4's tree: the tests that read them are red there by their absence).
DOOR, FIELDS, ARGUMENTS = f"{PKG}.core.door", f"{PKG}.core.fields", f"{PKG}.core.arguments"
#: The rule's pricing modules: every public function of each is a door, or named with its reason (door.NOT_DOORS).
PRICING_MODULES = ("core/equation.py", "core/cells.py", "rule.py", "anchor.py", "typerule.py", "pricing.py",
                   "declaration.py", "published/parity.py", "published/decomposition.py")
#: The registry's entries a caller's quantity reaches (a node and a declared cell; an expression).
REGISTRY_DOORS = ("Registry.sigma", "Registry.evaluate")
ANCHORS = ("truenorth", "loihi", "odin", "hermes_64", "neurram", "sram_cim_macro")
GENERIC = "generic_estimated_22nm"
PROFILES = AR.parent / "profiles"
READOUT_BITS = 8
CELL = "v('S-SRAM-10')"
#: Set V's values (the check's list, less the values of a count): a field's bad and edge values.
V_BAD = (0, -1, -0.0, 0.5, 7.5, "4", "", None, True, False, math.inf, -math.inf, math.nan, [4], {"x": 1}, 10 ** 400,
         2 ** 53 + 1, 1e308, sys.float_info.max, 5e-324)
NODE_EXTREMES = (5e-324, sys.float_info.min, 1e-9, 0.5, 1e150, 1.4e157, 1e200, 1e300, sys.float_info.max)


def _module(dotted):
    return importlib.import_module(dotted)


# ---------------------------------------------------------------------------------------------------------- outcomes
def _census(cores, rows, cols, weight_bits, form="declared"):
    """Quantities and the core types: 'declared'; 'no_bits' (the census withholds its weight bits); 'as_declared'
    (cores only: read at the description's own core)."""
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": weight_bits, "tiles": cores}
    withheld, geometry = None, [CoreGeometry(rows, cols, cores)]
    if form == "no_bits":
        values.pop("weight_bits")
        withheld = {"weight_bits": "core types of several widths (a test census)"}
    elif form == "as_declared":
        values, geometry = {"cores_physical": cores, "weight_bits": weight_bits, "tiles": cores}, None
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()}, withheld), geometry


def _crash(caught):
    return None if is_toolchain_refusal(caught) and str(caught).strip() else f"{type(caught).__name__}: {caught}"


def _load_block(source, reg, validity=None):
    """The load judge: None where accepted, ('refused', text), or ('CRASH', text)."""
    try:
        declaration.judge_block(declaration.AreaRuleBlock.from_dict(source), reg)
    except Exception as caught:  # noqa: BLE001 - every exception is recorded
        crash = _crash(caught)
        return ("refused", str(caught)) if crash is None else ("CRASH", crash)
    return None


def _price(physics, cen):
    """price_absolute on ``physics`` and the census ``cen``: ('priced', band) | ('refused', reason) | (defect, ...)."""
    try:
        q, geometry = _census(*cen)
        priced = price_absolute(q, physics, geometry=geometry, readout_bits=READOUT_BITS)
    except Exception as caught:  # noqa: BLE001
        crash = _crash(caught)
        return ("refused", str(caught)) if crash is None else (f"CRASH {crash}", None)
    term = {t.name: t for t in priced.terms}.get(AREA)
    if term is None:
        reason = {r.name: r.reason for r in priced.refusals}.get(AREA)
        return ("refused", reason) if reason and str(reason).strip() else ("NO NAMED REFUSAL", None)
    band = (term.band.low, term.band.nominal, term.band.high)
    good = all(math.isfinite(x) for x in band) and 0.0 < band[0] <= band[1] <= band[2]
    return ("priced" if good else f"BAD BAND {band}"), band


def _shipped():
    """Every shipped description with an area_rule block, as its JSON."""
    out = {}
    for path in sorted(PROFILES.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("area_rule") is not None:
            out[data["name"]] = data
    return out


# ---------------------------------------------------------------------------------------------------------- set V
class TestTheValidityGeometryMAK1:
    @pytest.mark.parametrize("field", ["cell_bits", "axons", "neurons"])
    def test_every_bad_value_is_a_named_refusal_at_load_and_at_the_price_by_the_same_words(self, field):
        """Red on UB4's tree: the loader accepts every value (MA-K1), and the price raises TypeError, ValueError or
        OverflowError, or prices a bool, a fraction or a core of 0 rows."""
        defects = []
        for name, data in _shipped().items():
            if data["area_rule"]["terms"] == "complete":
                continue
            physics = store_declared(get_platform_physics(name))
            geometry = physics.validity.geometry
            w = (data["area_rule"].get("organization") or {}).get("weight_bits_per_cell", 4)
            for value in V_BAD:
                if field == "cell_bits" and value is None:
                    continue  # absent cell bits: a census states its own or the type rule names its refusal
                mod = copy.deepcopy(data)
                mod["validity"]["geometry"][field] = value
                try:
                    profile_from_dict(mod)
                    load = "accepted"
                except Exception as caught:  # noqa: BLE001
                    load = str(caught) if _crash(caught) is None else f"CRASH {_crash(caught)}"
                if load == "accepted" or load.startswith("CRASH") or f"validity.geometry.{field}" not in load:
                    defects.append(f"{name} load {field} = {value!r}: {load[:160]}")
                bad = replace(physics, validity=replace(physics.validity, geometry=replace(geometry, **{field: value})))
                for cen in ((8, 64, 64, w, "no_bits"), (8, 64, 64, w, "as_declared")):
                    state, reason = _price(bad, cen)
                    if state != "refused" or reason != load:
                        defects.append(f"{name} price {field} = {value!r} on {cen[-1]}: {state} {str(reason)[:120]}")
        assert not defects, defects[:20]

    def test_the_shipped_validity_geometries_pass_the_judge(self):
        """A guard: every shipped description loads and prices at its own core (no shipped value lies outside)."""
        for name in _shipped():
            physics = store_declared(get_platform_physics(name))
            g = physics.validity.geometry
            state, _band = _price(physics, (1, g.axons, g.neurons, g.cell_bits or 4))
            assert state in ("priced", "refused") and not state.startswith("CRASH"), (name, state)


# ---------------------------------------------------------------------------------------------------------- extremes
class TestTheExtremesMAK2:
    @pytest.mark.parametrize("base", sorted(registry.load_registry().types) + list(ANCHORS))
    def test_every_node_extreme_with_every_cell_kind_is_a_named_refusal_or_an_ordered_band(self, base):
        """Red on UB4's tree: a node from 1.4e157 nm up overflows the F2 cell's square (OverflowError at rule.py:183)
        for every F2 kind on the generic type rule and on every anchor priced with its own memory share."""
        reg = registry.load_registry()
        name = GENERIC if base in reg.types else base
        physics = priced_physics(name)
        block = physics.area_rule
        if base in reg.types:
            block = replace(block, type=base)
            cens = ((100, 256, 256, 4, "declared"),)
        else:
            block = replace(block, anchor={**block.anchor, "own_share": {"basis": "own_memory_share"}})
            org = block.organization
            cens = ((4, max(1, org["rows"] // 2), org["cols"], org["weight_bits_per_cell"], "declared"),)
        defects = []
        for kind in sorted(reg.files["cell_kinds"]["kinds"]) + [None]:
            for node in NODE_EXTREMES:
                changed = replace(block, cell_kind=kind, node={**block.node, "node_nm": node})
                load = _load_block({**changed.to_dict()}, reg)
                if load is not None and load[0] == "CRASH":
                    defects.append(f"{kind} {node} load {load[1][:120]}")
                for cen in cens:
                    state, _band = _price(replace(physics, area_rule=changed), cen)
                    if state not in ("priced", "refused"):
                        defects.append(f"{kind} {node} price {state[:160]}")
        assert not defects, defects[:20]

    def test_an_f2_cell_beyond_the_floats_is_a_named_refusal_where_it_is_made(self):
        """Red on UB4's tree: OverflowError out of the cell's square; the cell's judge names the cell."""
        core = _module(f"{PKG}.core.equation")
        reg = registry.load_registry()
        with pytest.raises(AreaRuleRefusal) as caught:
            core.resolved_cell(reg, "T3", core.Declared(1e300, "rram_1t1r"))
        assert "cell" in str(caught.value)

    def test_a_converter_law_beyond_the_floats_is_a_named_refusal(self):
        """Red on UB4's tree: OverflowError out of a law's 2 ** (alpha (b - ref)) at a reading's lifted bits."""
        reg = registry.load_registry()
        with pytest.raises(AreaRuleRefusal):
            rule.converter_area(reg, "sar_dedicated", 2 ** 40, check=False)


# ---------------------------------------------------------------------------------------------------------- door cases
def _anchor_block(name="loihi"):
    return priced_physics(name), get_platform_physics(name).area_rule


#: (case, the description, the block the loader refuses as a replace of the shipped one)
DOOR_CASES = {
    "mi-E1 terms": (lambda b: replace(b, terms="no_such_terms")),
    "mi-E1 outside": (lambda b: replace(b, outside="no_such_outside")),
    "mi-K3 an anchor organization without weight bits per cell": (
        lambda b: replace(b, organization={k: v for k, v in b.organization.items() if k != "weight_bits_per_cell"})),
    "an organization that cites nothing": (lambda b: replace(b, organization={**b.organization, "citation": []})),
    "a published size beside an anchor": (lambda b: replace(b, published_size={"n_pub": b.anchor["n_pub"]})),
    "an own unit the evidence does not hold": (lambda b: replace(b, converter={"own_unit": "NO-SUCH-ID"})),
    "a cell kind the registry does not hold": (lambda b: replace(b, cell_kind="no_such_kind")),
    "a converter kind that requires another cell kind": (lambda b: replace(b, converter={"kind": "sar_bitcell_reusing"})),
    "mi-K2 a held-logic statement of no citation": (
        lambda b: replace(b, anchor={**b.anchor, "logic_held_per_core": {"citation": []}})),
    "mi-K2 a held-logic text": (lambda b: replace(b, anchor={**b.anchor, "logic_held_per_core": "no"})),
}


class TestTheDoorCases:
    @pytest.mark.parametrize("case", sorted(DOOR_CASES))
    def test_a_block_the_loader_refuses_is_refused_at_the_price_by_the_same_words(self, case):
        """Red on UB4's tree (mi-E1, mi-K2, mi-K3: checks made at load only): the price prices the block."""
        physics, block = _anchor_block()
        changed = DOOR_CASES[case](block)
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(changed, registry.load_registry())
        for cen in ((4, 64, 512, 8, "declared"), (128, 128, 1024, 8, "declared")):
            state, reason = _price(replace(physics, area_rule=changed), cen)
            assert (state, reason) == ("refused", str(caught.value)), (state, reason, str(caught.value))

    @pytest.mark.parametrize("value", [None, {}])
    def test_a_type_rules_reading_without_its_type_is_refused_at_the_price_by_the_same_words_mi_k3(self, value):
        """Red on UB4's tree: the price drops the reading and the band loses R2's low edge, with no flag."""
        physics = priced_physics(GENERIC)
        block = physics.area_rule
        reading = {k: v for k, v in block.reading_low_edge.items() if k != "type"}
        if value != {}:
            reading["type"] = value
        changed = replace(block, reading_low_edge=reading)
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(changed, registry.load_registry())
        state, reason = _price(replace(physics, area_rule=changed), (100, 256, 256, 4))
        assert (state, reason) == ("refused", str(caught.value)), (state, reason)

    @pytest.mark.parametrize("field", ["rows", "cols", "weight_bits_per_cell"])
    def test_a_type_rules_organization_field_stated_as_none_is_refused_at_load_and_at_the_price(self, field):
        """Red on mutant e_load_org (the load judge's judge of the organization removed): the judge's table reads a
        field stated as None as absent, so the load accepts what the price's door refuses (the check's generator
        found 39 such declarations on that mutant)."""
        physics = priced_physics(GENERIC)
        changed = replace(physics.area_rule, organization={**(physics.area_rule.organization or {}), field: None})
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(changed, registry.load_registry())
        state, reason = _price(replace(physics, area_rule=changed), (100, 256, 256, 4))
        assert (state, reason) == ("refused", str(caught.value)), (state, reason)

    def test_the_held_logic_statement_of_record_is_a_citation_object(self):
        """A guard: TrueNorth's held logic is a citation object with its citation; it loads and prices as held."""
        physics = priced_physics("truenorth")
        held = physics.area_rule.anchor["logic_held_per_core"]
        assert isinstance(held, dict) and held["citation"]
        declaration.judge_block(physics.area_rule, registry.load_registry())
        assert anchor.anchor_law(registry.load_registry(), physics.area_rule).logic_held_per_core is True


# ---------------------------------------------------------------------------------------------------------- mi-K1
class TestTheBandsOrderMiK1:
    @pytest.mark.parametrize("band", [(2.0, 1.0, 3.0), (1.0, 3.0, 2.0), (0.0, 1.0, 2.0), (1.0, math.inf, 2.0),
                                      (math.nan, 1.0, 2.0)])
    def test_a_price_whose_band_is_not_ordered_and_positive_is_refused_where_it_is_made(self, band):
        """Red on UB4's tree: the price is made, and an unordered band reaches metrics' Band as an unnamed
        ValueError (the band's judge judged each edge alone)."""
        low, nominal, high = band
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.AreaPrice(nominal=nominal, low=low, high=high)
        assert AREA in str(caught.value)

    def test_a_reading_beside_the_band_is_judged_where_it_is_made(self):
        """Red on UB4's tree: a reading of no finite area is carried as a flag value."""
        with pytest.raises(AreaRuleRefusal):
            rule.AreaPrice(nominal=1.0, low=1.0, high=1.0, readings={"fixed_outside_reading": math.inf})


# ---------------------------------------------------------------------------------------------------------- the judge
class TestTheKindsAreClosed:
    def test_every_kind_states_both_ends(self):
        """Red on UB4's tree: a kind's upper end defaults to infinity (POSITIVE, CELL and WEIGHTS state none)."""
        judge = _module(f"{PKG}.core.judge")
        kinds = [v for v in vars(judge).values() if isinstance(v, judge.Kind)]
        assert len(kinds) >= 7
        for kind in kinds:
            assert math.isfinite(kind.high) and math.isfinite(kind.low) and kind.low <= kind.high, kind
        assert "high" not in judge.Kind._field_defaults

    def test_a_count_above_two_to_the_53_is_refused_exactly(self):
        """Red on UB4's tree: 2^53 + 1 passes a count's kind, because its float rounds to 2^53 (the check's note)."""
        judge = _module(f"{PKG}.core.judge")
        assert judge.judged("rows", 2 ** 53, judge.COUNT) == 2 ** 53
        with pytest.raises(AreaRuleRefusal, match="rows"):
            judge.judged("rows", 2 ** 53 + 1, judge.COUNT)

    def test_an_expression_the_registry_cannot_evaluate_is_a_named_refusal(self):
        """Red on UB4's tree: ZeroDivisionError, OverflowError and TypeError out of Registry.evaluate."""
        reg = registry.load_registry()
        for expression in ("1 / 0", "10.0 ** 400", "v('S-SRAM-10', 99)", ["x"], 4, "1 +"):
            with pytest.raises(AreaRuleRefusal):
                reg.evaluate(expression)
        for node in ("28", [28], None, math.nan, -1, True):
            with pytest.raises(AreaRuleRefusal):
                reg.sigma(node)


# ---------------------------------------------------------------------------------------------------------- where made
class TestTheQuantitiesJudgedWhereMade:
    """Each derived quantity is judged where it is made, from inputs every door accepts (each test red on its named
    mutant of mut_r5.py, the judge at that site removed; UB4's tree has none of these judges)."""

    def test_a_pooled_count_beyond_the_floats_is_refused_where_the_count_form_makes_it(self):
        """g_counts_judged: a pooled census's mean core of 1e200 x 1e200 leaves the floats in the count form."""
        core = _module(f"{PKG}.core.equation")
        org = get_platform_physics("loihi").area_rule.organization
        with pytest.raises(AreaRuleRefusal, match="the count form's bits"):
            core.scaled_counts("bits", org, 1e200, 1e200, 8, True)

    def test_an_anchor_ratio_that_underflows_to_zero_is_refused_where_it_is_made(self):
        """g_ratio: a pooled mean core of the floats' least size gives a ratio of 0, no size above 0."""
        org = get_platform_physics("loihi").area_rule.organization
        with pytest.raises(AreaRuleRefusal, match="the anchor's ratio"):
            anchor.anchor_ratio(org, rule.CensusCore(5e-324, 5e-324, 8, pooled=True))

    def test_an_anchor_core_band_beyond_the_floats_is_refused_where_it_is_made(self):
        """g_core_band: a published core and a ratio of 1e300 each make an infinite band."""
        with pytest.raises(AreaRuleRefusal, match="the anchor's core band"):
            anchor.anchor_core_band(1e300, 1e300, None)

    def test_a_pooled_census_of_no_cells_is_refused_where_its_mean_core_is_made(self):
        """g_pooled_mean: a census of 0 cells (a count of at least 0) makes a mean core of 0 rows."""
        q = Quantities({k: QuantityValue(float(v), "static") for k, v in
                        {"cores_physical": 4, "cells_physical": 0, "neurons_physical": 256, "weight_bits": 4}.items()})
        with pytest.raises(AreaRuleRefusal, match="pooled census: its mean core's rows"):
            pricing.census_types(q, None, 4)

    def test_a_parity_row_is_judged_where_it_is_made(self):
        """g_row: a row of a published area below 0 is made silently."""
        with pytest.raises(AreaRuleRefusal, match="the row of 'x'"):
            parity.Row("x", "T1", "core", True, "c", False, False, -1.0)

    def test_a_block_whose_own_text_cannot_be_written_is_never_made(self):
        """g_drops: a block with drops of no texts is made, and its own text (to_dict) then fails to write."""
        block = get_platform_physics(GENERIC).area_rule
        with pytest.raises(AreaRuleRefusal, match="drops"):
            replace(block, drops=math.inf)
        assert replace(block, drops=("a dropped row",)).to_dict()["drops"] == ["a dropped row"]

    def test_a_reading_law_without_its_bits_is_a_named_refusal(self):
        """g_law_bits: a reading (check False) at no stated bits reaches the law's 2 ** (alpha (None - ref))."""
        with pytest.raises(AreaRuleRefusal, match="declares none"):
            rule.converter_area(registry.load_registry(), "sar_dedicated", None, (), False)

    def test_a_published_split_outside_a_share_names_its_expression_in_the_refusal(self):
        """e_split_raw: N-3's path keeps the share's refusal, which names the expression the split read."""
        block = get_platform_physics("hermes_64").area_rule
        split = {"basis": "published_split", "expression": "r('types/t4:s') * 1000"}
        law = anchor.anchor_law(registry.load_registry(), replace(block, anchor={**block.anchor, "own_share": split}))
        assert law.own_share.value is None and "own_share.expression" in law.own_share.refusal, law.own_share


class TestTheLoadersRefusalClass:
    """The loader's refusal of a typed figure is its declaration error (AreaRuleDeclarationError, the class v8
    records for the uncited cell, UB4's note): each red on its named mutant, where the table's judge alone refuses."""

    def test_a_typed_declared_cell_is_a_declaration_error_at_load(self):
        """r_N_C2: judged_block's declared-cell judge removed."""
        block = get_platform_physics(GENERIC).area_rule
        typed = replace(block, node={**block.node, "bitcell_um2": 0.1})
        with pytest.raises(declaration.AreaRuleDeclarationError, match="bitcell_um2"):
            declaration.judge_block(typed, registry.load_registry())

    def test_a_typed_published_size_is_a_declaration_error_at_load(self):
        """e_load_published: the load judge's citation of the published size removed."""
        block = get_platform_physics("isaac_like").area_rule
        typed = replace(block, published_size={**block.published_size, "n_pub": 168})
        with pytest.raises(declaration.AreaRuleDeclarationError, match="published_size.n_pub"):
            declaration.judge_block(typed, registry.load_registry())


# ---------------------------------------------------------------------------------------------------------- the door
def _public_functions():
    """(module path, qualified name) of every public function and method of the pricing modules, by AST."""
    out = []
    for rel in PRICING_MODULES:
        tree = ast.parse((AR / rel).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                out.append((rel, node.name))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                out += [(rel, f"{node.name}.{item.name}") for item in node.body
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")]
    return out


def _resolve(rel, qualname):
    module = _module(f"{PKG}.{rel[:-3].replace('/', '.')}")
    obj = module
    for part in qualname.split("."):
        obj = inspect.getattr_static(obj, part) if inspect.isclass(obj) else getattr(obj, part)
    return obj.fget if isinstance(obj, property) else getattr(obj, "__func__", obj)


class TestTheOneDoor:
    def test_every_public_function_of_the_rule_is_a_door_or_named_with_its_reason(self):
        """Red on UB4's tree (no door): a public function of a pricing module that reads a caller's value unjudged."""
        door = _module(DOOR)
        found, named = _public_functions(), door.NOT_DOORS
        for rel, qualname in found:
            fn = _resolve(rel, qualname)
            if qualname in named.get(rel, {}):
                assert named[rel][qualname].strip() and not getattr(fn, "__door__", False), (rel, qualname)
            else:
                assert getattr(fn, "__door__", False), f"{rel}:{qualname} is neither a door nor named with its reason"
        for rel, names in named.items():
            assert set(names) <= {q for r, q in found if r == rel}, (rel, set(names))
        for qualname in REGISTRY_DOORS:
            assert getattr(_resolve("registry.py", qualname), "__door__", False), qualname
        assert len([f for f in found if f[1] not in named.get(f[0], {})]) == len(door.DOORS) - len(REGISTRY_DOORS)

    def test_every_argument_of_every_door_has_its_judge(self):
        """Red on UB4's tree (no table): a door's argument without a judge would reach the arithmetic unjudged."""
        door, arguments = _module(DOOR), _module(ARGUMENTS)
        for qualname, fn in door.DOORS.items():
            for name in inspect.signature(fn.__wrapped__).parameters:
                if name in ("self", "cls"):
                    continue
                judge = fn.__judges__.get(name, arguments.ARGUMENTS.get(name))
                assert judge in arguments.JUDGES, (qualname, name, judge)

    def test_the_loaders_judge_and_the_doors_judge_are_one_call(self, monkeypatch):
        """Red on UB4's tree (no door): the loader judges the block, its validity geometry and the rows beside the
        block through the one function, and the price's door judges the same three through the same judges (the
        fourth delta round, MA-C2: the rows beside the block are judged at the price as at load; red on UB5's tree,
        where the price judged the block and the validity geometry only)."""
        door, arguments = _module(DOOR), _module(ARGUMENTS)
        calls, real = [], door.judged_input
        monkeypatch.setattr(door, "judged_input", lambda args, *a, **k: calls.append(set(args)) or real(args, *a, **k))
        judges = set()
        for name in ("block", "validity geometry", "beside"):
            monkeypatch.setitem(arguments.JUDGES, name, lambda *a, _n=name, _j=arguments.JUDGES[name], **k: (
                judges.add(_n), _j(*a, **k))[1])
        physics = store_declared(profile_from_dict(json.loads((PROFILES / "loihi.json").read_text())))
        at_load, load_calls = set(judges), list(calls)
        judges.clear()
        calls.clear()
        door.forget(registry.load_registry())
        assert _price(physics, (64, 64, 512, 8))[0] == "priced"
        assert {"block", "validity", "beside"} <= set().union(*load_calls), load_calls
        assert any("physics" in args for args in calls), calls
        assert at_load == {"block", "validity geometry", "beside"} == judges, (at_load, judges)

    def test_the_door_judges_a_frozen_input_once_and_a_changed_one_again(self, monkeypatch):
        """Red on UB4's tree (no door): the cost bound, a memo keyed by the frozen input, so a price judges a shipped
        block once; a changed block is judged again. The fourth delta round keys a frozen block by its identity (no
        pickle), so a block cannot change in place (its holders are frozen when it is made: a TypeError, red on UB5's
        tree, where the change in place was a new pickled text); changed by replace, it is a new input."""
        door, arguments = _module(DOOR), _module(ARGUMENTS)
        count, real = [], arguments.JUDGES["block"]
        monkeypatch.setitem(arguments.JUDGES, "block", lambda *a, **k: count.append(1) or real(*a, **k))
        physics = priced_physics("odin")
        door.forget(registry.load_registry())
        for _ in range(3):
            assert _price(physics, (1, 256, 256, 4))[0] == "priced"
        assert len(count) == 1
        block = replace(physics.area_rule, organization=copy.deepcopy(dict(physics.area_rule.organization)))
        changed = replace(physics, area_rule=block)
        assert _price(changed, (1, 256, 256, 4))[0] == "priced"
        with pytest.raises(TypeError):
            block.organization["rows"] = 0  # a frozen block's holder: no change in place
        changed = replace(physics, area_rule=replace(block, organization={**block.organization, "rows": 0}))
        assert _price(changed, (1, 256, 256, 4))[0] == "refused"
        assert len(count) == 3


# ---------------------------------------------------------------------------------------------------------- the sweep
#: Each holder of declared fields (a declaration's object or a door's argument), by the source texts each module of the
#: rule holds it under. A read is a subscript (load), a .get or an 'in' test with a constant text key, or an attribute
#: load that is not a called method. The answers the rule makes (a ratio, a band, a price, a flag) are no holder.
HOLDERS = {
    "block": {"anchor.py": {"block"}, "declaration.py": {"block", "self", "data"}, "pricing.py": {"block"},
              "typerule.py": {"block"}, "rule.py": {"block"}},
    "node": {"declaration.py": {"self.node", "self.node or {}", "block.node or {}"}},
    "anchor": {"anchor.py": {"a"}, "declaration.py": {"anchor"}, "pricing.py": {"block.anchor"}},
    "own share": {"anchor.py": {"spec"}, "declaration.py": {"share"}},
    "organization": {"core/equation.py": {"org"}, "anchor.py": {"org", "organization"},
                     "declaration.py": {"organization"}},
    "converter": {"declaration.py": {"converter", "block.converter or {}"}, "pricing.py": {"block.converter or {}"}},
    "reading": {"typerule.py": {"block.reading_low_edge or {}"}, "declaration.py": {"block.reading_low_edge"}},
    "published size": {"pricing.py": {"block.published_size"}, "declaration.py": {"block.published_size"}},
    "physics": {"pricing.py": {"physics"}, "core/arguments.py": {"physics"}},
    "physics validity": {"pricing.py": {"physics.validity"},
                         "core/arguments.py": {"value.validity", "physics.validity"}},
    "validity geometry": {"pricing.py": {"validity", "own_core"}},
    "census core": {"anchor.py": {"census"}, "core/equation.py": {"census"}, "typerule.py": {"census"},
                    "pricing.py": {"census"}},
    "core type": {"pricing.py": {"core"}},
    "law": {"anchor.py": {"law"}, "core/fields.py": {"law"}},
    "own share answer": {"anchor.py": {"law.own_share", "share"}},
    "declaration": {"core/equation.py": {"declared"}},
    "counts": {"core/equation.py": {"counts"}},
    "point": {"published/parity.py": {"point"}, "published/decomposition.py": {"point"},
              "core/points.py": {"point"}},
    "point organization": {"published/parity.py": {"org", 'point["organization"]'},
                           "core/points.py": {"point_org", 'point.get("organization") or {}'}},
    "point converter": {"published/parity.py": {"conv", 'point["converter"]', 'point.get("converter") or {}'},
                        "core/points.py": {"point_conv", 'point.get("converter") or {}'}},
    "memory beside": {"published/parity.py": {"beside"}},
    "scope": {"published/parity.py": {"scope", "primary", "s"}, "core/points.py": {"sc"}},
    "fixed part": {"published/parity.py": {"part"}},
    "array figure": {"published/decomposition.py": {"figure"}, "core/points.py": {"figure", "figure or {}"}},
    "core terms": {"published/parity.py": {"terms"}, "published/decomposition.py": {"row.terms"},
                   "anchor.py": {"terms"}, "core/equation.py": {"terms"}},
    "row": {"published/decomposition.py": {"row", "r"}, "core/points.py": {"row", "r"},
            "published/parity.py": {"r"}},
    "stack": {"published/decomposition.py": {"s", "worst"}, "core/points.py": {"stack"}},
    # the fourth delta round (the check's MA-C3): a row's readings, which strict_spread reads
    "parity reading": {"published/parity.py": {"reading"}, "core/points.py": {"reading"}},
}


def _sweep():
    """{holder: {field: [module:line]}} over every module of the rule (ast only)."""
    reads = {h: {} for h in HOLDERS}
    for path in sorted(AR.rglob("*.py")):
        rel = path.relative_to(AR).as_posix()
        text = path.read_text()
        tree = ast.parse(text)
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            held = key = None
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and \
                    isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                held, key = node.value, node.slice.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and \
                    node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                held, key = node.func.value, node.args[0].value
            elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and \
                    isinstance(node.left.value, str) and isinstance(node.ops[0], (ast.In, ast.NotIn)):
                held, key = node.comparators[0], node.left.value
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and id(node) not in called and \
                    not node.attr.startswith("_"):
                held, key = node.value, node.attr
            if held is None:
                continue
            source = ast.get_source_segment(text, held)
            for holder, texts in HOLDERS.items():
                if source in texts.get(rel, ()):
                    reads[holder].setdefault(key, []).append(f"{rel}:{node.lineno}")
    return reads


class TestTheStructuralSweep:
    def test_every_read_of_a_declared_field_is_a_field_of_the_judges_table(self):
        """Red on UB4's tree (no table): the validity geometry's cell_bits, axons and neurons are read
        (pricing.py:74, :143) and judged nowhere (MA-K1). A field read later fails here until it has its judge."""
        fields = _module(FIELDS)
        reads = _sweep()
        missing = {h: sorted(set(r) - set(fields.FIELDS[h])) for h, r in reads.items()}
        assert not any(missing.values()), missing
        assert all(reads.values()), {h: bool(r) for h, r in reads.items()}  # every holder is read somewhere
        assert sum(len(v) for r in reads.values() for v in r.values()) > 150
        # the census quantities are read by a name the code holds (pricing.POOLED_TOTALS and the weight bits)
        assert set(fields.FIELDS["census quantities"]) == set(pricing.POOLED_TOTALS) | {"weight_bits"}


# ---------------------------------------------------------------------------------------------------------- my family
#: The family's values for any argument, and for each field of a structured argument.
FAMILY_BAD = (None, 0, -1, -0.0, 0.5, True, "4", "", [4], {}, math.inf, math.nan, 10 ** 400, 2 ** 53 + 1,
              sys.float_info.max, 5e-324)
FIELD_BAD = (None, 0, -1, 0.5, True, "4", [4], {"x": 1}, math.inf, math.nan, 2 ** 53 + 1)


def _valid(result, depth=0):
    """A door's answer is of its kind: every number finite and not negative, a band positive and ordered."""
    if depth > 6 or result is None or isinstance(result, (str, bool)):
        return True
    if isinstance(result, (int, float)):
        return math.isfinite(result) and result >= 0
    if isinstance(result, rule.AreaPrice):
        if not (0.0 < result.low <= result.nominal <= result.high and math.isfinite(result.high)):
            return False
    if isinstance(result, pricing.RuleArea) and result.band is not None:
        low, nominal, high = result.band
        if not (0.0 < low <= nominal <= high and math.isfinite(high)):
            return False
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return all(_valid(getattr(result, f.name), depth + 1) for f in dataclasses.fields(result))
    if isinstance(result, tuple) and hasattr(result, "_fields"):
        return all(_valid(v, depth + 1) for v in result)
    if isinstance(result, dict):
        return all(_valid(v, depth + 1) for v in result.values())
    if isinstance(result, (list, tuple)):
        return all(_valid(v, depth + 1) for v in result)
    return True


def _variants(value):
    """The family's changes of one argument: each bad value, and each field of a structured value changed."""
    out = [(repr(v)[:40], v) for v in FAMILY_BAD]
    if isinstance(value, dict):
        for key in list(value)[:14]:
            out += [(f"[{key!r}] = {v!r}"[:60], {**value, key: v}) for v in FIELD_BAD]
            out.append((f"[{key!r}] dropped", {k: x for k, x in value.items() if k != key}))
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for name in value._fields:
            out += [(f".{name} = {v!r}"[:60], value._replace(**{name: v})) for v in FIELD_BAD]
    elif dataclasses.is_dataclass(value) and not isinstance(value, type) and type(value).__name__ in (
            "AreaRuleBlock", "AnchorLaw", "ValidityGeometry", "Row", "PlatformPhysics"):
        for f in dataclasses.fields(value):
            if f.name == "source":
                continue
            for v in FIELD_BAD:
                try:
                    out.append((f".{f.name} = {v!r}"[:60], replace(value, **{f.name: v})))
                except Exception:  # noqa: BLE001 - a value its own class refuses when made is no input of a door
                    continue
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], tuple) and len(value[0]) == 2:
        count, core = value[0]
        out += [(f"[0] count {v!r}"[:60], [(v, core), *value[1:]]) for v in FIELD_BAD]
        out += [(f"[0] core {label}", [(count, v), *value[1:]]) for label, v in _variants(core)[len(FAMILY_BAD):]]
    return out


def _base_calls():
    """The doors' base calls: every door's bound arguments as a run of shipped prices calls it (a profile hook on
    the doors' own code), plus the direct calls the shipped prices do not make. The door's memo is forgotten first
    (the fourth delta round: a judgement remembered by an earlier test of the worker would hide a door's call)."""
    door = _module(DOOR)
    door.forget(registry.load_registry())
    getattr(door, "forget_domain", lambda: None)()
    codes = {getattr(fn, "__wrapped__", fn).__code__: qualname for qualname, fn in door.DOORS.items()}
    calls = {}

    def hook(frame, event, _arg):
        if event == "call" and frame.f_code in codes:
            name = codes[frame.f_code]
            if len(calls.setdefault(name, [])) < 2:
                calls[name].append(dict(inspect.getargvalues(frame).locals))

    reg = registry.load_registry()
    sys.setprofile(hook)
    try:
        for name in ANCHORS:
            physics = priced_physics(name)
            org = physics.area_rule.organization
            for cen in ((4, max(1, org["rows"] // 2), org["cols"], 2 * org["weight_bits_per_cell"]),):
                _price(physics, cen)
            law = anchor.anchor_law(reg, physics.area_rule)
            anchor.price_anchor(law, 4, rule.CensusCore(org["rows"], org["cols"], org["weight_bits_per_cell"]))
            anchor.device_rows(org, 128)
        generic = priced_physics(GENERIC)
        for block in (generic.area_rule, replace(generic.area_rule, type="T3")):
            _price(replace(generic, area_rule=block), (100, 256, 256, 4))
            _price(replace(generic, area_rule=block), (100, 256, 256, 4, "no_bits"))
            typerule.price_block(reg, block, 10, rule.CensusCore(256, 256, 4), 8, 8)
            typerule.price_type_rule(reg, block, 10, rule.CensusCore(256, 256, 4), 8, 8)
        _price(priced_physics("isaac_like"), (16128, 128, 128, 2))
        rows = parity.estimate(reg, "held_out")
        parity.headline(rows)
        parity.strict_form(parity.estimate(reg, "strict"))
        parity.strict_spread(rows)
        parity.stats([r.ratio for r in parity.headline(rows) if r.refused is None])
        decomposition.sram_array_fact(decomposition.array_decomposition(reg, rows))
        rule.wizard_also(generic.area_rule)
        reg.sigma(28.0)
        reg.evaluate(CELL)
        # the doors a shipped price reaches only off its descriptions' own declarations: a cited cell, a reading
        cited = replace(generic.area_rule, node={**generic.area_rule.node, "bitcell_um2": CELL})
        declaration.declared_cell_flags(reg, cited)
        typerule.reads_declared_cell(reg, cited)
        for kind in ("sram_6t_push_rule", "rram_1t1r", "pcm_analog_unit_cell", "cim_10t1c_mom"):
            rule.reads_sigma(reg, kind)
            rule.cell_area(reg, kind, 28.0, (), 0.1, True)
        rule.outside_rule("declared_parts", None, {"profile": 3.0})
        rule.outside_gamma("type", 0.8)
        rule.held_outside_part(reg, "T1")
        rule.check_bits("sar_dedicated", 8, 7, 10)
        rule.converter_area(reg, "sar_dedicated", 8, (), False)
        decomposition.scope_class(reg, ["T1-TN-04"])
        parity.factor(0.5)
    finally:
        sys.setprofile(None)
    return calls


@pytest.fixture(scope="module")
def base_calls():
    return _base_calls()


class TestTheFamilyOverEveryDoor:
    @pytest.mark.parametrize("rel", PRICING_MODULES + ("registry.py",))
    def test_every_argument_of_every_door_with_bad_values_is_a_named_refusal_or_a_value_of_its_kind(self, rel,
                                                                                                      base_calls):
        """Red on UB4's tree (no door): TypeError, ValueError, KeyError, AttributeError, OverflowError and
        ZeroDivisionError out of the rule's public functions, and answers of no finite value."""
        door = _module(DOOR)
        module = f"{PKG}.{rel[:-3].replace('/', '.')}"
        doors = {q: fn for q, fn in door.DOORS.items() if fn.__module__ == module}
        assert doors, rel
        assert all(base_calls.get(q) for q in doors), [f"{q}: no base call" for q in doors if not base_calls.get(q)]
        defects, tried = [], 0
        for qualname, fn in doors.items():
            for base in base_calls[qualname]:
                for param, value in base.items():
                    if param in ("self", "cls"):
                        continue
                    for label, variant in _variants(value):
                        args = {**base, param: variant}
                        target = getattr(args.pop("self"), fn.__name__) if "self" in args else fn
                        tried += 1
                        try:
                            got = target(**args)
                        except Exception as caught:  # noqa: BLE001 - every exception is recorded
                            if _crash(caught) is not None:
                                defects.append(f"{qualname}({param} {label}): {_crash(caught)[:140]}")
                            continue
                        if not _valid(got):
                            defects.append(f"{qualname}({param} {label}): an answer of no kind {repr(got)[:120]}")
        assert tried > 100 and not defects, (tried, len(defects), defects[:25])


def test_the_shipped_descriptions_and_points_pass_the_door():
    """A guard: every shipped description loads and its block passes the door; the published points pass it."""
    reg = registry.load_registry()
    for name in _shipped():
        declaration.judge_block(get_platform_physics(name).area_rule, reg)
    assert len(parity.estimate(reg, "in_sample")) > 15
