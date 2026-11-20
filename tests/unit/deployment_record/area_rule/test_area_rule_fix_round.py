"""[U-AREA] The fix round of the build (UB/verify.md: MA-1, MA-2, MA-3, mi-1, mi-3, mi-4), tests first: each test is
red on the build's tree (UB/uarea.patch on R5, applied digest 9d4ca82c) and on its named mutant (UB2/scripts/mut_r2.py),
green on the fixed tree. Guards (green on the build's tree too) are named as such.

- MA-1: a type rule of every type prices through the absolute pricer; a type table that states no outside part makes
  the fixed-outside reading a named refusal beside the price, never a TypeError.
- MA-2: a declared bitcell is an expression over evidence ids (judged as an anchor's figures are), refused by name when
  typed, and the price that rests on it carries the flag ``declared_cell`` with its citation, at every site that reads
  the description's node cell (a type rule's cells, an anchor's own memory share).
- MA-3: one core-area rule: the held-out estimate prices every published core by the type rule's own equation
  (``typerule.core_terms``) on the registry viewed without the chip; every parity row equals the first build's to
  1e-12 (data/parity_build1.json, written by UB2/scripts/c01_parity_capture.py on the build's tree); no module names a
  technology type or a cell or converter kind (T-10 extended).
- mi-1: the rule's expressions are parsed once: a second call parses nothing.
- mi-3: a bound converter kind refuses by name where the run declares no bits.
- mi-4: the type map, the logic-rule reference kind, the law's kind, the SRAM family and the area axes are read from
  data or stated once.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import (
    declaration, evidence, pricing, registry, rule, typerule)
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal
from mimarsinan.deployment_record.platform_physics.area_rule.published import parity
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G

AREA = "chip_area_mm2"
PACKAGE = Path(registry.__file__).resolve().parent
CORES, ROWS, COLS, WEIGHT_BITS, READOUT_BITS = 1000, 256, 256, 4, 8


def _types():
    """The technology types, read from the type tables' own 'type' fields (data, never a list in a test)."""
    return sorted(json.loads(p.read_text())["type"] for p in (registry.DATA / "types").glob("*.json"))


def _census():
    values = {"cores_physical": CORES, "axons_physical": CORES * ROWS, "neurons_physical": CORES * COLS,
              "cells_physical": CORES * ROWS * COLS, "weight_bits": WEIGHT_BITS, "tiles": CORES}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _type_rule_physics(type_name, **changes):
    """The generic description's type rule read as ``type_name`` (no reading beside it), with ``changes`` to its block."""
    physics = get_platform_physics("generic_estimated_22nm")
    block = replace(physics.area_rule, type=type_name, reading_low_edge=None, **changes)
    return replace(physics, area_rule=block)


def _price(physics, readout_bits=READOUT_BITS):
    pricing_ = price_absolute(_census(), physics, geometry=[CoreGeometry(ROWS, COLS, CORES)],
                              readout_bits=readout_bits)
    term = {t.name: t for t in pricing_.terms}.get(AREA)
    refusal = {r.name: r.reason for r in pricing_.refusals}.get(AREA)
    flags = {f.kind: f for f in pricing_.flags if f.name == AREA}
    return term, refusal, flags


def _data_copy(tmp_path, edit):
    data = tmp_path / "data"
    shutil.copytree(registry.DATA, data)
    edit(data)
    return registry.Registry.load(data)


def _edit_json(data, rel, change):
    path = data / rel
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------------- MA-1
class TestEveryTypeThroughThePricerMA1:
    @pytest.mark.parametrize("type_name", _types())
    def test_a_type_rule_of_every_type_prices_through_the_pricer(self, type_name):
        """Red on the build's tree for T4 (a TypeError out of price_absolute, rule.py:247); a guard for the others."""
        term, refusal, flags = _price(_type_rule_physics(type_name))
        assert refusal is None and term is not None, refusal
        reg = registry.load_registry()
        held = rule.held_outside_part(reg, type_name).value
        want = term.band.nominal * rule.type_gamma(reg, type_name) + held
        assert G.close(flags["fixed_outside_reading"].value, want)
        assert term.band.low <= term.band.nominal <= term.band.high

    def test_t4_holds_the_published_outside_part_of_the_die_its_gamma_reads(self):
        """MA-1's data: the HERMES die less its core grid (T4-HE-01, T4-HE-02), the part its gamma reads."""
        reg = registry.load_registry()
        die, core = reg.evidence.value("T4-HE-01"), reg.evidence.value("T4-HE-02")
        count = reg.evaluate("s('T4-HE-01', 'scope', '64 cores', 64)")
        part = rule.held_outside_part(reg, "T4")
        assert G.close(part.value, die - count * core)
        assert G.close(rule.type_gamma(reg, "T4"), count * core / die)

    def test_a_table_that_states_no_outside_part_makes_the_reading_a_named_refusal(self, tmp_path, monkeypatch):
        reg = _data_copy(tmp_path, lambda data: _edit_json(data, "types/t4.json",
                                                           lambda doc: doc.update(outside_parts=None)))
        monkeypatch.setattr(pricing, "load_registry", lambda: reg)
        term, refusal, flags = _price(_type_rule_physics("T4"))
        assert refusal is None and term is not None, refusal  # the price stands: the reading is beside the band
        refused = flags["fixed_outside_reading_refused"].reason
        assert "T4" in refused and "states no published outside part" in refused
        assert "fixed_outside_reading" not in flags


# ---------------------------------------------------------------------------------------------------------- MA-2
class TestDeclaredCellMA2:
    @staticmethod
    def _block(node):
        physics = get_platform_physics("generic_estimated_22nm")
        return replace(physics.area_rule, node=node)

    def test_an_uncited_declared_cell_is_a_named_refusal_at_load(self):
        block = self._block({"node_nm": 3, "bitcell_um2": 0.0199})
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, registry.load_registry())
        assert "bitcell_um2" in str(caught.value) and "cited by id" in str(caught.value)

    def test_an_uncited_declared_cell_is_a_named_refusal_at_the_pricer(self):
        physics = get_platform_physics("generic_estimated_22nm")
        physics = replace(physics, area_rule=self._block({"node_nm": 3, "bitcell_um2": "0.0199"}))
        term, refusal, _f = _price(physics)
        assert term is None and refusal is not None and "bitcell_um2" in refusal and "cited by id" in refusal

    def test_a_cited_declared_cell_prices_flagged_with_its_citation(self):
        declaration.judge_block(self._block({"node_nm": 3, "bitcell_um2": "v('S-SRAM-10')"}),
                                registry.load_registry())
        physics = get_platform_physics("generic_estimated_22nm")
        cited = replace(physics, area_rule=self._block({"node_nm": 3, "bitcell_um2": "v('S-SRAM-10')"}))
        at_ten = replace(physics, area_rule=self._block({"node_nm": 10}))  # sigma(10) is S-SRAM-10's cell alone
        term, refusal, flags = _price(cited)
        ten, _r, ten_flags = _price(at_ten)
        assert refusal is None, refusal
        assert G.close(term.band.nominal, ten.band.nominal) and G.close(term.band.low, ten.band.low)
        flag = flags["declared_cell"]
        assert "S-SRAM-10" in flag.reason and G.close(flag.value, registry.load_registry().evidence.value("S-SRAM-10"))
        assert "declared_cell" not in ten_flags
        assert "S-SRAM-10" in term.band.basis

    def test_an_anchor_reads_its_declared_cell_for_its_own_memory_share(self):
        """N-3's held-logic edge needs the anchor's own share; a cited cell makes it price where the node lies outside
        the series (the description's node cell is read at every site through one accessor)."""
        physics = get_platform_physics("loihi")
        block = replace(physics.area_rule, node={"node_nm": 4, "bitcell_um2": "v('S-SRAM-10')"})
        physics = replace(physics, area_rule=block)
        pricing_ = price_absolute(_census_of(physics, 128, 64, 512), physics,
                                  geometry=[CoreGeometry(64, 512, 128)])
        kinds = {f.kind for f in pricing_.flags if f.name == AREA}
        assert "band_edge_refused" not in kinds and "declared_cell" in kinds


def _census_of(physics, cores, rows, cols):
    g = physics.validity.geometry
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": g.cell_bits, "tiles": cores}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


# ---------------------------------------------------------------------------------------------------------- MA-3
def _rows_by_mode(reg):
    return {mode: parity.estimate(reg, mode) for mode in parity.MODES}


class TestOneCoreRuleMA3:
    def test_every_parity_row_equals_the_first_builds_to_1e_12(self):
        """A guard on the build's tree (its own values); red on a value mutant of the estimate."""
        want = json.loads((G.DATA / "parity_build1.json").read_text())
        reg = registry.Registry.load()
        got = _rows_by_mode(reg)
        bad = []
        for mode, rows in want["rows"].items():
            assert len(got[mode]) == len(rows), mode
            for w, r in zip(rows, got[mode]):
                key = f"{mode}:{w['chip']}|{w['scope']}"
                for name in ("chip", "type", "scope", "primary", "cls", "model", "scope_mismatch", "refused",
                             "refused_by"):
                    if getattr(r, name) != w[name]:
                        bad.append((key, name, getattr(r, name), w[name]))
                for name in ("published_mm2", "ratio", "array_um2", "eta", "gamma", "count"):
                    a, b = getattr(r, name), w[name]
                    if (a is None) != (b is None) or (a is not None and not G.close(a, b)):
                        bad.append((key, name, a, b))
                if (r.terms is None) != (w["terms"] is None):
                    bad.append((key, "terms", r.terms, w["terms"]))
                elif r.terms is not None:
                    bad += [(key, n, getattr(r.terms, n), v) for n, v in w["terms"].items()
                            if not G.close(getattr(r.terms, n), v)]
                if [x.label for x in r.readings] != [x["label"] for x in w["readings"]]:
                    bad.append((key, "reading labels", [x.label for x in r.readings], w["readings"]))
                else:
                    bad += [(key, x.label, x.ratio, y["ratio"]) for x, y in zip(r.readings, w["readings"])
                            if not (G.close(x.ratio, y["ratio"]) and G.close(x.converter_um2, y["converter_um2"])
                                    and x.type_default == y["type_default"])]
        assert not bad, bad[:10]

    def test_the_stacks_strict_form_and_stats_equal_the_first_builds(self):
        want = json.loads((G.DATA / "parity_build1.json").read_text())
        reg = registry.Registry.load()
        held, strict = parity.estimate(reg, "held_out"), parity.estimate(reg, "strict")
        for scope_rule, stacks in want["stacks"].items():
            got = parity.array_decomposition(reg, held, scope_rule=scope_rule)
            assert [s.chip for s in got] == [s["chip"] for s in stacks]
            for s, w in zip(got, stacks):
                for name in ("published_um2", "array_um2", "array_ratio", "moved_to_rest_um2", "total_um2",
                             "total_macro_scope_um2"):
                    a, b = getattr(s, name), w[name]
                    assert (a is None and b is None) or G.close(a, b), (scope_rule, s.chip, name, a, b)
            fact = parity.sram_array_fact(got)
            assert fact["chip"] == want["sram_array_fact"][scope_rule]["chip"]
            assert G.close(fact["max_deviation"], want["sram_array_fact"][scope_rule]["max_deviation"])
        form = parity.strict_form(strict)
        for name, value in want["strict_form"].items():
            assert form[name] == value if isinstance(value, (str, list, int)) else G.close(form[name], value), name
        stats = parity.stats([r.ratio for r in parity.headline(held) if r.refused is None])
        for name, value in want["stats_held_out"].items():
            assert G.close(stats[name], value), name

    def test_the_held_out_estimate_prices_its_cores_by_the_type_rule(self, monkeypatch):
        """One core-area rule: doubling the type rule's core doubles every priced core-scope ratio of the estimate."""
        real = typerule.core_terms
        monkeypatch.setattr(typerule, "core_terms", lambda *a, **k: (lambda t: t._replace(core_um2=2 * t.core_um2))(
            real(*a, **k)))
        doubled = {f"{r.chip}|{r.scope}": r for r in parity.estimate(registry.Registry.load(), "held_out")}
        monkeypatch.setattr(typerule, "core_terms", real)
        reg = registry.Registry.load()
        rows = {f"{r.chip}|{r.scope}": r for r in parity.estimate(reg, "held_out")}
        linear = [f"{p['name']}|{s['scope']}" for p in reg.published_points() for s in p["scopes"]
                  if not s.get("sub_arrays") and not s.get("fixed_mm2")]  # a scope linear in its core
        checked = [k for k in linear if rows[k].refused is None]
        assert len(checked) >= 10
        for key in checked:
            assert G.close(doubled[key].ratio, 2 * rows[key].ratio), key

    def test_no_type_or_kind_name_in_a_module(self):
        """T-10 extended (MA-3, mi-4): no string of a module (docstrings aside) names a technology type or a cell or
        converter kind; the code reads them from the registry's data."""
        reg = registry.Registry.load()
        names = set(_types()) | set(reg.files["cell_kinds"]["kinds"]) | set(reg.files["converter_kinds"]["kinds"])
        found = {}
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text())
            docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(
                n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    hits = [n for n in names if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(n)}(?![A-Za-z0-9_-])",
                                                          node.value)]
                    if hits:
                        found.setdefault(path.relative_to(PACKAGE).as_posix(), set()).update(hits)
        assert found == {}


# ---------------------------------------------------------------------------------------------------------- mi-1
class TestParsedOnceMi1:
    def test_a_second_rule_call_parses_nothing(self, monkeypatch):
        physics = get_platform_physics("generic_estimated_22nm")
        census, geometry = _census(), [CoreGeometry(ROWS, COLS, CORES)]
        first = pricing.rule_area(physics, census, geometry, None)
        calls = []
        real = ast.parse
        monkeypatch.setattr(evidence.ast, "parse", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        second = pricing.rule_area(physics, census, geometry, None)
        assert calls == [] and second.band == first.band

    def test_a_second_anchor_price_parses_nothing(self, monkeypatch):
        """An anchor's price reads its figures' evidence ids (the basis) on every call: parsed once, never again."""
        physics = get_platform_physics("truenorth")
        census, geometry = _census_of(physics, 4096, 256, 256), [CoreGeometry(256, 256, 4096)]
        first = price_absolute(census, physics, geometry=geometry)
        calls = []
        real = ast.parse
        monkeypatch.setattr(evidence.ast, "parse", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        second = price_absolute(census, physics, geometry=geometry)
        assert calls == [] and [t.value for t in second.terms] == [t.value for t in first.terms]

    def test_a_loaded_registry_evaluates_its_values_without_parsing(self, monkeypatch):
        """The registry's expressions are parsed at load: a fresh registry's evaluations parse nothing."""
        reg = registry.Registry.load()
        calls = []
        real = ast.parse
        monkeypatch.setattr(evidence.ast, "parse", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        expressions = [e for _p, obj in reg.walk() for e in [(obj.get("recompute") or {}).get("expression")] + [
            p.get("expression") for p in (obj.get("recompute") or {}).get("points") or () if isinstance(p, dict)]
            if isinstance(e, str)]
        for expression in expressions:
            try:
                reg.evaluate(expression)
            except (AreaRuleRefusal, evidence.HeldOut):
                pass
        assert len(expressions) > len(_types()) and calls == []

    def test_a_second_rule_call_evaluates_no_expression(self, monkeypatch):
        """The registry is immutable after load: a second call reads every value, evaluation and node cell from its
        memo (the evaluator is never entered again)."""
        physics = get_platform_physics("generic_estimated_22nm")
        census, geometry = _census(), [CoreGeometry(ROWS, COLS, CORES)]
        first = pricing.rule_area(physics, census, geometry, None)
        calls = []
        real = registry.evaluate
        monkeypatch.setattr(registry, "evaluate", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        second = pricing.rule_area(physics, census, geometry, None)
        assert calls == [] and second.band == first.band

    def test_a_second_estimate_parses_nothing(self, monkeypatch):
        reg = registry.Registry.load()
        first = parity.estimate(reg, "held_out")
        calls = []
        real = ast.parse
        monkeypatch.setattr(evidence.ast, "parse", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        second = parity.estimate(reg, "held_out")
        assert calls == [] and [r.ratio for r in second] == [r.ratio for r in first]


# ---------------------------------------------------------------------------------------------------------- mi-3
class TestBoundKindWithoutBitsMi3:
    def test_a_bound_kind_refuses_by_name_where_the_run_declares_no_bits(self):
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.converter_area(registry.load_registry(), "sar_bitcell_reusing", None)
        assert str(caught.value) == ("converter kind sar_bitcell_reusing: its area depends on the converter's bits "
                                     "(8 to 8) and the run declares none (readout_law.adc_bits)")

    def test_through_the_pricer_a_bound_kind_without_bits_is_a_named_refusal(self):
        physics = _type_rule_physics("T2_analog", cell_kind="cim_10t1c_mom",
                                     converter={"kind": "sar_bitcell_reusing", "per_core": 64})
        # the fourth delta round (MA-C2): a physics the loader accepts, its converter area stated once (the generic's
        # own area_per_adc dropped beside the block's converter kind)
        physics = replace(physics, constants={k: v for k, v in physics.constants.items() if k != "area_per_adc"})
        term, refusal, _f = _price(physics, readout_bits=None)
        assert term is None and "sar_bitcell_reusing" in refusal and "declares none" in refusal
        term, refusal, _f = _price(physics, readout_bits=8)  # a guard: at its bound's bits it prices at its bound
        assert refusal is None and term is not None


# ---------------------------------------------------------------------------------------------------------- mi-4
class TestNamesAsDataMi4:
    def test_the_type_map_is_read_from_the_tables(self, tmp_path):
        def edit(data):
            doc = json.loads((data / "types" / "t4.json").read_text())
            doc["type"] = "T5"
            (data / "types" / "t5.json").write_text(json.dumps(doc, indent=1) + "\n")
        reg = _data_copy(tmp_path, edit)
        assert "T5" in reg.types and reg.type_table("T5")["type"] == "T5"
        block = replace(get_platform_physics("generic_estimated_22nm").area_rule, type="T5", reading_low_edge=None)
        declaration.judge_block(block, reg)
        assert "T5" not in registry.load_registry().types

    def test_the_logic_rule_reference_kind_is_named_in_data(self, tmp_path):
        def edit(data):
            def change(doc):
                kinds = doc["kinds"]
                twin = json.loads(json.dumps(kinds["sram_6t_logic_rule"]))
                twin["factor_of_sigma"] = {"value": 2 * twin["factor_of_sigma"]["value"], "unit": "1",
                                           "evidence_kind": "derived", "citation": [], "derivation": "a test twin",
                                           "recompute": {"expression": f"2 * r('cell_kinds:kinds.sram_6t_logic_rule."
                                                                       f"factor_of_sigma')"}, "note": ""}
                kinds["sram_6t_logic_rule_twin"] = twin
                doc["logic_rule_reference"] = "sram_6t_logic_rule_twin"
            _edit_json(data, "cell_kinds.json", change)
        reg = _data_copy(tmp_path, edit)
        shipped = registry.load_registry()
        got = rule.cell_area(reg, "cim_8t_multiplying_bitcell_mom", 22.0)
        assert G.close(got, 2 * rule.cell_area(shipped, "cim_8t_multiplying_bitcell_mom", 22.0))

    def test_the_laws_refusal_names_its_kind(self, tmp_path):
        def edit(data):
            _edit_json(data, "converter_kinds.json",
                       lambda doc: doc["kinds"].update(sar_law_twin=doc["kinds"]["sar_dedicated"]))
        reg = _data_copy(tmp_path, edit)
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.converter_area(reg, "sar_law_twin", 8, ("T2-CR-", "T2-JA-"))
        assert str(caught.value) == "converter kind sar_law_twin: fewer than two published bit points remain"

    def test_the_sram_array_fact_reads_the_tables_cell_family(self, tmp_path):
        reg = _data_copy(tmp_path, lambda data: _edit_json(data, "types/t3.json",
                                                           lambda doc: doc.update(cell_family="sram")))
        fact = parity.sram_array_fact(parity.array_decomposition(reg, parity.estimate(reg, "held_out")))
        shipped = registry.load_registry()
        base = parity.sram_array_fact(parity.array_decomposition(shipped, parity.estimate(shipped, "held_out")))
        resistive = [p["name"] for p in reg.published_points() if p["type"] == "T3" and p.get("array_figure")]
        assert resistive and set(resistive) <= set(fact["chips"]) and not set(resistive) & set(base["chips"])

    def test_the_area_rule_axes_are_stated_once(self):
        from mimarsinan.deployment_record import fidelity_build
        from mimarsinan.gui.wizard import physics_panel
        assert physics_panel._AREA_RULE_AXES is fidelity_build.AREA_RULE_AXES
