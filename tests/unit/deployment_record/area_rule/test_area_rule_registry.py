"""[U-AREA] The registry of the area rule: the node series (T-1, T-2), the evidence check (T-9), the module and key
rules (T-10) and the evidence quotes (T-16). Expected values: the design's golden file (revision 3) through _golden.

Mutants named by the design (section 6) and caught here: the series without its range (T-2); the reference cell
replaced by the first cell (T-1); a fitted value without held_out (T-9); a literal chip constant in a module (T-10);
N-4: a typo in a quote-reading expression (T-9).
"""

from __future__ import annotations

import ast
import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from mimarsinan.deployment_record.platform_physics.area_rule import evidence, registry
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

from unit.deployment_record.area_rule import _golden as G

PACKAGE = Path(registry.__file__).resolve().parent
NODES_IN = (65.0, 45.0, 40.0, 32.0, 28.0, 22.0, 16.0, 14.0, 10.0, 7.0, 5.0)
NODES_OUT = (4.0, 3.0, 90.0, 130.0)
#: Chip and foundry names that may appear in no module and no registry key (u04's list of the design).
CHIP_NAMES = ("truenorth", "odin", "morphic", "reckon", "loihi", "jiang", "imagine", "cap-ram", "capram", "valavi",
              "neurram", "isaac", "puma", "hermes", "jia", "stellar", "intel", "tsmc")


class TestNodeSeriesT1:
    def test_the_reference_cells_equal_the_golden(self):
        reg = registry.load_registry()
        want = G.golden()["T-1 node_series"]
        bad = G.all_close((n, reg.sigma(n), want[f"{n:g}"]) for n in NODES_IN)
        assert not bad, bad

    def test_the_65_nm_reference_is_the_geometric_mean_of_its_two_cells(self):
        reg = registry.load_registry()
        assert G.close(reg.sigma(65.0), (0.57 * 0.5) ** 0.5)
        assert reg.node_reference_ids(65.0) == ("S-SRAM-65", "S-SRAM-65-valavi2019a")

    def test_mutant_reference_replaced_by_the_first_cell_is_caught(self, monkeypatch):
        reg = registry.Registry.load()
        monkeypatch.setattr(registry, "node_reference", lambda values: values[0])
        assert not G.close(reg.sigma(65.0), G.golden()["T-1 node_series"]["65"])


class TestNodeRangeT2:
    @pytest.mark.parametrize("node", NODES_OUT)
    def test_a_node_outside_the_series_is_a_named_refusal(self, node):
        reg = registry.load_registry()
        with pytest.raises(AreaRuleRefusal) as caught:
            reg.sigma(node)
        assert str(caught.value) == G.golden()["T-1 node_series"][f"{node:g}"]

    def test_a_declared_cell_prices_outside_the_series(self):
        reg = registry.load_registry()
        assert reg.sigma(4.0, declared_um2=0.0312) == 0.0312

    def test_mutant_series_without_its_range_is_caught(self, monkeypatch):
        reg = registry.Registry.load()
        monkeypatch.setattr(registry, "node_in_range", lambda node, low, high: True)
        value = reg.sigma(4.0)  # the mutant extrapolates below the series instead of refusing
        assert value > 0.0


class TestRegistryCheckT9:
    def test_the_registry_check_finds_no_problem(self):
        result = registry.check_registry(registry.load_registry())
        assert result.problems == ()
        assert result.chip_names_in_keys == ()
        assert result.values_checked >= 50 and result.recomputed >= 40

    def test_every_derived_and_fitted_value_recomputes(self):
        result = registry.check_registry(registry.load_registry())
        assert result.recomputed == result.derived_or_fitted

    def _mutated(self, tmp_path, edit):
        data = tmp_path / "data"
        shutil.copytree(registry.DATA, data)
        edit(data)
        return registry.check_registry(registry.Registry.load(data))

    def test_mutant_fitted_value_without_held_out_is_caught(self, tmp_path):
        def edit(data):
            path = data / "types" / "t1.json"
            doc = json.loads(path.read_text())
            del doc["eta"]["held_out"]
            path.write_text(json.dumps(doc))
        assert any("held-out" in p for p in self._mutated(tmp_path, edit).problems)

    def test_mutant_quote_reading_typo_is_caught(self, tmp_path):
        """N-4: the four literal derived values read their quote or entry; a typo in the text fails."""
        def edit(data):
            path = data / "cell_kinds.json"
            text = path.read_text().replace("about 38% larger", "about 83% larger")
            path.write_text(text)
        assert self._mutated(tmp_path, edit).problems

    def test_mutant_value_typo_is_caught(self, tmp_path):
        def edit(data):
            path = data / "converter_kinds.json"
            doc = json.loads(path.read_text())
            doc["kinds"]["cco_time_based"]["default"]["value"] = 410.0
            path.write_text(json.dumps(doc))
        assert any("recomputes" in p for p in self._mutated(tmp_path, edit).problems)

    def test_the_four_literal_values_read_their_quote_or_entry(self):
        """N-4: 1.38, 1.8, 1.0 and the CCO default are expressions over the evidence, never literals."""
        reg = registry.load_registry()
        for path in ("cell_kinds:kinds.cim_amortized_6t_charge_domain.factor_of_logic_6t",
                     "cell_kinds:kinds.cim_8t_multiplying_bitcell_mom.factor_of_logic_6t",
                     "cell_kinds:kinds.cim_logic_rule_6t_digital.factor_of_logic_6t",
                     "converter_kinds:kinds.cco_time_based.default"):
            expression = reg.object(path)["recompute"]["expression"]
            assert evidence.read_ids(expression), (path, expression)


def _module_sources():
    """Every module of the package and its sub-packages, by its path inside the package."""
    return {p.relative_to(PACKAGE).as_posix(): p.read_text() for p in sorted(PACKAGE.rglob("*.py"))}


def numeric_literals(source: str) -> list:
    """Every numeric literal of a module (booleans are not numbers)."""
    return [n.value for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex))
            and not isinstance(n.value, bool)]


def chip_names_in(text: str, names=CHIP_NAMES) -> list:
    return sorted({n for n in names if re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", text.lower())})


class TestModulesAndKeysT10:
    def test_no_numeric_literal_outside_the_units_module(self):
        found = {name: numeric_literals(src) for name, src in _module_sources().items() if name != "units.py"}
        assert {k: v for k, v in found.items() if v} == {}

    def test_the_units_module_holds_named_constants_only(self):
        tree = ast.parse(_module_sources()["units.py"])
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # the docstring
            assert isinstance(node, (ast.Assign, ast.AnnAssign, ast.ImportFrom)), ast.dump(node)[:80]

    def test_no_chip_name_in_a_module(self):
        names = CHIP_NAMES + tuple(p["name"].lower() for p in registry.load_registry().published_points())
        found = {name: chip_names_in(src, names) for name, src in _module_sources().items()}
        assert {k: v for k, v in found.items() if v} == {}

    def test_no_chip_name_in_a_registry_key(self):
        assert registry.check_registry(registry.load_registry()).chip_names_in_keys == ()

    def test_mutant_literal_chip_constant_in_a_module_is_caught(self):
        source = _module_sources()["rule.py"] + "\nTRUENORTH_CORE_MM2 = 0.0936\n"
        assert numeric_literals(source) and chip_names_in(source)


class TestEvidenceT16:
    def test_every_cited_id_is_an_evidence_entry(self):
        reg = registry.load_registry()
        assert reg.unknown_citations() == ()

    def test_a_quoted_figure_holds_its_number(self):
        """Every published (measured or layout) scalar figure the rule cites appears in its own quote."""
        missing = evidence.figures_without_their_number(registry.load_registry().evidence)
        assert missing == ()

    def test_every_page_text_reading_is_pinned_and_holds_its_number(self):
        ev = registry.load_registry().evidence
        assert ev.readings, "no page-text reading"
        for r in ev.readings:
            if r["kind"] == "page_text":
                assert re.fullmatch(r"[0-9a-f]{64}", r["page_sha256"]), r["id"]
                assert isinstance(r["line"], int) and r["text"] in r["line_text"], r["id"]
            if r.get("number_in_text") is not None:
                assert r["number_in_text"] in r["text"], r["id"]
                assert r["value"] == float(r["number_in_text"].replace(",", "")) * r["scale"], r["id"]

    def test_the_kind_basis_quotes_are_at_their_pages(self):
        ev = registry.load_registry().evidence
        ids = {r["id"] for r in ev.readings}
        assert set(ev.kind_basis_quotes) <= ids and len(ev.kind_basis_quotes) == 4

    def test_record_only_entries_are_in_their_pinned_records(self):
        """m16: S-SRAM-5, -7, -14, -16, -32, -65 against the digest-pinned asta records."""
        ev = registry.load_registry().evidence
        found = {r["id"]: r for r in ev.record_only}
        assert set(found) == {"S-SRAM-5", "S-SRAM-7", "S-SRAM-14", "S-SRAM-16", "S-SRAM-32", "S-SRAM-65"}
        assert all(r["quote_in_record"] and re.fullmatch(r"[0-9a-f]{64}", r["record_sha256"])
                   for r in found.values())

    def test_every_figure_states_its_array_scope_class(self):
        """T-26u's data: the second corrected set declares one array scope class per figure."""
        ev = registry.load_registry().evidence
        classes = set(ev.array_scope_classes)
        assert classes == {"bitcells", "macro_or_block", "none"}
        assert all(e["array_scope"]["class"] in classes for e in ev.entries.values() if "array_scope" in e)
