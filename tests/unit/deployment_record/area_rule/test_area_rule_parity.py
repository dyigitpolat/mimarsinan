"""[U-AREA] The held-out estimate of the published points: parity (T-6), named refusals (T-7), converter kinds
(T-22), MorphIC's core scope (T-23), the decomposition at the published scope (T-26u) and the strict form (ii) with
the band (T-27u). Expected values: the design's golden file (revision 3) through _golden.

Mutants named by the design and caught here: gamma dropped on a die (T-6); a chip pie used at chip scope (T-6); a
kind's bits unchecked (T-7); a declared kind substituted by the type default (T-7); an array compared at the macro
scope against a bitcells-only figure (T-26); form (i)'s readings mixed into form (ii) (T-27); N-5: a bitcell-reusing
converter beside another cell kind (T-22).
"""

from __future__ import annotations

import json
import math
import shutil

import pytest

from mimarsinan.deployment_record.platform_physics.area_rule import declaration, registry, rule
from mimarsinan.deployment_record.platform_physics.area_rule.published import parity
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

from unit.deployment_record.area_rule import _golden as G


def _rows(mode, reg=None):
    return {f"{r.chip}|{r.scope}": r for r in parity.estimate(reg or registry.load_registry(), mode)}


def _ratio_or_refusal(row):
    return row.ratio if row.refused is None else "refused: " + row.refused


class TestHeldOutParityT6:
    def test_every_row_equals_the_golden(self):
        want = G.golden()["T-6 held_out_parity_after"]
        rows = _rows("held_out")
        assert set(rows) == set(want)
        bad = []
        for key, w in want.items():
            got = _ratio_or_refusal(rows[key])
            if isinstance(w["loco"], str) or isinstance(got, str):
                if got != w["loco"]:
                    bad.append((key, got, w["loco"]))
            elif not G.close(got, w["loco"]):
                bad.append((key, got, w["loco"]))
        assert not bad, bad

    def test_ten_priced_two_named_refusals(self):
        rows = parity.estimate(registry.load_registry(), "held_out")
        head = parity.headline(rows)
        priced = [r for r in head if r.refused is None]
        assert len(priced) == 10
        assert sorted(r.chip for r in head if r.refused is not None) == ["IMAGINE", "Loihi 2"]
        want = G.golden()["T-6 held_out_stats_after"]["loco"]["primary_silicon_no_mismatch"]
        got = parity.stats([r.ratio for r in priced])
        for key in ("n", "geomean", "min", "max", "worst_factor", "within_2x", "within_3x"):
            assert G.close(got[key], want[key]), (key, got[key], want[key])

    def test_mutant_gamma_dropped_on_a_die_is_caught(self, monkeypatch):
        monkeypatch.setattr(parity, "die_gamma", lambda reg, type_name, excl: 1.0)
        rows = _rows("held_out", registry.Registry.load())
        want = G.golden()["T-6 held_out_parity_after"]["TrueNorth|die"]["loco"]
        assert not G.close(rows["TrueNorth|die"].ratio, want)

    def test_mutant_chip_pie_at_chip_scope_is_caught(self, tmp_path):
        data = tmp_path / "data"
        shutil.copytree(registry.DATA, data)
        path = data / "types" / "t3.json"
        doc = json.loads(path.read_text())
        for p in doc["s"]["recompute"]["points"]:
            p["expression"] = p["expression"].replace(" / r('types/t3:gamma')", "")
        path.write_text(json.dumps(doc))
        rows = _rows("held_out", registry.Registry.load(data))
        assert not G.close(rows["ISAAC (model)|tile"].ratio,
                           G.golden()["T-6 held_out_parity_after"]["ISAAC (model)|tile"]["loco"])


class TestNamedRefusalsT7:
    def test_strict_and_held_out_refusals_by_name(self):
        want = G.golden()["T-7 refusals"]
        reg = registry.load_registry()
        for mode, key in (("strict", "strict"), ("held_out", "loco")):
            head = parity.headline(parity.estimate(reg, mode))
            assert sorted(r.chip for r in head if r.refused is not None) == want[key], mode

    def test_strict_refusal_texts_equal_the_golden(self):
        want = G.golden()["T-6 held_out_parity_after"]
        rows = _rows("strict")
        for key in ("CAP-RAM|macro", "IMAGINE|macro", "NeuRRAM|die", "Loihi 2|core (vendor)"):
            assert _ratio_or_refusal(rows[key]) == want[key]["strict"], key

    def test_mutant_kind_bits_unchecked_is_caught(self, monkeypatch):
        monkeypatch.setattr(rule, "check_bits", lambda kind, bits, low, high: None)
        rows = _rows("strict", registry.Registry.load())
        assert rows["CAP-RAM|macro"].refused is None

    def test_mutant_declared_kind_substituted_by_the_type_default_is_caught(self, monkeypatch):
        real = rule.resolve_kind
        monkeypatch.setattr(rule, "resolve_kind", lambda reg, type_name, declared: real(reg, type_name, None))
        rows = _rows("held_out", registry.Registry.load())
        assert rows["IMAGINE|macro"].refused is None


class TestConverterKindsT22:
    def test_the_bitcell_reusing_bound_and_its_open_low_edge(self):
        reg = registry.load_registry()
        unit = rule.converter_area(reg, "sar_bitcell_reusing", 8)
        assert G.close(unit.value, G.golden()["T-22 converter kinds"]["bitcell_reusing_bound_um2"])
        assert unit.bound == "upper"

    def test_held_out_the_bitcell_reusing_kind_is_a_named_refusal(self):
        rows = _rows("held_out")
        assert _ratio_or_refusal(rows["IMAGINE|macro"]) == G.golden()["T-22 converter kinds"]["imagine_loco"]

    def test_in_sample_the_bitcell_reusing_kind_prices_at_its_bound(self):
        rows = _rows("in_sample")
        assert G.close(rows["IMAGINE|macro"].ratio, G.golden()["T-22 converter kinds"]["imagine_insample_ratio"])

    def test_the_cco_default_and_its_band_edge(self):
        reg = registry.load_registry()
        facts = G.moves()["design_facts"]
        assert G.close(rule.converter_area(reg, "cco_time_based", None).value, facts["cco_default_um2"])
        edge = reg.value("converter_kinds:kinds.cco_time_based.band_edge")
        assert G.close(edge, facts["cco_band_edge_um2"])

    def test_n5_the_bitcell_reusing_kind_requires_its_cell_kind(self):
        reg = registry.load_registry()
        block = declaration.AreaRuleBlock.from_dict({
            "type": "T2_analog", "terms": "type_rule", "outside": "type", "node": {"node_nm": 65},
            "cell_kind": "cim_amortized_6t_charge_domain",
            "converter": {"kind": "sar_bitcell_reusing", "per_core": 32}})
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(block, reg)
        assert "sar_bitcell_reusing" in str(caught.value) and "cim_10t1c_mom" in str(caught.value)

    def test_n5_mutant_pairing_unchecked_is_caught(self, monkeypatch):
        monkeypatch.setattr(declaration, "required_cell_kind", lambda reg, kind: None)
        block = declaration.AreaRuleBlock.from_dict({
            "type": "T2_analog", "terms": "type_rule", "outside": "type", "node": {"node_nm": 65},
            "cell_kind": "cim_amortized_6t_charge_domain",
            "converter": {"kind": "sar_bitcell_reusing", "per_core": 32}})
        declaration.judge_block(block, registry.load_registry())  # the mutant lets the pairing through

    def test_a_kind_outside_its_bits_is_a_named_refusal(self):
        reg = registry.load_registry()
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.converter_area(reg, "sar_dedicated", 6)
        assert str(caught.value) == "converter kind sar_dedicated: its law holds over 7 to 10 bits; 6 requested"


class TestMorphicCoreScopeT23:
    def test_the_core_scope_point_is_derived_from_its_two_figures(self):
        want = G.golden()["T-23 morphic_core_scope"]
        reg = registry.load_registry()
        point = next(p for p in reg.published_points() if p["name"] == "MorphIC")
        scope = next(s for s in point["scopes"] if s["primary"])
        assert G.close(reg.evaluate(scope["published_mm2"]), want["mm2"])
        assert scope["published_evidence_kind"] == "derived"
        assert sorted(scope["published_ids"]) == sorted(want["ids"])


class TestDecompositionT26u:
    def test_each_array_compares_at_its_published_scope(self):
        want = G.golden()["T-26 decomposition at the published scope"]
        reg = registry.load_registry()
        stacks = {s.chip: s for s in parity.array_decomposition(reg, parity.estimate(reg, "held_out"))}
        tn = stacks["TrueNorth"]
        assert tn.scope_class == "bitcells"
        assert G.close(tn.array_ratio, want["after"]["array_bitcell_scope_ratio"])
        assert G.close(tn.moved_to_rest_um2, want["after"]["moved_to_rest_um2"])
        assert G.close(tn.total_um2, tn.total_macro_scope_um2)
        others = [s for c, s in stacks.items() if c != "TrueNorth"]
        assert others and all(s.scope_class == "macro_or_block" for s in others)

    def test_the_largest_sram_array_deviation(self):
        reg = registry.load_registry()
        fact = parity.sram_array_fact(parity.array_decomposition(reg, parity.estimate(reg, "held_out")))
        want = G.moves()["design_facts"]["arrays_unit"]
        assert fact["chip"] == want["sram_max_chip"] and G.close(fact["max_deviation"], want["sram_max_deviation"])
        assert sorted(fact["chips"]) == sorted(want["sram_labels"])

    def test_mutant_macro_scope_against_a_bitcells_figure_is_caught(self):
        reg = registry.load_registry()
        rows = parity.estimate(reg, "held_out")
        fact = parity.sram_array_fact(parity.array_decomposition(reg, rows, scope_rule="macro"))
        want = G.moves()["design_facts"]["arrays_mutant_macro_scope"]
        assert fact["chip"] == want["sram_max_chip"] and G.close(fact["max_deviation"], want["sram_max_deviation"])


class TestStrictFormAndBandT27u:
    def test_form_ii_prices_eight_and_refuses_three_by_kind(self):
        want = G.golden()["T-27 strict forms"]["form_ii"]
        reg = registry.load_registry()
        form = parity.strict_form(parity.estimate(reg, "strict"))
        assert form["n_priced"] == want["n_priced"] and G.close(form["worst_factor"], want["worst_factor"])
        assert form["worst_chip"] == want["worst_chip"]
        assert form["refused_by_kind"] == want["refused_by_kind"]
        assert G.close(form["readings_max_factor"], want["readings_max_factor"])
        assert G.close(form["spread_min"], want["spread_min"]) and G.close(form["spread_max"], want["spread_max"])

    def test_the_pooled_strict_factors_are_the_form_ii_values(self):
        reg = registry.load_registry()
        spread = parity.strict_spread(parity.estimate(reg, "strict"))
        values = [v for x in spread.values() for v in x["values"]]
        low, high = reg.band_object("pooled")["value"]
        assert G.close(low, min(1 / max(values), reg.band_held_out_factors("pooled")[0]))
        assert G.close(high, max(1 / min(values), reg.band_held_out_factors("pooled")[1]))

    def test_the_registry_band_points_equal_the_unit_estimate(self):
        reg = registry.load_registry()
        spread = parity.strict_spread(parity.estimate(reg, "strict"))
        held = {r.chip: r.ratio for r in parity.headline(parity.estimate(reg, "held_out")) if r.refused is None}
        for p in reg.band_points():
            got = spread.get(p["chip"], {}).get("values", [])
            assert len(got) == len(p["strict"]) and all(G.close(a, b) for a, b in zip(got, p["strict"])), p["chip"]
            assert (p["held_out"] is None and p["chip"] not in held) or G.close(p["held_out"], held[p["chip"]])

    def test_mutant_form_i_readings_mixed_into_form_ii_is_caught(self):
        reg = registry.load_registry()
        spread = parity.strict_spread(parity.estimate(reg, "strict"), readings="all")
        values = [v for x in spread.values() for v in x["values"]]
        assert G.close(1 / min(values), G.golden()["T-27 strict forms"]["form_i_under_unit"][1])
