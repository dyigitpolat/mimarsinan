"""[U-AREA] The rule at the rule level: a missing parameter refuses by name (T-8), the band and the generic projection
(T-13), the outside field (T-19), an area_global_fixed beside gamma (T-20, the profile case), the wizard's naming
(T-21), the type rule's fixed-outside reading (T-28u) and the block's schema and judge. Expected values: the design's
golden file and the build's moves (_golden).

Mutants named by the design and caught here: area_global_fixed accepted beside gamma (T-20); the resident's
proportional outside part held for a type rule (T-28).
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics.area_rule import declaration, registry, rule, typerule
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

from unit.deployment_record.area_rule import _golden as G


def _block(name):
    return declaration.AreaRuleBlock.from_dict(G.declarations()[name])


class TestMissingParameterT8:
    def test_a_t3_description_at_6_bits_declaring_no_kind(self):
        """The type's default kind (the GS/s SAR) publishes 8-bit areas only: a named refusal, never a silent fit."""
        reg = registry.load_registry()
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.converter_area(reg, rule.resolve_kind(reg, "T3", None), 6)
        assert str(caught.value) == "converter kind sar_gsps publishes 8-bit areas only; 6 requested"

    def test_a_declared_kind_resolves_before_the_type_default(self):
        reg = registry.load_registry()
        assert rule.resolve_kind(reg, "T3", "neuron_integrating") == "neuron_integrating"
        assert rule.resolve_kind(reg, "T3", None) == "sar_gsps"
        assert rule.resolve_kind(reg, "T2_analog", None) == "sar_dedicated"
        assert rule.resolve_kind(reg, "T1", None) is None

    def test_an_unknown_kind_is_a_named_refusal(self):
        with pytest.raises(AreaRuleRefusal):
            rule.converter_area(registry.load_registry(), "sar_unpublished", 8)


class TestBandT13:
    def test_the_type_bands(self):
        want = G.golden()["T-13 band"]
        reg = registry.load_registry()
        assert all(G.close(a, b) for a, b in zip(rule.band_factors(reg, "T1"), (want["T1"]["L"], want["T1"]["H"])))
        for t in ("T2_analog", "T2_digital", "T3", "T4"):
            got = rule.band_factors(reg, t)
            assert G.close(got[0], want["pooled"]["L"]) and G.close(got[1], want["pooled"]["H"]), t

    def test_the_generic_projection_at_1000_cores(self):
        want = G.golden()["T-13 band"]["generic_1000_cores_8bit (lanes out)"]
        got = typerule.price_block(registry.load_registry(), _block("generic_estimated_22nm"), 1000,
                               rule.CensusCore(rows=256, cols=256, weight_bits=8))
        bad = G.all_close((f, getattr(got, f), want[f]) for f in ("nominal", "low", "high"))
        assert not bad, bad
        assert G.close(got.reading("t1_reading"), want["t1_reading"])
        assert [f.kind for f in got.flags if f.kind == "projection"] == ["projection"]

    def test_the_generic_projection_rows_of_the_design(self):
        reg = registry.load_registry()
        block = _block("generic_estimated_22nm")
        bad = []
        for row in G.moves()["design_facts"]["own_and_off_generic"]:
            got = typerule.price_block(reg, block, row["cores"], rule.CensusCore(rows=256, cols=256,
                                                                              weight_bits=row["weight_bits"]))
            bad += G.all_close((f"{row['cores']} {f}", getattr(got, f), row["after"][f])
                               for f in ("nominal", "low", "high"))
        assert not bad, bad

    def test_truenorth_at_512_cores(self):
        want = G.golden()["T-13 band"]["truenorth_512 (lanes out)"]
        got = typerule.price_block(registry.load_registry(), _block("truenorth"), 512,
                               rule.CensusCore(rows=256, cols=256, weight_bits=1))
        bad = G.all_close((f, getattr(got, f), want[f]) for f in ("nominal", "low", "high"))
        assert not bad, bad


class TestOutsideFieldT19:
    def test_declared_parts_take_gamma_1_and_the_fixed_part(self):
        got = rule.outside_rule("declared_parts", None, {"profile": 22.88, "override": None, "store": None})
        assert (got.gamma, got.fixed_mm2, got.fixed_from) == (1.0, 22.88, "profile")

    def test_an_override_replaces_the_profile_row_for_declared_parts(self):
        got = rule.outside_rule("declared_parts", None, {"profile": 22.88, "override": 30.0, "store": None})
        assert (got.fixed_mm2, got.fixed_from) == (30.0, "override")

    @pytest.mark.parametrize("outside", ("own_die", "type"))
    def test_own_die_and_type_take_gamma_and_no_fixed_part(self, outside):
        got = rule.outside_rule(outside, 0.8777, {"profile": None, "override": None, "store": None})
        assert (got.gamma, got.fixed_mm2) == (0.8777, 0.0)

    def test_declared_parts_without_any_part_keep_r5s_refusal(self):
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.outside_rule("declared_parts", None, {"profile": None, "override": None, "store": None})
        assert "neither chip_components nor area_global_fixed" in str(caught.value)

    @pytest.mark.parametrize("name,outside", [("truenorth", "own_die"), ("odin", "type"), ("isaac_like",
                                                                                         "declared_parts")])
    def test_the_blocks_declare_their_outside_field(self, name, outside):
        assert _block(name).outside == outside


class TestStatedTwiceT20RuleLevel:
    @pytest.mark.parametrize("place", ("profile", "override", "store"))
    def test_an_area_global_fixed_beside_gamma_is_a_named_refusal(self, place):
        fixed = {"profile": None, "override": None, "store": None, place: 46.6}
        with pytest.raises(AreaRuleRefusal) as caught:
            rule.outside_rule("own_die", 0.8916, fixed)
        assert str(caught.value) == f"the outside part is stated twice: gamma (own_die) and area_global_fixed ({place})"

    def test_the_judge_refuses_a_block_beside_an_area_global_fixed(self):
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(_block("truenorth"), registry.load_registry(), fixed_stated={"profile": 46.6})
        assert "stated twice" in str(caught.value)

    def test_mutant_accepted_beside_gamma_is_caught(self, monkeypatch):
        monkeypatch.setattr(rule, "stated_fixed_parts", lambda fixed: {})
        got = rule.outside_rule("own_die", 0.8916, {"profile": 46.6, "override": None, "store": None})
        assert got.fixed_mm2 == 0.0  # the mutant silently drops the stated part


class TestWizardT21RuleLevel:
    def test_without_a_block_the_axis_names_area_global_fixed(self):
        assert rule.wizard_also(None) == "area_global_fixed"

    @pytest.mark.parametrize("name,expected", [("truenorth", "area_rule.anchor.die"), ("odin", "area_rule.type"),
                                               ("isaac_like", "area_global_fixed")])
    def test_with_a_block_the_axis_names_its_outside_source(self, name, expected):
        assert rule.wizard_also(_block(name)).startswith(expected)


class TestTypeRuleReadingT28u:
    def test_the_held_outside_part_is_the_largest_published_part(self):
        want = G.golden()["T-28 type-rule fixed-outside reading"]
        part = rule.held_outside_part(registry.load_registry(), "T2_digital")
        assert G.close(part.value, want["held"]["mm2"])
        assert sorted(part.parts) == sorted(p["die"] for p in want["parts"])

    def test_the_reading_is_the_core_grid_plus_the_held_part(self):
        want = G.golden()["T-28 type-rule fixed-outside reading"]["rows"]
        ladder = {r["row"]: r for r in G.moves()["design_facts"]["ladder"]}
        reg = registry.load_registry()
        block = _block("generic_estimated_22nm")
        bad = []
        for row, w in want.items():
            r = ladder[row]
            rows, cols = (int(x) for x in r["geometry"].split("x"))
            got = typerule.price_block(reg, block, r["cores"], rule.CensusCore(rows=rows, cols=cols,
                                                                            weight_bits=r["weight_bits"]))
            bad += G.all_close([(f"{row} nominal", got.nominal, w["nominal"]),
                                (f"{row} reading", got.reading("fixed_outside_reading"), w["fixed_reading"])])
            assert got.low <= got.nominal <= got.high
        assert not bad, bad

    def test_the_reading_is_never_an_edge(self):
        reg = registry.load_registry()
        got = typerule.price_block(reg, _block("generic_estimated_22nm"), 242, rule.CensusCore(rows=256, cols=64,
                                                                                            weight_bits=4))
        reading = got.reading("fixed_outside_reading")
        assert reading not in (got.low, got.high)

    def test_mutant_resident_part_held_is_caught(self):
        """MA-3: the resident's proportional outside part held for a type rule gives B2 x4.31, not x7.46."""
        want = G.golden()["T-28 type-rule fixed-outside reading"]["rows"]
        b0, b2 = want["C1.T-INT.b.B0.256x256"], want["C1.T-INT.b.B2.256x256"]
        right = b0["fixed_reading"] / b2["fixed_reading"]
        resident_part = b0["nominal"] - b0["nominal"] * rule.type_gamma(registry.load_registry(), "T2_digital")
        mutant = (b0["fixed_reading"] - G.golden()["T-28 type-rule fixed-outside reading"]["held"]["mm2"]
                  + resident_part) / (b2["fixed_reading"] - G.golden()["T-28 type-rule fixed-outside reading"]
                                      ["held"]["mm2"] + resident_part)
        assert not G.close(mutant, right, 1e-3)


class TestBlockSchema:
    @pytest.mark.parametrize("name", sorted(G.declarations()))
    def test_every_build_block_is_judged_clean_and_round_trips(self, name):
        block = _block(name)
        declaration.judge_block(block, registry.load_registry())
        assert declaration.AreaRuleBlock.from_dict(block.to_dict()) == block

    def test_an_unknown_field_is_a_named_refusal(self):
        data = dict(G.declarations()["odin"], colour="blue")
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.AreaRuleBlock.from_dict(data)
        assert "colour" in str(caught.value)

    def test_an_unknown_type_is_a_named_refusal(self):
        data = dict(G.declarations()["odin"], type="T9")
        with pytest.raises(AreaRuleRefusal):
            declaration.judge_block(declaration.AreaRuleBlock.from_dict(data), registry.load_registry())

    def test_an_anchor_figure_must_read_the_evidence(self):
        data = dict(G.declarations()["odin"])
        data["anchor"] = dict(data["anchor"], core_mm2="0.0864")
        with pytest.raises(AreaRuleRefusal) as caught:
            declaration.judge_block(declaration.AreaRuleBlock.from_dict(data), registry.load_registry())
        assert "core_mm2" in str(caught.value)

    def test_n2_neurram_declares_its_differential_pair(self):
        org = _block("neurram").organization
        assert org["cells_per_weight"] == 2 and "TXT-wan2021edge-p06-15" in org["citation"]["cells_per_weight"]

    def test_n13_truenorth_declares_its_logic_basis(self):
        held = _block("truenorth").anchor["logic_held_per_core"]
        assert held["citation"] == ["T1-TN-01", "T1-TN-06"]

    def test_the_digest_names_the_block(self):
        a, b = declaration.block_digest(_block("odin")), declaration.block_digest(_block("truenorth"))
        assert a != b and len(a) == 16 and declaration.block_digest(None) == "none"
