"""[U-AREA] Anchors: identities at their own point (T-3, T-4, rule level), the ratio needs no cell (T-5, N-3), the
cases off geometry and word (T-17 with N-2, N-12, N-13 and minor G), the continuous band (T-18) and the exploration
priced row by row (T-12 at the rule level, N-1). The blocks are the build's declarations (W4 places them in the
profiles); the expected values are the design's golden file and the build's moves (_golden).

Mutants named by the design and caught here: rho fixed at 1 (T-17, T-12); gamma dropped on a die (T-3); N-2: the
weight-row reading of a differential-pair census (T-17); minor G: the type share held at the published core (T-17).
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.platform_physics.area_rule import anchor, declaration, registry
from mimarsinan.deployment_record.platform_physics.area_rule.evidence import AreaRuleRefusal

from unit.deployment_record.area_rule import _golden as G

ANCHORS = ("truenorth", "odin", "loihi", "hermes_64", "neurram", "sram_cim_macro")


def _law(name, reg=None):
    block = declaration.AreaRuleBlock.from_dict(G.declarations()[name])
    return anchor.anchor_law(reg or registry.load_registry(), block)


def _census(rows, cols, weight_bits, pooled=False):
    return anchor.CensusCore(rows=rows, cols=cols, weight_bits=weight_bits, pooled=pooled)


def _band(price):
    return price.nominal, price.low, price.high


class TestAnchorLawsAgainstTheDesign:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_core_gamma_size_and_own_share(self, name):
        law, want = _law(name), G.moves()["laws"][name]
        assert G.close(law.core_mm2 * 1e6, want["core_um2"])
        assert G.close(law.gamma, want["gamma"]) and law.n_pub == want["n_pub"]
        assert (law.scope, law.outside) == (want["scope"], want["outside"])
        assert G.close(law.own_share.value, want["s_anchor"])


class TestIdentitiesT3T4RuleLevel:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_the_anchor_at_its_own_point(self, name):
        law = _law(name)
        got = anchor.price_anchor(law, law.n_pub)
        want = G.golden()["T-3/T-4 anchor_identities (lanes out)"][name]
        bad = G.all_close((f, getattr(got, f), want[f]) for f in ("nominal", "low", "high"))
        assert not bad, bad
        assert [f.kind for f in got.flags] == ["anchor_identity"]

    def test_the_die_anchors_are_identities_to_their_published_dies(self):
        for name in ("truenorth", "loihi", "hermes_64", "neurram"):
            law = _law(name)
            got = anchor.price_anchor(law, law.n_pub)
            assert got.low == got.nominal == got.high
            assert G.close(got.nominal, law.die_mm2, 1e-12), name

    def test_mutant_gamma_dropped_on_a_die_is_caught(self, monkeypatch):
        monkeypatch.setattr(anchor, "own_die_gamma", lambda n_pub, core_mm2, die_mm2: 1.0)
        law = anchor.anchor_law(registry.Registry.load(), declaration.AreaRuleBlock.from_dict(
            G.declarations()["truenorth"]))
        assert not G.close(anchor.price_anchor(law, law.n_pub).nominal, law.die_mm2)


class TestNoCellNeededT5:
    def _loihi2_like(self):
        """A die anchor at Intel 4 (no published Intel 4 cell): Loihi 2's published core and die (T1-L2-01, -02)
        with Loihi's organization, its 192 KB memory per core (intel_loihi2_brief p.7)."""
        data = dict(G.declarations()["loihi"])
        data["node"] = {"node_nm": 4}
        data["anchor"] = dict(data["anchor"], core_mm2="v('T1-L2-02')", die_mm2="v('T1-L2-01')",
                              n_pub="s('T1-L2-01', 'scope', '128 neuron cores', 128)")
        return anchor.anchor_law(registry.load_registry(), declaration.AreaRuleBlock.from_dict(data))

    @pytest.mark.parametrize("cores", (1, 64, 128, 1024))
    def test_it_prices_at_any_core_count(self, cores):
        law = self._loihi2_like()
        got = anchor.price_anchor(law, cores)
        assert G.close(got.nominal, cores * law.core_mm2 / law.gamma)

    def test_its_own_point_is_its_published_die(self):
        law = self._loihi2_like()
        assert G.close(anchor.price_anchor(law, law.n_pub).nominal, law.die_mm2)

    def test_n3_off_geometry_the_held_logic_edge_is_a_named_refusal(self):
        """N-3: the edge needs s_anchor, the anchor's memory share, which needs sigma at Intel 4 (rule B refuses);
        the nominal prices (rho needs no cell) and the band edge is refused by name."""
        law = self._loihi2_like()
        got = anchor.price_anchor(law, 128, _census(64, 512, 8))
        assert got.nominal > 0.0 and math.isfinite(got.nominal)
        refused = [f for f in got.flags if f.kind == "band_edge_refused"]
        assert refused and "sigma(4 nm)" in refused[0].reason


class TestOffGeometryAndWordT17:
    def test_the_design_cases_after_the_build(self):
        bad = []
        for case in G.moves()["t17"]:
            law = _law(case["description"])
            got = anchor.price_anchor(law, case["cores"], _census(case["rows"], case["cols"], case["weight_bits"]))
            want = case["build"]
            bad += G.all_close([(f"{case['description']} {case['cores']} {case['rows']}x{case['cols']} "
                                 f"{case['weight_bits']}b {f}", getattr(got, f), want[f])
                                for f in ("nominal", "low", "high")])
            assert got.cells_per_weight == want["cells_per_weight"]
            assert G.close(got.ratio, want["ratio"])
        assert not bad, bad

    def test_mutant_ratio_fixed_at_1_is_caught(self, monkeypatch):
        real = anchor.anchor_ratio
        monkeypatch.setattr(anchor, "anchor_ratio", lambda org, census: real(org, census)._replace(ratio=1.0))
        caught = 0
        for case in G.moves()["t17"]:
            law = _law(case["description"])
            got = anchor.price_anchor(law, case["cores"], _census(case["rows"], case["cols"], case["weight_bits"]))
            caught += not G.close(got.nominal, case["build"]["nominal"])
        assert caught == sum(1 for c in G.moves()["t17"] if not G.close(c["build"]["ratio"], 1.0))

    def test_n13_logic_held_per_core_sets_the_nominal(self):
        """N-13: TrueNorth's figures fix its logic per core: the held-logic form is the nominal."""
        law = _law("truenorth")
        assert law.logic_held_per_core
        got = anchor.price_anchor(law, 4096, _census(128, 128, 1))
        assert got.nominal > got.low and G.close(got.high, max(got.high, got.nominal))
        assert any(f.kind == "off_geometry" and "held" in f.reason for f in got.flags)

    def test_n12_the_wider_word_reading_is_the_low_edge(self):
        """N-12: 4-bit weights on ODIN's 3-bit cells: 5 stored bits per cell, 0.1056 mm2 bare (the low edge)."""
        law = _law("odin")
        got = anchor.price_anchor(law, 1, _census(256, 256, 4))
        want = next(c for c in G.moves()["t17"] if c["description"] == "odin" and c["weight_bits"] == 4)["build"]
        assert G.close(got.low, want["low"])
        reading = [f for f in got.flags if f.kind == "wider_word_reading"]
        assert reading and G.close(reading[0].value, want["low"])

    def test_n2_a_differential_pair_census_in_device_rows(self):
        case = G.moves()["t17_n2"]
        law = _law("neurram")
        rows = anchor.device_rows(law.organization, case["weight_rows"])
        assert rows == case["weight_rows"] * case["cells_per_weight"]
        got = anchor.price_anchor(law, case["cores"], _census(rows, case["cols"], case["weight_bits"]))
        assert G.close(got.ratio, case["build"]["ratio"]) and G.close(got.nominal, case["build"]["nominal"])
        mutant = anchor.anchor_ratio(law.organization, _census(case["weight_rows"], case["cols"], case["weight_bits"]))
        assert G.close(mutant.ratio, case["mutant_weight_rows"]["ratio"]) and not G.close(mutant.ratio, 1.0)

    def test_g_an_anchor_without_a_published_die_holds_an_outside_part_of_0(self):
        for case in G.moves()["t17_g"]:
            law = _law(case["description"])
            assert law.outside == "type"
            got = anchor.price_anchor(law, case["cores"], _census(law.organization["rows"],
                                                                  law.organization["cols"], case["weight_bits"]))
            bad = G.all_close((f, getattr(got, f), case["build"][f]) for f in ("nominal", "low", "high"))
            assert not bad, bad
            assert G.close(got.fixed_reading, case["cores"] * law.core_mm2)
            assert not G.close(got.low, case["mutant_type_share_held"])

    def test_a_pooled_census_prices_linear_terms_and_refuses_a_per_core_ceiling(self):
        pooled = G.golden()["T-17 anchors off geometry"]["pooled"]
        law = _law("neurram")
        got = anchor.anchor_ratio(law.organization, _census(256, 192, 4, pooled=True))
        assert G.close(got.ratio, pooled[0]["result"]["ratio"])
        org = dict(law.organization, columns_per_converter=4)
        with pytest.raises(AreaRuleRefusal) as caught:
            anchor.anchor_ratio(org, _census(256, 192, 4, pooled=True))
        assert str(caught.value) == pooled[1]["refused"]


class TestContinuousBandT18:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_zero_width_at_ratio_1_and_small_near_it(self, name):
        law = _law(name)
        share = law.own_share.value
        a = law.core_mm2
        at_one = anchor.anchor_core_band(a, 1.0, share, law.logic_held_per_core)
        assert at_one.low == at_one.high == a
        for eps in (1e-6, -1e-6, 1e-3):
            band = anchor.anchor_core_band(a, 1.0 + eps, share, law.logic_held_per_core)
            assert 0.0 <= band.high - band.low <= 2 * abs(eps) * a
            assert band.low > 0.0


class TestExplorationT12RuleLevel:
    """N-1: every explored row of an anchor description recomputed (nominal, low, high), not only counted."""

    def test_every_anchor_row_equals_its_expected_price(self):
        reg = registry.load_registry()
        data = G.regime()
        laws = {name: _law(name, reg) for name in ANCHORS}
        bad = []
        for row in data["rows"]:
            target, cores, bits = row[0], row[1], row[2]
            if target not in laws:
                continue
            t = data["targets"][target]
            got = anchor.price_anchor(laws[target], cores, _census(t["rows"], t["cols"], bits))
            bad += G.all_close([(f"{target} {cores} {bits} {f}", g, w)
                                for f, g, w in zip(("nominal", "low", "high"), _band(got), row[7:10])])
            assert sorted(f.kind for f in got.flags if f.kind in anchor.CENSUS_FLAG_KINDS) == sorted(row[10]), row
        assert not bad, bad[:5]

    def test_the_golden_odin_row_after_the_build(self):
        law = _law("odin")
        got = anchor.price_anchor(law, 1, _census(256, 256, 4))
        old = G.golden()["T-12 odin rows (one core, 4-bit)"][0]
        assert G.close(got.nominal, old["nominal"]) and G.close(got.high, old["high"])
        moved = next(m for m in G.moves()["moves"] if m["path"] == "T-12 odin rows (one core, 4-bit)[0].low")
        assert G.close(got.low, moved["new"])

    def test_mutant_ratio_fixed_at_1_fails_the_row_values(self, monkeypatch):
        real = anchor.anchor_ratio
        monkeypatch.setattr(anchor, "anchor_ratio", lambda org, census: real(org, census)._replace(ratio=1.0))
        law = _law("odin")
        got = anchor.price_anchor(law, 1, _census(256, 256, 4))
        assert not G.close(got.nominal, G.golden()["T-12 odin rows (one core, 4-bit)"][0]["nominal"])
