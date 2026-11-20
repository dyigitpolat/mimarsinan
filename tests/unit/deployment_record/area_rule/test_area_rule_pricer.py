"""[U-AREA] The pricer's rule path beyond the design's named cases (build part 2: W3 to W5), each red before the code.

W3: a census of several core types prices the sum of their grids under one outside part; a census that proves only
its totals prices one pooled core whose terms are linear (equal to its declared core types where they are), and a
per-core ceiling on it refuses by name; a census without cells refuses by name; a declared converter kind resolves at
the run's readout bits (review N-6), a kind whose area needs bits refusing by name where the run declares none; a
complete-terms block names its published size and the size flags it (ruling R3); the flags are computed at read and
never sealed; the external gate (CL2) is unchanged. W4: every shipped profile carries its block as the build declares
it, the dropped rows are gone, and each description document names its block. W5: the block serializes only when
present; the converter area stated twice and the outside part stated twice refuse at every entry (override, store, a
run's converter table); the wizard reads the block as the area axis's head; the fidelity report names the chip-area
axis not comparable across the block and the gate neither passes nor fails it.

Expected values come from the rule modules at the rule level (one core type at a time), the design's golden file and
the build's data (_golden); no value is typed here.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.report import candidate_cost_report
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED, CoreGeometry
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule import anchor, declaration, registry, rule
from mimarsinan.deployment_record.platform_physics.registry import available_profiles, profile_path
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G
from unit.deployment_record.area_rule._physics import store_declared

AREA = "chip_area_mm2"


def _q(**values):
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items() if v is not None})


def _types_census(types, weight_bits):
    """A census of core types ``(count, rows, cols)``."""
    return _q(cores_physical=sum(n for n, _r, _c in types), axons_physical=sum(n * r for n, r, _c in types),
              neurons_physical=sum(n * c for n, _r, c in types), cells_physical=sum(n * r * c for n, r, c in types),
              weight_bits=weight_bits, tiles=sum(n for n, _r, _c in types))


def _priced(pricing):
    terms = {t.name: t for t in pricing.terms}
    refusals = {r.name: r.reason for r in pricing.refusals}
    return terms.get(AREA), refusals.get(AREA)


def _grid(law, types, weight_bits):
    """(nominal, low, high) of the core grid in mm^2 from the rule-level per-core band, one core type at a time."""
    out = [0.0, 0.0, 0.0]
    for n, r, c in types:
        census = anchor.CensusCore(rows=r, cols=c, weight_bits=weight_bits)
        ratio = anchor.anchor_ratio(law.organization, census)
        band = anchor.anchor_core_band(law.core_mm2, ratio.ratio, law.own_share.value, law.logic_held_per_core)
        out = [out[0] + n * band.nominal, out[1] + n * band.low, out[2] + n * band.high]
    return tuple(out)


class TestCoreTypesW3:
    def test_two_core_types_sum_their_grids_under_one_outside_part(self):
        physics = get_platform_physics("truenorth")
        law = anchor.anchor_law(registry.load_registry(), physics.area_rule)
        types = ((2048, 256, 256), (2048, 128, 128))
        grid = _grid(law, types, 1)
        held = law.n_pub * law.core_mm2 * (1.0 / law.gamma - 1.0)
        edges = (grid[0] / law.gamma, grid[0] + held, grid[1] / law.gamma, grid[2] / law.gamma, grid[1] + held,
                 grid[2] + held)
        term, refusal = _priced(price_absolute(_types_census(types, 1), physics,
                                               geometry=[CoreGeometry(r, c, n) for n, r, c in types]))
        assert refusal is None, refusal
        assert not G.all_close([("nominal", term.band.nominal, grid[0] / law.gamma),
                                ("low", term.band.low, min(edges)), ("high", term.band.high, max(edges))])

    def test_the_size_flags_read_the_whole_chip(self):
        physics = get_platform_physics("truenorth")
        types = ((2048, 256, 256), (2048, 128, 128))
        pricing = price_absolute(_types_census(types, 1), physics,
                                 geometry=[CoreGeometry(r, c, n) for n, r, c in types])
        kinds = [f.kind for f in pricing.flags if f.name == AREA]
        assert "off_geometry" in kinds
        assert not {"beyond_published", "below_published", "anchor_identity"} & set(kinds)

    @pytest.mark.parametrize("name", ("truenorth", "neurram", "loihi", "generic_estimated_22nm"))
    @pytest.mark.parametrize("judged", (None, UNJUDGED))
    def test_a_pooled_census_prices_as_its_declared_core_types_where_terms_are_linear(self, name, judged):
        physics = get_platform_physics(name)
        g = physics.validity.geometry
        types = ((25, g.axons, g.neurons), (23, g.axons, g.neurons // 2))  # a mean core of a fraction of a column
        census = _types_census(types, g.cell_bits)
        declared, _r = _priced(price_absolute(census, physics,
                                              geometry=[CoreGeometry(r, c, n) for n, r, c in types]))
        pooled, refusal = _priced(price_absolute(census, physics, geometry=judged))
        assert refusal is None, refusal
        assert G.close(pooled.band.nominal, declared.band.nominal)

    def test_a_pooled_census_is_flagged_and_the_declared_types_are_not(self):
        physics = get_platform_physics("truenorth")
        types = ((24, 256, 256), (24, 256, 128))
        census = _types_census(types, 1)
        pooled = price_absolute(census, physics, geometry=UNJUDGED)
        declared = price_absolute(census, physics, geometry=[CoreGeometry(r, c, n) for n, r, c in types])
        assert "pooled_census" in {f.kind for f in pooled.flags}
        assert "pooled_census" not in {f.kind for f in declared.flags}

    def test_a_per_core_ceiling_on_a_pooled_census_is_a_named_refusal(self):
        physics = get_platform_physics("neurram")
        org = dict(physics.area_rule.organization, columns_per_converter=4)
        physics = replace(physics, area_rule=replace(physics.area_rule, organization=org))
        _t, refusal = _priced(price_absolute(_types_census(((24, 256, 256), (24, 256, 128)), 4), physics,
                                             geometry=UNJUDGED))
        assert refusal == G.golden()["T-17 anchors off geometry"]["pooled"][1]["refused"]

    def test_a_census_that_states_no_geometry_is_read_at_the_descriptions_core(self):
        """Read as declared (UNJUDGED, or a census of cores alone): its cores at the description's own core, named."""
        physics = get_platform_physics("neurram")
        g = physics.validity.geometry
        for judged in (UNJUDGED, None):
            pricing = price_absolute(_q(cores_physical=24, weight_bits=4), physics, geometry=judged)
            term, refusal = _priced(pricing)
            assert refusal is None and "read_as_declared" in {f.kind for f in pricing.flags}
            at_core, _r = _priced(price_absolute(_types_census(((24, g.axons, g.neurons),), 4), physics,
                                                 geometry=[CoreGeometry(g.axons, g.neurons, 24)]))
            assert G.close(term.band.nominal, at_core.band.nominal)

    def test_a_census_that_proves_its_cores_without_their_cells_is_a_named_refusal(self):
        census = _q(cores_physical=48, axons_physical=48 * 256, neurons_physical=48 * 256, weight_bits=4)
        _t, refusal = _priced(price_absolute(census, get_platform_physics("neurram")))
        assert refusal is not None and "cells_physical" in refusal


class TestTypeRuleBandW3:
    def test_a_reading_never_lifts_the_low_edge_above_the_nominal(self):
        """A census whose neuron state outweighs its cells (one row of 1-bit weights) puts the T1 reading above the
        T2 digital nominal: the low edge is the lower of the type's edge and the reading's."""
        physics = get_platform_physics("generic_estimated_22nm")
        pricing = price_absolute(_types_census(((10, 1, 256),), 1), physics, geometry=[CoreGeometry(1, 256, 10)])
        term, refusal = _priced(pricing)
        assert refusal is None and term.band.low <= term.band.nominal <= term.band.high
        reading = next(f.value for f in pricing.flags if f.kind == "t1_reading")
        reg = registry.load_registry()
        assert reading > term.band.nominal
        assert G.close(term.band.low, min(term.band.nominal * reg.band_of_type("T2_digital")[0],
                                          reading * reg.band_of_type("T1")[0]))


class TestDeclaredPartsW3:
    def test_a_type_rule_with_declared_parts_takes_gamma_1_and_r5s_outside_part(self):
        """The outside field decides: declared parts take gamma 1 and R5's part (the chip components' sum) beside the
        grid; the same type rule with outside 'type' takes the type's gamma and no fixed part."""
        physics = get_platform_physics("study_compass")
        g = physics.validity.geometry
        node = {"node_nm": physics.validity.technology_node_nm}
        blocks = {outside: declaration.AreaRuleBlock.from_dict({"type": "T2_digital", "terms": "type_rule",
                                                                "outside": outside, "node": node})
                  for outside in ("declared_parts", "type")}
        census = _types_census(((576, g.axons, g.neurons),), 4)
        geometry = [CoreGeometry(g.axons, g.neurons, 576)]
        declared, _r = _priced(price_absolute(census, replace(physics, area_rule=blocks["declared_parts"]),
                                              geometry=geometry))
        # the fourth delta round (MA-C2): a physics the loader accepts for the type's gamma, its outside part stated
        # once (no area_global_fixed, the chip components' sum, beside the type's gamma)
        bare = replace(physics, chip_components=(), constants={k: v for k, v in physics.constants.items()
                                                              if k != "area_global_fixed"})
        typed, _r = _priced(price_absolute(census, replace(bare, area_rule=blocks["type"]), geometry=geometry))
        gamma = rule.type_gamma(registry.load_registry(), "T2_digital")
        fixed = physics.band("area_global_fixed").nominal * 1e6
        assert G.close(declared.band.nominal, typed.band.nominal * gamma + fixed)


def _t3_type_rule(physics):
    """A T3 type rule on ``physics`` (the fourth delta round, MA-C2: its outside part is the description's declared
    part, a physics the loader accepts; outside 'type' beside its area_global_fixed was the outside part stated
    twice)."""
    block = declaration.AreaRuleBlock.from_dict({"type": "T3", "terms": "type_rule", "outside": "declared_parts",
                                                 "node": {"node_nm": physics.validity.technology_node_nm}})
    return replace(physics, area_rule=block)


class TestReadoutBitsN6:
    def _price(self, physics, bits):
        census = _types_census(((48, 256, 256),), 4)
        return _priced(price_absolute(census, physics, geometry=[CoreGeometry(256, 256, 48)], readout_bits=bits))

    def test_a_type_rules_default_kind_needs_the_runs_bits(self):
        physics = _t3_type_rule(get_platform_physics("isaac_like"))
        _t, refusal = self._price(physics, None)
        assert refusal == ("converter kind sar_gsps: its area depends on the converter's bits (8 to 8) and the run "
                           "declares none (readout_law.adc_bits)")
        term, refusal = self._price(physics, 8)
        assert refusal is None and term.band.nominal > 0.0
        _t, refusal = self._price(physics, 6)
        assert refusal == "converter kind sar_gsps publishes 8-bit areas only; 6 requested"

    def test_an_anchors_declared_kind_resolves_at_the_runs_bits(self):
        physics = get_platform_physics("neurram")
        converter = dict(physics.area_rule.converter, kind="sar_gsps")
        gsps = replace(physics, area_rule=replace(physics.area_rule, converter=converter))
        _t, refusal = self._price(gsps, None)
        assert refusal is not None and refusal.startswith("converter kind sar_gsps: its area depends")
        term, refusal = self._price(gsps, 8)
        assert refusal is None
        own, _r = self._price(physics, None)
        assert G.close(term.band.nominal, own.band.nominal)  # the anchor scales counts: the unit is its own

    def test_a_kind_without_bits_prices_at_any_bits_with_its_flag(self):
        physics = get_platform_physics("hermes_64")
        census = _types_census(((64, 256, 256),), 4)
        for bits in (None, 6, 8):
            pricing = price_absolute(census, physics, geometry=[CoreGeometry(256, 256, 64)], readout_bits=bits)
            term, refusal = _priced(pricing)
            assert refusal is None and G.close(term.band.nominal, G.golden()[
                "T-3/T-4 anchor_identities (lanes out)"]["hermes_64"]["nominal"])
            assert "converter_bits_not_stated" in {f.kind for f in pricing.flags}


class TestExternalGateUnchanged:
    """Guard: CL2's external gate stays the gate of every pricer; the area rule prices the macro once its store is
    declared, as every price of the macro."""

    def test_the_macro_without_its_store_is_refused_by_the_gate(self):
        census = _types_census(((1, 128, 1024),), 8)
        _t, refusal = _priced(price_absolute(census, get_platform_physics("sram_cim_macro"),
                                             geometry=[CoreGeometry(128, 1024, 1)]))
        assert refusal.startswith("absolute_pricing: sram_cim_macro is read only with its backing store declared")

    def test_with_its_store_the_macro_prices_its_identity(self):
        census = _types_census(((1, 128, 1024),), 8)
        term, refusal = _priced(price_absolute(census, store_declared(get_platform_physics("sram_cim_macro")),
                                               geometry=[CoreGeometry(128, 1024, 1)]))
        want = G.golden()["T-3/T-4 anchor_identities (lanes out)"]["sram_cim_macro"]
        assert refusal is None and G.close(term.band.nominal, want["nominal"])


class TestCompleteTermsSizeW3:
    def _flags(self, name, cores):
        g = get_platform_physics(name).validity.geometry
        pricing = price_absolute(_types_census(((cores, g.axons, g.neurons),), g.cell_bits),
                                 get_platform_physics(name), geometry=[CoreGeometry(g.axons, g.neurons, cores)])
        return {f.kind for f in pricing.flags if f.name == AREA}

    def test_the_published_size_is_its_cited_expression(self):
        block = get_platform_physics("isaac_like").area_rule
        assert registry.load_registry().evaluate(block.published_size["n_pub"]) == G.regime()["targets"][
            "isaac_like"]["published_cores"]

    def test_beyond_and_below_the_published_size_are_flagged(self):
        n_pub = int(G.regime()["targets"]["isaac_like"]["published_cores"])
        assert {"complete_terms", "beyond_published"} <= self._flags("isaac_like", 2 * n_pub)
        assert {"complete_terms", "below_published"} <= self._flags("isaac_like", n_pub // 2)
        assert not {"beyond_published", "below_published"} & self._flags("isaac_like", n_pub)

    def test_a_chip_without_a_published_size_takes_no_size_flag(self):
        assert self._flags("study_compass", 10_000) == {"complete_terms"}


class TestFlagsAtReadW3:
    def test_a_physics_without_a_block_prices_with_no_flag(self):
        physics = replace(get_platform_physics("study_compass"), area_rule=None)
        g = physics.validity.geometry
        pricing = price_absolute(_types_census(((576, g.axons, g.neurons),), 4), physics,
                                 geometry=[CoreGeometry(g.axons, g.neurons, 576)])
        assert pricing.flags == () and _priced(pricing)[0] is not None

    def test_the_flags_are_in_the_basis_and_never_in_the_report(self):
        census = _types_census(((512, 256, 256),), 1)
        report = candidate_cost_report(census, get_platform_physics("truenorth"),
                                       geometry=(CoreGeometry(256, 256, 512),))
        area = next(t for t in report.area if t.name == AREA)
        assert "below the published chip" in area.source
        data = report.to_dict()
        assert "flags" not in data and all("flags" not in term for group in ("area", "energy", "latency")
                                           for term in data[group])


class TestProfilesW4:
    @pytest.mark.parametrize("name", sorted(G.declarations()))
    def test_each_profile_carries_the_builds_block(self, name):
        data = json.loads(profile_path(name).read_text())
        assert data["area_rule"] == G.declarations()[name]

    def test_every_shipped_profile_has_a_block(self):
        assert sorted(available_profiles()) == sorted(G.declarations())
        assert all("area_rule" in json.loads(profile_path(name).read_text()) for name in available_profiles())

    @pytest.mark.parametrize("name,dropped", [
        ("truenorth", ("area_global_fixed",)), ("loihi", ("area_global_fixed",)), ("odin", ("area_global_fixed",)),
        ("hermes_64", ("area_global_fixed",)),
        ("generic_estimated_22nm", ("area_per_cell", "area_per_cell_per_weight_bit", "area_per_neuron_logic",
                                    "area_per_state_bit", "area_per_router", "area_per_tile_fixed",
                                    "area_global_fixed"))])
    def test_the_dropped_rows_are_gone(self, name, dropped):
        assert not set(dropped) & set(json.loads(profile_path(name).read_text())["constants"])

    def test_loihis_footprint_reads_its_vendor_core(self):
        core = registry.load_registry().evaluate("v('T1-LO-05')")
        assert G.close(get_platform_physics("loihi").band("area_per_core_total").nominal, core * 1e-6)

    @pytest.mark.parametrize("name", sorted(G.declarations()))
    def test_each_description_document_names_its_block(self, name):
        text = profile_path(name).with_suffix(".md").read_text()
        assert "area_rule" in text and get_platform_physics(name).area_rule.outside in text


class TestSerializationW5:
    @pytest.mark.parametrize("name", sorted(G.declarations()))
    def test_the_block_round_trips(self, name):
        physics = get_platform_physics(name)
        back = type(physics).from_dict(json.loads(json.dumps(physics.to_dict())))
        assert back.area_rule == physics.area_rule and physics.to_dict()["area_rule"] == G.declarations()[name]

    def test_a_physics_without_a_block_serializes_without_the_key(self):
        assert "area_rule" not in replace(get_platform_physics("truenorth"), area_rule=None).to_dict()


class TestLoaderEntriesW5:
    def test_an_override_of_the_converter_area_beside_a_kind(self):
        with pytest.raises(ValueError) as caught:
            apply_overrides(get_platform_physics("neurram"),
                            {"area_per_adc": {"nominal": 1270.0, "unit": "um^2", "derivation": "test", "note": "test"}})
        assert "the converter area is stated twice" in str(caught.value)

    def test_a_runs_converter_table_beside_a_kind(self):
        from mimarsinan.deployment_record.platform_physics.converter import with_converter
        table = {"rows": [{"bits": 8, "area_per_adc": {"nominal": 1270.0, "unit": "um^2", "evidence_kind": "measured",
                                                       "citation": "test", "note": "test"}}]}
        with pytest.raises(ValueError) as caught:
            with_converter(get_platform_physics("neurram"), table)
        assert "the converter area is stated twice" in str(caught.value)

    def test_a_store_restatement_beside_gamma(self):
        physics = get_platform_physics("study_compass")
        gamma = declaration.AreaRuleBlock.from_dict({"type": "T2_digital", "terms": "type_rule", "outside": "type",
                                                     "node": {"node_nm": physics.validity.technology_node_nm}})
        physics = replace(physics, area_rule=gamma, constants={k: v for k, v in physics.constants.items()
                                                               if k != "area_global_fixed"})
        with pytest.raises(ValueError) as caught:
            apply_overrides(physics, {"backing_store": {"host": {"note": "a host-owned store (test)"}}})
        assert "stated twice" in str(caught.value) and "store" in str(caught.value)


class TestWizardHeadW5:
    def test_a_block_is_the_area_axis_head(self):
        from mimarsinan.gui.wizard import physics_panel
        assert physics_panel._first_missing(get_platform_physics("neurram"), AREA) != "area_per_core_total"

    def test_without_a_block_or_a_source_the_head_is_named(self):
        from mimarsinan.gui.wizard import physics_panel
        bare = replace(get_platform_physics("neurram"), area_rule=None)
        assert physics_panel._first_missing(bare, AREA) == "area_per_core_total"


class TestFidelityReportW5:
    def test_the_area_axis_across_the_block_is_named_not_comparable(self):
        from mimarsinan.deployment_record.fidelity_build import enforce_fidelity_gates, fidelity_report_for_record
        from unit.deployment_record.test_fidelity_from_record import _candidate, _record
        record = _record()
        platform = dict(record.identity.platform)
        sealed = dict(platform["platform_physics_resolved"])
        law = anchor.anchor_law(registry.load_registry(), get_platform_physics("truenorth").area_rule)
        sealed.pop("area_rule")
        sealed["constants"] = dict(sealed["constants"], area_global_fixed={
            "low": law.n_pub * law.core_mm2 * (1 / law.gamma - 1), "nominal": law.n_pub * law.core_mm2 * (
                1 / law.gamma - 1), "high": law.n_pub * law.core_mm2 * (1 / law.gamma - 1), "unit": "mm^2",
            "evidence_kind": "derived", "citation": "a pre-unit record's die remainder (test)", "derivation": "test",
            "note": "", "overridden": False, "key": "area_global_fixed"})
        platform["platform_physics_resolved"] = sealed
        record = replace(record, identity=replace(record.identity, platform=platform))
        report = fidelity_report_for_record(record, _candidate())
        area = next(a for a in report.axes if a.key == AREA)
        assert area.basis.startswith("not comparable: the area rules differ (block ")
        assert area.in_band is None and area.relative_error is None
        enforce_fidelity_gates(report)
