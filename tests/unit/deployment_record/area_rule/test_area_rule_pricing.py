"""[U-AREA] The rule in the toolchain: the descriptions' blocks priced by price_absolute (W3, W4), the outside part
through the loader (W5), the wizard (W5), the fidelity gate across the block (W5), the lanes (T-25) and the converter
resolution (N-6). These need the profiles' blocks and the pricer's rule path; on R5 they are red.

Tests: T-3, T-4 (identities through the pricer), T-12 (the exploration priced, N-1), T-13 (the band through the
pricer), T-14 (flags), T-20 (profile, override, store), T-21 (the wizard), T-24 (the fidelity gate), T-25 (lanes),
N-6 (the converter stated twice; kinds at the run's readout bits).

A description read only with an external component declared (CL2: the SRAM-CIM macro's backing store) is priced with
that component declared as test values (``_physics.priced_physics``), as every price of it is; the gate itself is
unchanged (test_area_rule_pricer.TestExternalGateUnchanged).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.registry import profile_path
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G
from unit.deployment_record.area_rule._physics import priced_physics

ANCHORS = ("truenorth", "odin", "loihi", "hermes_64", "neurram", "sram_cim_macro")
FLAG_KINDS = ("anchor_identity", "beyond_published", "below_published", "off_geometry", "off_word_width",
              "weight_bits_not_stated")


def _census(cores, rows, cols, weight_bits=None, tiles=None, **extra):
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": weight_bits,
              "tiles": tiles if tiles is not None else cores, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items() if v is not None})


def _area(physics, cores, rows, cols, weight_bits=None, tiles=None, **extra):
    pricing = price_absolute(_census(cores, rows, cols, weight_bits, tiles, **extra), physics,
                             geometry=[CoreGeometry(rows, cols, cores)])
    terms = {t.name: t for t in pricing.terms}
    refusals = {r.name: r.reason for r in pricing.refusals}
    band = terms["chip_area_mm2"].band if "chip_area_mm2" in terms else None
    return band, refusals.get("chip_area_mm2"), pricing


def _geometry(name):
    g = get_platform_physics(name).validity.geometry
    return g.axons, g.neurons, g.cell_bits


class TestIdentitiesT3T4:
    @pytest.mark.parametrize("name", ANCHORS)
    def test_each_anchor_prices_its_own_point(self, name):
        law = G.moves()["laws"][name]
        rows, cols, bits = _geometry(name)
        band, refusal, _p = _area(priced_physics(name), int(law["n_pub"]), rows, cols, bits)
        assert refusal is None, refusal
        want = G.golden()["T-3/T-4 anchor_identities (lanes out)"][name]
        bad = G.all_close((f, getattr(band, f), want[f]) for f in ("nominal", "low", "high"))
        assert not bad, bad

    def test_truenorth_prices_its_published_die(self):
        band, _r, _p = _area(get_platform_physics("truenorth"), 4096, 256, 256, 1)
        assert G.close(band.nominal, 430.0) and band.low == band.high == band.nominal


class TestExplorationT12:
    """N-1: every explored configuration priced and compared, values not only counts."""

    def test_every_row_prices_and_equals_its_expected_value(self):
        data = G.regime()
        bad, refused, points = [], 0, 0
        for row in data["rows"]:
            target, cores, bits, n_points = row[0], row[1], row[2], row[3]
            t = data["targets"][target]
            tiles = math.ceil(cores / t["cores_per_tile"])
            band, refusal, _p = _area(priced_physics(target), cores, t["rows"], t["cols"], bits, tiles)
            points += n_points
            if refusal is not None:
                refused += n_points
                continue
            want = row[7:10]
            pairs = [("nominal", band.nominal, want[0])] + ([("low", band.low, want[1]), ("high", band.high, want[2])]
                                                           if want[1] is not None else [])
            bad += G.all_close([(f"{target} {cores} {bits} {f}", g, w) for f, g, w in pairs], rel=1e-9)
        assert points == data["totals"]["points"] and refused == 0
        assert not bad, bad[:5]

    def test_the_counts_of_the_design(self):
        want = G.golden()["T-12 exploration (lanes out)"]
        data = G.regime()
        beyond, identity, offword = {}, {}, {}
        for row in data["rows"]:
            target, cores, bits, n_points = row[0], row[1], row[2], row[3]
            t = data["targets"][target]
            _b, _r, pricing = _area(priced_physics(target), cores, t["rows"], t["cols"], bits,
                                    math.ceil(cores / t["cores_per_tile"]))
            kinds = {f.kind for f in pricing.flags if f.name == "chip_area_mm2"}
            for kind, tally in (("beyond_published", beyond), ("anchor_identity", identity),
                                ("off_word_width", offword)):
                if kind in kinds:
                    tally[target] = tally.get(target, 0) + n_points
        assert beyond == want["beyond_published"] and identity == want["identity_points"]
        assert offword == want["off_word_width_points"]


class TestBandThroughThePricerT13:
    def test_truenorth_at_512_cores(self):
        want = G.golden()["T-13 band"]["truenorth_512 (lanes out)"]
        band, _r, _p = _area(get_platform_physics("truenorth"), 512, 256, 256, 1)
        assert not G.all_close((f, getattr(band, f), want[f]) for f in ("nominal", "low", "high"))

    def test_the_generic_at_1000_cores(self):
        want = G.golden()["T-13 band"]["generic_1000_cores_8bit (lanes out)"]
        band, _r, _p = _area(get_platform_physics("generic_estimated_22nm"), 1000, 256, 256, 8, 250)
        assert not G.all_close((f, getattr(band, f), want[f]) for f in ("nominal", "low", "high"))


class TestFlagsT14:
    @pytest.mark.parametrize("cores,rows,cols,bits,kind", [
        (32768, 256, 256, 1, "beyond_published"), (512, 256, 256, 1, "below_published"),
        (4096, 128, 128, 1, "off_geometry"), (4096, 256, 256, 1, "anchor_identity")])
    def test_truenorth_flags(self, cores, rows, cols, bits, kind):
        _b, _r, pricing = _area(get_platform_physics("truenorth"), cores, rows, cols, bits)
        assert kind in {f.kind for f in pricing.flags if f.name == "chip_area_mm2"}

    def test_off_word_width(self):
        _b, _r, pricing = _area(get_platform_physics("odin"), 1, 256, 256, 4)
        assert "off_word_width" in {f.kind for f in pricing.flags}

    def test_weight_bits_not_stated(self):
        _b, _r, pricing = _area(get_platform_physics("neurram"), 48, 256, 256, None)
        assert "weight_bits_not_stated" in {f.kind for f in pricing.flags}

    def test_the_flag_is_in_the_term_basis(self):
        band, _r, pricing = _area(get_platform_physics("truenorth"), 512, 256, 256, 1)
        source = next(t for t in pricing.terms if t.name == "chip_area_mm2").source
        assert "below the published chip" in source


class TestStatedTwiceT20:
    def test_a_profile_with_a_block_and_an_area_global_fixed(self):
        data = json.loads(profile_path("truenorth").read_text())
        data["constants"]["area_global_fixed"] = {"nominal": 46.6, "unit": "mm^2", "evidence_kind": "derived",
                                                  "derivation": "430 - 4096 x 0.0936"}
        with pytest.raises(ValueError) as caught:
            profile_from_dict(data)
        assert "stated twice" in str(caught.value)

    def test_an_override_of_area_global_fixed_beside_gamma(self):
        with pytest.raises(ValueError) as caught:
            apply_overrides(get_platform_physics("truenorth"),
                            {"area_global_fixed": {"nominal": 46.6, "unit": "mm^2", "derivation": "test",
                                                   "note": "test"}})
        assert "stated twice" in str(caught.value)

    def test_the_block_survives_an_override(self):
        physics = apply_overrides(get_platform_physics("truenorth"),
                                  {"t_cycle": {"nominal": 1e-3, "unit": "s", "derivation": "test", "note": "test"}})
        assert physics.area_rule == get_platform_physics("truenorth").area_rule


class TestWizardT21:
    def test_the_wizard_names_the_blocks_outside_source(self):
        from mimarsinan.gui.wizard import physics_panel
        assert physics_panel.axis_also_constant(get_platform_physics("truenorth"), "chip_area_mm2").startswith(
            "area_rule.anchor.die")
        assert physics_panel.axis_also_constant(get_platform_physics("study_compass"),
                                                "chip_area_mm2") == "area_global_fixed"


class TestFidelityGateT24:
    def test_across_the_block_the_axis_is_named_not_comparable(self):
        from mimarsinan.deployment_record import fidelity_build
        block = get_platform_physics("truenorth")
        none = replace(block, area_rule=None)
        reason = fidelity_build.chip_area_comparability(block, none)
        assert reason.startswith("not comparable: the area rules differ (block ") and reason.endswith("against none)")
        assert fidelity_build.chip_area_comparability(block, block) is None

    def test_the_gate_neither_passes_nor_fails_an_axis_not_comparable(self):
        from mimarsinan.deployment_record import fidelity_build
        from mimarsinan.deployment_record.fidelity import FidelityReport, compare_axis
        axis = compare_axis(key="chip_area_mm2", unit="mm^2", direction="min", predicted=147.5, measured=430.0,
                            predicted_band=None, basis="not comparable: the area rules differ (block x against none)")
        report = FidelityReport(cell_key="c", run_dir="r", axes=(axis,), terms=(), measured_effective_activity=None,
                                declared_activity_factor=None)
        fidelity_build.enforce_fidelity_gates(report)


class TestLanesT25:
    def test_cores_only_and_with_lanes(self):
        want = G.moves()["design_facts"]["m11_per_4bit_core_mm2"]
        cores = 21730
        band, _r, _p = _area(get_platform_physics("generic_estimated_22nm"), cores, 256, 256, 4,
                             math.ceil(cores / 4))
        assert G.close(band.nominal / cores, want["unit_cores_only"])
        lanes = resolve_platform_physics("generic_estimated_22nm", {}, lane_block="ng45_os_lane")
        # the ladder row's lanes as R5 priced them (its lanes adder per core over one lane's area), 4 cores per tile
        lanes_per_core = round(want["r5_lanes_per_core"] / (lanes.band("area_per_lane").nominal * 1e6))
        band_l, _r, _p = _area(lanes, cores, 256, 256, 4, math.ceil(cores / 4), lanes_per_tile=4 * lanes_per_core)
        assert G.close(band_l.nominal / cores, want["unit_with_lanes"], 1e-6)


class TestConverterResolutionN6:
    def test_a_converter_table_beside_a_blocks_kind_is_stated_twice(self):
        data = json.loads(profile_path("neurram").read_text())
        data["constants"]["area_per_adc"] = {"nominal": 1270.0, "unit": "um^2", "evidence_kind": "measured",
                                             "citation": "T3-NR-03", "note": "test"}
        with pytest.raises(ValueError) as caught:
            profile_from_dict(data)
        assert "the converter area is stated twice" in str(caught.value)

    def test_a_kind_resolves_at_the_runs_readout_bits(self):
        physics = get_platform_physics("neurram")
        census = _census(48, 256, 256, 4)
        priced = price_absolute(census, physics, geometry=[CoreGeometry(256, 256, 48)], readout_bits=6)
        assert "chip_area_mm2" in {t.name for t in priced.terms}
