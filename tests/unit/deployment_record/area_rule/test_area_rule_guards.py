"""[U-AREA] Guards: what the unit must not move, green on R5 and green after it.

T-11: the study chips' complete declared terms (ruling R1): R5's own path, unchanged (the design's golden values).
T-15: energy's aggregate judgment is unchanged (area's anchor path replaces its cell-bits bound for area only).
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.area_rule import _golden as G

STUDY = ("study_compass", "study_vr_like", "study_isaac_like")


def _census(cores, rows, cols, weight_bits, tiles):
    values = {"cores_physical": cores, "axons_physical": cores * rows, "neurons_physical": cores * cols,
              "cells_physical": cores * rows * cols, "weight_bits": weight_bits, "tiles": tiles}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


class TestStudyChipsT11:
    @pytest.mark.parametrize("name", STUDY)
    def test_the_complete_terms_price_as_on_r5(self, name):
        g = get_platform_physics(name).validity.geometry
        pricing = price_absolute(_census(576, g.axons, g.neurons, 4, 576 // 4), get_platform_physics(name),
                                 geometry=[CoreGeometry(g.axons, g.neurons, 576)])
        band = next(t for t in pricing.terms if t.name == "chip_area_mm2").band
        want = G.golden()["T-3/T-4 anchor_identities (lanes out)"][name]
        bad = G.all_close((f, getattr(band, f), want[f]) for f in ("nominal", "low", "high"))
        assert not bad, bad

    @pytest.mark.parametrize("name", ("isaac_like",) + STUDY)
    def test_the_design_rows_off_their_size(self, name):
        rows = G.moves()["design_facts"]["own_and_off_complete"][name]
        g = get_platform_physics(name).validity.geometry
        cpt = G.regime()["targets"][name]["cores_per_tile"]
        bad = []
        for row in rows:
            n, bits = row["cores"], row["weight_bits"]
            pricing = price_absolute(_census(n, g.axons, g.neurons, bits, math.ceil(n / cpt)),
                                     get_platform_physics(name), geometry=[CoreGeometry(g.axons, g.neurons, n)])
            band = next(t for t in pricing.terms if t.name == "chip_area_mm2").band
            bad += G.all_close([(f"{n} {f}", getattr(band, f), row["after"][f]) for f in ("nominal", "low", "high")])
        assert not bad, bad


def _per_event_physics(cell_bits):
    """A per-event energy aggregate measured on a 256 x 256 core of ``cell_bits``-bit cells (energy.py:114 judges it
    with validity.aggregate_scales, as on R5)."""
    return profile_from_dict({
        "format_version": 1, "name": "e", "display_name": "E", "description_file": "e.md",
        "validity": {"measurement_kind": "silicon", "geometry": {
            "axons": 256, "neurons": 256, "cell_bits": cell_bits, "core_count": None, "scaling": "cells"}},
        "constants": {"e_synaptic_event_total": {"nominal": 10.0, "unit": "pJ", "evidence_kind": "estimated",
                                                 "note": "test"}},
    })


class TestEnergyJudgmentT15:
    def _energy(self, physics, weight_bits, geometry):
        values = {"synaptic_events": 1000, "host_macs": 0, "cores_physical": 1, "weight_bits": weight_bits}
        pricing = price_absolute(Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()}),
                                 physics, geometry=geometry)
        terms = {t.name for t in pricing.terms}
        return terms, {r.name: r.reason for r in pricing.refusals}

    def test_wider_weights_than_the_cell_refuse_the_energy_by_name(self):
        terms, reasons = self._energy(_per_event_physics(4), 8, [CoreGeometry(256, 256, 1)])
        assert "energy_per_inference_mj" not in terms
        reason = reasons["energy_per_inference_mj"]
        assert reason.startswith("e_synaptic_event_total lies outside its validity")
        assert "256 x 256 core of 4-bit cells; this program's weights take 8 bits" in reason

    def test_off_its_core_the_energy_refuses_by_name(self):
        terms, reasons = self._energy(_per_event_physics(None), 4, [CoreGeometry(128, 128, 1)])
        assert reasons["energy_per_inference_mj"].startswith("e_synaptic_event_total lies outside its validity")

    def test_inside_its_validity_the_energy_prices(self):
        terms, _reasons = self._energy(_per_event_physics(8), 8, [CoreGeometry(256, 256, 1)])
        assert "energy_per_inference_mj" in terms
