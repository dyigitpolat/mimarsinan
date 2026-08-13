"""The cross-platform comparison: one workload census, priced by many targets."""

import json

import pytest

from mimarsinan.deployment_record.study import (
    ComparisonRow,
    CrossPlatformComparison,
    compare_platforms,
    render_comparison,
)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

_CENSUS = {
    "cores_physical": 100, "cells_physical": 100 * 256 * 256,
    "neurons_physical": 100 * 256, "axons_physical": 100 * 256,
    "macs": 1e7, "synaptic_events": 1e7, "latency_steps": 64, "timesteps": 32,
    "sync_count": 8, "segment_cores": 100, "tiles": 25, "host_macs": 0,
    "noc_total_hops": 1e5, "noc_intra_tile_packets": 1e5, "boundary_events": 1e5,
}


def _census(**over):
    values = {**_CENSUS, **over}
    return Quantities({
        key: QuantityValue(float(v), "static") for key, v in values.items()
    })


class TestTheComparison:
    def _compare(self, profiles=("truenorth", "loihi"), **over):
        return compare_platforms(_census(**over), profiles)

    def test_it_prices_one_census_with_every_named_target(self):
        comparison = self._compare()
        assert [row.profile for row in comparison.rows] == ["truenorth", "loihi"]

    def test_each_row_carries_the_absolute_axes(self):
        row = self._compare(profiles=("truenorth",)).rows[0]
        assert row.values["chip_area_mm2"] > 0
        assert row.values["energy_per_inference_mj"] > 0
        assert row.values["e2e_latency_s"] > 0

    def test_an_axis_a_target_cannot_back_is_absent_not_zero(self):
        """Loihi declares no t_cycle (it is asynchronous), so it has no latency —
        which must never be reported as a latency of zero, the best possible."""
        loihi = self._compare(profiles=("loihi",)).rows[0]
        assert "e2e_latency_s" not in loihi.values
        assert "e2e_latency_s" in loihi.unavailable

    def test_the_reason_travels_with_the_absence(self):
        loihi = self._compare(profiles=("loihi",)).rows[0]
        assert "t_cycle" in loihi.unavailable["e2e_latency_s"]


class TestEvidenceIsDisclosed:
    def test_each_row_states_its_measurement_kind(self):
        rows = {r.profile: r for r in compare_platforms(_census(), ("truenorth", "loihi"))
                .rows}
        assert rows["truenorth"].measurement_kind == "silicon"
        assert rows["loihi"].measurement_kind == "simulation"

    def test_each_row_states_how_much_of_it_is_estimated(self):
        """A comparison where one side is estimated must SAY so, per the standing
        discipline — a count, not a footnote."""
        rows = {r.profile: r for r in compare_platforms(
            _census(), ("truenorth", "generic_estimated_22nm")).rows}
        assert rows["truenorth"].evidence_counts.get("estimated", 0) == 0
        assert rows["generic_estimated_22nm"].evidence_counts["estimated"] > 0

    def test_a_comparison_mixing_bases_is_flagged(self):
        comparison = compare_platforms(_census(), ("truenorth", "loihi"))
        assert comparison.mixes_measurement_kinds is True

    def test_a_single_basis_comparison_is_not_flagged(self):
        comparison = compare_platforms(_census(), ("truenorth",))
        assert comparison.mixes_measurement_kinds is False


class TestTheArtifact:
    def test_it_round_trips(self):
        comparison = compare_platforms(_census(), ("truenorth", "loihi"))
        restored = CrossPlatformComparison.from_dict(
            json.loads(json.dumps(comparison.to_dict()))
        )
        assert restored == comparison

    def test_an_unknown_field_is_rejected(self):
        payload = compare_platforms(_census(), ("truenorth",)).to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="unknown fields"):
            CrossPlatformComparison.from_dict(payload)

    def test_the_rendering_names_every_target_and_axis(self):
        text = render_comparison(compare_platforms(_census(), ("truenorth", "loihi")))
        assert "truenorth" in text and "loihi" in text
        assert "chip_area_mm2" in text

    def test_the_rendering_discloses_the_evidence(self):
        text = render_comparison(compare_platforms(
            _census(), ("truenorth", "generic_estimated_22nm")))
        assert "estimated" in text
        assert "silicon" in text

    def test_an_unavailable_axis_renders_as_a_dash_with_its_reason(self):
        text = render_comparison(compare_platforms(_census(), ("loihi",)))
        assert "t_cycle" in text


class TestRefusals:
    def test_an_unknown_profile_fails_loud(self):
        with pytest.raises(KeyError, match="nosuchchip"):
            compare_platforms(_census(), ("nosuchchip",))

    def test_comparing_nothing_fails_loud(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_platforms(_census(), ())


class TestTheRowIsSelfDescribing:
    def test_a_row_knows_which_axes_it_could_not_back(self):
        row = ComparisonRow(
            profile="p", display_name="P", measurement_kind="silicon",
            values={"chip_area_mm2": 1.0}, bands={}, unavailable={"e2e_latency_s": "x"},
            evidence_counts={"published": 1},
        )
        assert row.available_axes == ("chip_area_mm2",)
