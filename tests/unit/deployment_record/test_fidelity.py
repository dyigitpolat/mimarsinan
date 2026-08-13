"""Estimate vs signoff: what the search predicted against what the run measured."""

import json
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.fidelity import (
    FIDELITY_FILENAME,
    AxisComparison,
    FidelityReport,
    compare_axis,
    save_fidelity_report,
)


def _axis(predicted=10.0, measured=11.0, band=(8.0, 12.0), **over):
    kwargs = dict(
        key="e2e_latency_s", unit="s", direction="min",
        predicted=predicted, measured=measured, predicted_band=band,
    )
    kwargs.update(over)
    return compare_axis(**kwargs)


class TestOneAxis:
    def test_a_measurement_inside_the_band_is_in_band(self):
        assert _axis(measured=11.0).in_band is True

    def test_a_measurement_outside_the_band_is_not(self):
        assert _axis(measured=99.0).in_band is False

    def test_the_band_edges_count_as_inside(self):
        assert _axis(measured=8.0).in_band is True
        assert _axis(measured=12.0).in_band is True

    def test_relative_error_is_against_the_measurement(self):
        """The measurement is the reference: the prediction is what was wrong."""
        assert _axis(predicted=10.0, measured=20.0).relative_error == pytest.approx(-0.5)
        assert _axis(predicted=30.0, measured=20.0).relative_error == pytest.approx(0.5)

    def test_a_zero_measurement_has_no_relative_error(self):
        assert _axis(measured=0.0).relative_error is None

    def test_an_unpredicted_axis_records_the_measurement_alone(self):
        axis = _axis(predicted=None, band=None)
        assert axis.predicted is None
        assert axis.in_band is None
        assert axis.relative_error is None
        assert axis.measured == 11.0

    def test_an_unmeasured_axis_records_the_prediction_alone(self):
        axis = _axis(measured=None)
        assert axis.measured is None
        assert axis.in_band is None

    def test_without_a_band_there_is_no_in_band_verdict(self):
        """A point prediction is not a claim about a range; saying 'out of band'
        would invent a tolerance nobody declared."""
        axis = _axis(band=None)
        assert axis.predicted == 10.0
        assert axis.in_band is None

    def test_an_axis_is_json_safe(self):
        payload = json.loads(json.dumps(_axis().to_dict()))
        assert payload["key"] == "e2e_latency_s"
        assert payload["unit"] == "s"


class TestTheReport:
    def _report(self, axes=None):
        return FidelityReport(
            cell_key="t0_44", run_dir="/tmp/run",
            axes=tuple(axes or [
                _axis(key="chip_area_mm2", unit="mm^2", measured=11.0),
                _axis(key="e2e_latency_s", measured=99.0),
                _axis(key="pass_count", unit="passes", predicted=6.0, measured=6.0,
                      band=None),
            ]),
        )

    def test_it_counts_what_landed_in_band(self):
        report = self._report()
        # Every axis answered on both sides, so all three are COMPARED; only the
        # banded ones can land in band, and only one of those did.
        assert report.compared_count == 3
        assert report.in_band_count == 1

    def test_structural_axes_are_reported_as_exact_agreement(self):
        report = self._report()
        exact = [a for a in report.axes if a.key == "pass_count"][0]
        assert exact.relative_error == pytest.approx(0.0)

    def test_a_report_round_trips(self):
        report = self._report()
        restored = FidelityReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert restored == report

    def test_an_unknown_field_is_rejected(self):
        payload = self._report().to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="unknown fields"):
            FidelityReport.from_dict(payload)

    def test_it_writes_beside_the_record(self, tmp_path):
        path = save_fidelity_report(self._report(), str(tmp_path))
        assert path.endswith(FIDELITY_FILENAME)
        with open(path, encoding="utf-8") as handle:
            assert FidelityReport.from_dict(json.load(handle)).cell_key == "t0_44"

    def test_an_empty_report_is_legal_and_says_nothing(self):
        report = FidelityReport(cell_key="k", run_dir="/tmp", axes=())
        assert report.compared_count == 0
        assert report.in_band_count == 0
