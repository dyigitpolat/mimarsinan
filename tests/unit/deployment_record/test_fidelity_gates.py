"""[R4] The proven-closed axes are gates now: disagree past tolerance, fail loud."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.fidelity import FidelityReport, compare_axis
from mimarsinan.deployment_record.fidelity_build import (
    ACTIVITY_MISS_WARN_RATIO,
    activity_warning,
    enforce_fidelity_gates,
)


def _report(axes=(), **fields):
    return FidelityReport(cell_key="c", run_dir="r", axes=tuple(axes), **fields)


def _axis(key, predicted, measured):
    return compare_axis(key=key, unit="u", direction="min",
                        predicted=predicted, measured=measured,
                        predicted_band=None)


class TestTheGates:
    def test_agreement_passes(self):
        enforce_fidelity_gates(_report([_axis("param_utilization_pct", 15.63, 15.63)]))

    def test_disagreement_fails_loud_and_names_the_axis(self):
        with pytest.raises(ValueError, match="param_utilization_pct"):
            enforce_fidelity_gates(
                _report([_axis("param_utilization_pct", 5.354, 10.709)]))

    def test_a_one_sided_axis_never_gates(self):
        """A gate judges agreement; absence has its own basis machinery."""
        enforce_fidelity_gates(_report([_axis("chip_area_mm2", 3.75, None)]))

    def test_ungated_axes_stay_report_only(self):
        enforce_fidelity_gates(
            _report([_axis("energy_per_inference_mj", 0.19, 379.0)]))


class TestTheActivityWarning:
    def test_a_2x_miss_warns_with_the_anchor(self):
        report = _report(measured_effective_activity=0.1118,
                         declared_activity_factor=0.05)
        warning = activity_warning(report)
        assert warning is not None and "0.1118" in warning

    def test_inside_the_band_stays_quiet(self):
        report = _report(measured_effective_activity=0.06,
                         declared_activity_factor=0.05)
        assert activity_warning(report) is None

    def test_no_census_no_warning(self):
        assert activity_warning(_report()) is None

    def test_the_threshold_is_symmetric(self):
        low = _report(measured_effective_activity=0.05 / (ACTIVITY_MISS_WARN_RATIO * 1.1),
                      declared_activity_factor=0.05)
        assert activity_warning(low) is not None
