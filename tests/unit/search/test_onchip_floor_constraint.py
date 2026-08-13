"""The on-chip floor as a typed candidate CONSTRAINT, not a crash downstream."""

import pytest

from mimarsinan.search.constraints import (
    ONCHIP_FLOOR_CONSTRAINT,
    ConstraintReport,
    onchip_floor_violation,
)


def _report(fraction, floor=0.2):
    return onchip_floor_violation(fraction=fraction, floor=floor)


class TestTheViolationIsTypedAndScaled:
    def test_a_candidate_above_the_floor_does_not_violate(self):
        report = _report(0.9)
        assert report is None

    def test_a_candidate_exactly_at_the_floor_does_not_violate(self):
        assert _report(0.2) is None

    def test_a_candidate_below_the_floor_violates(self):
        report = _report(0.05)
        assert isinstance(report, ConstraintReport)
        assert report.constraint == ONCHIP_FLOOR_CONSTRAINT
        assert report.measured == pytest.approx(0.05)
        assert report.limit == pytest.approx(0.2)

    def test_the_violation_grows_with_the_shortfall(self):
        """A gradient the optimizer can descend: a near-miss must rank above a
        candidate that put almost nothing on chip."""
        near = _report(0.19).violation
        far = _report(0.01).violation
        assert 0 < near < far

    def test_the_violation_is_the_shortfall_itself(self):
        assert _report(0.05).violation == pytest.approx(0.15)

    def test_the_report_says_what_it_measured_and_against_what(self):
        detail = _report(0.05).detail
        assert "5" in detail and "20" in detail
        assert "on-chip" in detail.lower()


class TestTheCensus:
    def test_a_report_is_json_safe(self):
        import json

        payload = _report(0.05).to_dict()
        assert json.loads(json.dumps(payload))["constraint"] == ONCHIP_FLOOR_CONSTRAINT

    def test_a_zero_floor_can_never_be_violated(self):
        """An operator who disables the floor gets no constraint, not a zero-width
        one that every candidate technically satisfies."""
        assert onchip_floor_violation(fraction=0.0, floor=0.0) is None
