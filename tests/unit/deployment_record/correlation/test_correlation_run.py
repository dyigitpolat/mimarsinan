"""Running a reference case: price its census, compare against the published value."""

import pytest

from mimarsinan.deployment_record.correlation.case import (
    PublishedValue,
    ReferenceCase,
)
from mimarsinan.deployment_record.correlation.library import correlate_all
from mimarsinan.deployment_record.correlation.run import correlate
from mimarsinan.deployment_record.correlation.report import render_correlation


def _case(published, census=None, overrides=None, tolerance=25.0):
    # ODIN's accelerated point over a one-second window: 37.5 MSOP/s, and one
    # latency step IS one SOP, so the window is 3.75e7 steps long.
    census = census or {
        "cores_physical": 1.0, "cells_physical": 65536.0,
        "neurons_physical": 256.0, "axons_physical": 256.0, "neurons_used": 256.0,
        "synaptic_events": 3.75e7, "macs": 3.75e7, "timesteps": 3.75e7,
        "latency_steps": 3.75e7, "segment_cores": 1.0, "sync_count": 0.0,
        "host_macs": 0.0, "host_ops_s": 0.0, "tiles": 1.0,
    }
    return ReferenceCase(
        name="c", profile="odin", citation="frenkel2019odin", description="d",
        independence="independent", operating_conditions={"supply_v": 0.55},
        census=census, census_sources={k: "published" for k in census},
        overrides=overrides or {}, published=published, tolerance_pct=tolerance,
    )


_POWER = {"average_power_mw": PublishedValue(
    value=0.477, citation="frenkel2019odin IV-A", quote="477uW")}


class TestTheComparison:
    def test_it_prices_the_census_with_the_named_profile(self):
        result = correlate(_case(_POWER))
        assert result.axes[0].predicted > 0

    def test_it_reports_the_signed_relative_error(self):
        result = correlate(_case(_POWER))
        axis = result.axes[0]
        expected = 100.0 * (axis.predicted - axis.published) / axis.published
        assert axis.error_pct == pytest.approx(expected)

    def test_a_case_inside_its_tolerance_passes(self):
        assert correlate(_case(_POWER)).passed is True

    def test_a_case_outside_its_tolerance_fails_rather_than_widening(self):
        """The tolerance is the contract; a miss is a finding, never a re-fit."""
        wrong = {"average_power_mw": PublishedValue(
            value=0.001, citation="c", quote="q")}
        result = correlate(_case(wrong))
        assert result.passed is False
        assert abs(result.axes[0].error_pct) > 25.0


class TestRefusalsTravel:
    def test_an_axis_the_physics_cannot_back_is_a_refusal_not_a_pass(self):
        """A missing prediction must never be scored as agreement."""
        case = _case({"chip_area_mm2": PublishedValue(
            value=0.086, citation="c", quote="q")},
            census={"synaptic_events": 1.0})
        result = correlate(case)
        assert result.passed is False
        assert result.axes[0].predicted is None
        assert result.axes[0].refusal is not None

    def test_an_unknown_axis_fails_loud(self):
        case = _case({"not_a_term": PublishedValue(
            value=1.0, citation="c", quote="q")})
        with pytest.raises(KeyError, match="not_a_term"):
            correlate(case)

    def test_an_unknown_profile_fails_loud(self):
        case = _case(_POWER)
        broken = ReferenceCase(**{**case.to_dict(), "profile": "nosuchchip",
                                  "published": case.published})
        with pytest.raises(KeyError, match="nosuchchip"):
            correlate(broken)


class TestOverridesAreDisclosed:
    def test_an_override_changes_the_prediction_and_is_recorded(self):
        """A case may state its own operating point, but never silently."""
        base = correlate(_case(_POWER))
        over = correlate(_case(_POWER, overrides={"e_mac": 16.86}))
        assert over.axes[0].predicted > base.axes[0].predicted
        assert "e_mac" in over.overrides_applied

    def test_an_override_of_an_undeclared_constant_fails_loud(self):
        """It must refuse with the REASON, not stumble into a bare dict KeyError:
        an override that could introduce a constant would let a case smuggle a
        number past the profile's evidence rules, and the message is what tells a
        reader that."""
        with pytest.raises(KeyError, match="never introduce a constant"):
            correlate(_case(_POWER, overrides={"e_sync_barrier": 1.0}))


class TestTheSuite:
    def test_it_runs_every_case_and_reports_each(self):
        results = correlate_all()
        assert len(results) >= 5
        assert all(r.case.name for r in results)

    def test_the_rendering_names_the_device_axis_and_error(self):
        text = render_correlation(correlate_all())
        assert "truenorth" in text and "loihi" in text and "odin" in text
        assert "%" in text

    def test_the_rendering_marks_self_consistency_cases(self):
        """A reader must be able to see which rows are circular at a glance."""
        text = render_correlation(correlate_all())
        assert "self_consistency" in text or "self-consistency" in text
