"""[recipe-economics] the lossless-entry fast path: when the FULL transform's
entry read is within noise of the blended entry read, the ladder and ramp
have nothing to smooth — skip straight to finalize. Measured motivation: the
ViT WQ tuner spent ~38 min of KD ramp to land 0.25pp BELOW its projection
entry (0.8576 → 0.8551)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver
from mimarsinan.tuning.orchestration.mbh_gate import entry_is_lossless
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


class _SchedulerMustNotRun:
    def run(self, committed, attempt):
        raise AssertionError("scheduler ran despite a lossless entry")


class _SchedulerRuns:
    def __init__(self):
        self.ran = False

    def run(self, committed, attempt):
        self.ran = True


def test_driver_short_circuits_to_finalize_on_lossless_entry():
    driver = AdaptationDriver(
        scheduler=_SchedulerMustNotRun(),
        attempt=lambda target: 1.0,
        finalize=lambda: "finalized",
        entry_short_circuit=lambda: True,
    )
    assert driver.run() == "finalized"


def test_driver_runs_scheduler_when_entry_is_damaging():
    scheduler = _SchedulerRuns()
    driver = AdaptationDriver(
        scheduler=scheduler,
        attempt=lambda target: 1.0,
        finalize=lambda: "finalized",
        entry_short_circuit=lambda: False,
    )
    assert driver.run() == "finalized"
    assert scheduler.ran


def test_driver_without_predicate_is_byte_identical_behavior():
    scheduler = _SchedulerRuns()
    driver = AdaptationDriver(
        scheduler=scheduler, attempt=lambda t: 1.0, finalize=lambda: "f",
    )
    assert driver.run() == "f"
    assert scheduler.ran


def test_predicate_on_the_three_measured_cases():
    se = 0.0077  # eval-2048 SE at acc ~0.86
    margin = TUNING_POLICY.lossless_entry_se_margin
    # ViT WQ: full 0.855131 vs blended 0.857646 — deficit 0.25pp < 1 SE: SKIP.
    assert entry_is_lossless(0.855131, 0.857646, se)
    # ViT AA: full 0.8114 vs blended ~0.87 — a 6pp crater: RUN the machinery.
    assert not entry_is_lossless(0.8114, 0.87, se)
    # Transform IMPROVES entry: trivially lossless.
    assert entry_is_lossless(0.87, 0.86, se)
    # Exactly at the margin boundary: lossless (>=).
    assert entry_is_lossless(0.86 - margin * se, 0.86, se)


def test_stabilization_budget_zeroed_after_short_circuit():
    from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
        SmoothAdaptationRunMixin,
    )

    host = SimpleNamespace(_entry_short_circuited=True)
    assert SmoothAdaptationRunMixin._stabilization_budget(host) == 0


def test_knob_off_globally_and_armed_by_the_lif_recipe():
    """The mechanism is generic; ARMING is recipe-owned (conversion_policy,
    the mode->proven-recipe SSOT) so trajectory-locked cells stay untouched
    until their mode's recipe graduates."""
    from mimarsinan.config_schema.registry import effective_value
    from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

    assert effective_value({}, "tuning_lossless_entry_fast_path") is False
    recipe = ConversionPolicy.derive("lif", None)
    assert recipe.knobs.get("tuning_lossless_entry_fast_path") is True
