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
    # ViT WQ: full 0.855131 vs pre-transform 0.857646 — deficit 0.25pp < 1 SE: SKIP.
    assert entry_is_lossless(0.855131, 0.857646, se)
    # ViT AA: full 0.8114 vs pre-transform ~0.87 — a 6pp crater: RUN the machinery.
    assert not entry_is_lossless(0.8114, 0.87, se)
    # Transform IMPROVES entry: trivially lossless.
    assert entry_is_lossless(0.87, 0.86, se)
    # Exactly at the margin boundary: lossless (>=).
    assert entry_is_lossless(0.86 - margin * se, 0.86, se)


# --- the S=32 scenario the fast path was blind to -----------------------------
#
# Measured: the LIF step entered carrying 0.9823 and its full-transform read was
# 0.7730 — a 21pp crater. The tuner's own ENTRY probe could not see it: the
# realizable (T-anneal) ramp pins the blend at 1.0 and puts the model in the
# TRANSFORMED state at rate 0, so the entry probe measures the damage against
# itself and the ladder was declared unnecessary.
S32_PRE_TRANSFORM = 0.9823
S32_FULL_TRANSFORM = 0.7730
S32_POST_TRANSFORM_ENTRY_PROBE = 0.7702


class _Cache(dict):
    def keys(self):  # noqa: D102 - dict-with-keys() is the pipeline cache surface
        return list(super().keys())


class _Reporter:
    def __init__(self):
        self.events = []

    def report(self, *a, **k):
        pass

    def event(self, kind, payload):
        self.events.append((kind, payload))


def _stub_tuner(monkeypatch, *, full_acc, entry_probe, step_anchor):
    """A tuner double whose ENTRY probe is post-transform and whose step anchor
    is the pre-transform read the pipeline carried into the step."""
    from mimarsinan.tuning.orchestration import mbh_gate, mbh_ledger

    pipeline = SimpleNamespace(
        cache=_Cache(),
        config={"num_classes": 10},
        reporter=_Reporter(),
        get_target_metric=lambda: step_anchor,
    )
    tuner = SimpleNamespace(
        pipeline=pipeline,
        trainer=SimpleNamespace(),
        _budget=SimpleNamespace(accuracy_se=lambda: 0.0077),
        probe=lambda: entry_probe,
        _clone_state=lambda: "state",
        _committed_rate=0.0,
        _entry_short_circuited=False,
    )
    monkeypatch.setattr(
        mbh_ledger, "full_transform_measurement", lambda t: full_acc,
    )
    return mbh_gate, tuner


def test_a_crater_measured_against_the_pre_transform_read_runs_the_ladder(
    monkeypatch,
):
    """The S=32 regression: 0.9823 -> 0.7730 must NEVER take the fast path,
    however close the tuner's own post-transform entry probe sits to it."""
    mbh_gate, tuner = _stub_tuner(
        monkeypatch,
        full_acc=S32_FULL_TRANSFORM,
        entry_probe=S32_POST_TRANSFORM_ENTRY_PROBE,
        step_anchor=S32_PRE_TRANSFORM,
    )
    assert mbh_gate.lossless_entry_short_circuit(tuner) is False
    assert tuner._entry_short_circuited is False
    assert tuner._committed_rate == 0.0


def test_the_post_transform_entry_probe_alone_would_have_skipped_it(monkeypatch):
    """Teeth: the OLD reference (the entry probe) calls the same crater
    lossless, so the new test is not passing for an unrelated reason."""
    se = 0.0077
    assert entry_is_lossless(S32_FULL_TRANSFORM, S32_POST_TRANSFORM_ENTRY_PROBE, se)
    assert not entry_is_lossless(S32_FULL_TRANSFORM, S32_PRE_TRANSFORM, se)


def test_a_genuinely_lossless_transform_still_short_circuits(monkeypatch):
    """The mechanism's motivating case survives: the full transform retains the
    read the model carried INTO the step, so there is nothing to smooth."""
    mbh_gate, tuner = _stub_tuner(
        monkeypatch, full_acc=0.855131, entry_probe=0.857646,
        step_anchor=0.857646,
    )
    assert mbh_gate.lossless_entry_short_circuit(tuner) is True
    assert tuner._entry_short_circuited is True
    assert tuner._committed_rate == 1.0


def test_without_a_pre_transform_reference_the_fast_path_is_refused(monkeypatch):
    """No anchor, no licence: an unanchored step cannot prove its transform
    was free, so it runs the machinery rather than guessing."""
    mbh_gate, tuner = _stub_tuner(
        monkeypatch, full_acc=0.99, entry_probe=0.10, step_anchor=None,
    )
    assert mbh_gate.lossless_entry_short_circuit(tuner) is False


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
