"""Contract for the DecisionRecord/DecisionTrace golden-trace artifact (P0)."""

import pytest

from mimarsinan.tuning.trace import DecisionRecord, DecisionTrace
from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
    SmoothAdaptationRunMixin,
)


def _catastrophic(i=0):
    return DecisionRecord(
        cycle_index=i, outcome="catastrophic", rate=0.5, committed=0.0,
        elapsed_sec=0.1, instant_acc=0.3, pre_cycle_acc=0.9,
        target=0.9, validation_baseline=0.9, rollback_tolerance=0.01,
    )


def _rollback(i=1):
    return DecisionRecord(
        cycle_index=i, outcome="rollback", rate=0.5, committed=0.25,
        elapsed_sec=0.2, instant_acc=0.85, pre_cycle_acc=0.9, post_acc=0.7,
        lr=1e-3, target=0.9, validation_baseline=0.9,
        rollback_threshold=0.89, absolute_floor=0.85, rollback_tolerance=0.01,
    )


def _commit(i=2):
    return DecisionRecord(
        cycle_index=i, outcome="commit", rate=0.5, committed=0.5,
        elapsed_sec=0.3, pre_cycle_acc=0.9, post_acc=0.9, lr=1e-3,
        reached_target=True, target=0.9, validation_baseline=0.9,
        rollback_threshold=0.89, absolute_floor=0.85, rollback_tolerance=0.01,
    )


def test_legacy_dict_key_sets_match_per_outcome():
    assert set(_catastrophic().as_legacy_dict()) == {
        "rate", "committed", "instant_acc", "outcome", "elapsed_sec",
    }
    assert set(_rollback().as_legacy_dict()) == {
        "rate", "committed", "instant_acc", "pre_cycle_acc",
        "post_acc", "lr", "outcome", "elapsed_sec",
    }
    assert set(_commit().as_legacy_dict()) == {
        "rate", "committed", "pre_cycle_acc", "post_acc",
        "lr", "reached_target", "outcome", "elapsed_sec",
    }


@pytest.mark.parametrize("rec", [_catastrophic(), _rollback(), _commit()])
def test_rate_and_committed_always_numeric(rec):
    legacy = rec.as_legacy_dict()
    assert isinstance(legacy["rate"], float)
    assert isinstance(legacy["committed"], float)
    # the printer applies :.4f — this must never raise
    _ = f"{legacy['rate']:.4f}{legacy['committed']:.4f}"


def test_log_cycle_summary_runs_over_all_outcomes():
    trace = DecisionTrace.new()
    trace.record(_catastrophic(0))
    trace.record(_rollback(1))
    trace.record(_commit(2))

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy._cycle_log = trace
    # must not raise (catastrophic lacks post_acc/lr; printer .get()s them)
    SmoothAdaptationRunMixin._log_cycle_summary(dummy)


def test_empty_trace_is_falsey_and_summary_noops():
    trace = DecisionTrace.new()
    assert not trace
    assert len(trace) == 0

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy._cycle_log = trace
    SmoothAdaptationRunMixin._log_cycle_summary(dummy)  # early-returns


def test_iteration_yields_legacy_dicts():
    trace = DecisionTrace.new()
    trace.record(_commit(0))
    entries = list(trace)
    assert entries[0]["outcome"] == "commit"
    assert trace[0]["post_acc"] == 0.9


def test_json_round_trip_is_byte_stable():
    trace = DecisionTrace.new()
    trace.record(_catastrophic(0))
    trace.record(_rollback(1))
    trace.record(_commit(2))
    text = trace.to_json()
    restored = DecisionTrace.from_json(text)
    assert restored.to_json() == text
    assert restored.records == trace.records
