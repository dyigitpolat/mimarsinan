"""Determinism + RNG-isolation invariants (report IV.8, invariant I6).

Two guarantees the controller relies on: (1) a probe block wrapped in
``rng_snapshot()`` leaves the committed path's RNG trajectory untouched, so
speculative evaluations never perturb reproducibility; and (2) a scripted full
``run()`` under a fixed seed replays a bit-identical decision trace. Both are
pure / GPU-free and run in milliseconds.
"""

import json
import random
import tempfile

import numpy as np
import torch

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
    rng_snapshot,
)
from mimarsinan.tuning.trace import DecisionTrace


# ``elapsed_sec`` is wall clock and ``seeds`` is environment metadata — neither
# is a decision field, and ``assert_trace_matches`` excludes them too. I6 is the
# reproducibility of the *decisions*, so we normalize them away before compare.
_NON_DECISION_FIELDS = ("elapsed_sec", "seeds")


def _decision_json(trace: DecisionTrace) -> str:
    records = json.loads(trace.to_json())
    for rec in records:
        for field in _NON_DECISION_FIELDS:
            rec.pop(field, None)
    return json.dumps(records, sort_keys=True)


# ── I6a: rng_snapshot isolates a probe block from the committed RNG path ──────

def test_rng_snapshot_isolates_a_probe_block():
    torch.manual_seed(20240611)
    committed_before = torch.rand(4)

    # A speculative probe draws a different, larger number of randoms; the
    # snapshot must rewind the RNG so the committed path is unaffected.
    with rng_snapshot():
        probe = torch.rand(37)
    committed_after = torch.rand(4)
    assert probe.numel() == 37

    # The committed path's draws are exactly what they would be with no probe.
    torch.manual_seed(20240611)
    assert torch.allclose(torch.rand(4), committed_before)
    assert torch.allclose(torch.rand(4), committed_after)


def test_rng_snapshot_isolates_numpy_and_python():
    np.random.seed(11)
    random.seed(11)
    np_before = np.random.rand()
    py_before = random.random()
    with rng_snapshot():
        np.random.rand(50)
        [random.random() for _ in range(50)]
    np_after = np.random.rand()
    py_after = random.random()

    np.random.seed(11)
    random.seed(11)
    assert np.random.rand() == np_before
    assert random.random() == py_before
    assert np.random.rand() == np_after
    assert random.random() == py_after


def test_deterministic_rng_fixture_pins_the_seed(deterministic_rng):
    drawn = torch.rand(5)
    torch.manual_seed(0)
    assert torch.allclose(drawn, torch.rand(5))


# ── I6b: a scripted run replays an identical decision trace under a seed ──────

def _scripted_trace(*, instant, post, ladder=False):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    pipeline = MockPipeline(config=cfg, working_directory=tempfile.mkdtemp())
    tuner = make_scripted_run_tuner(
        pipeline,
        make_tiny_supermodel(),
        instant_fn=instant,
        post_fn=post,
        ladder=ladder,
    )
    tuner.run()
    return tuner._cycle_log


def test_scripted_run_replays_identical_trace():
    a = _scripted_trace(instant=lambda r: 0.87, post=lambda r: 0.9)
    b = _scripted_trace(instant=lambda r: 0.87, post=lambda r: 0.9)
    assert _decision_json(a) == _decision_json(b)
    assert len(a) == len(b)
    assert len(a) >= 1


def test_scripted_ssa_ramp_replays_identical_trace():
    # One-shot at 1.0 fails catastrophically → the SmartSmoothAdaptation ramp
    # runs (a multi-cycle trace) — still bit-identical across runs.
    instant = lambda r: 0.1 if r >= 0.99 else 0.85
    a = _scripted_trace(instant=instant, post=lambda r: 0.9)
    b = _scripted_trace(instant=instant, post=lambda r: 0.9)
    outcomes = [r.outcome for r in a.records]
    assert outcomes[0] == "catastrophic"
    assert len(a.records) > 1  # the ramp produced multiple cycles
    assert _decision_json(a) == _decision_json(b)


def test_scripted_ladder_replays_identical_trace():
    a = _scripted_trace(instant=lambda r: 0.87, post=lambda r: 0.9, ladder=True)
    b = _scripted_trace(instant=lambda r: 0.87, post=lambda r: 0.9, ladder=True)
    assert _decision_json(a) == _decision_json(b)


def test_decision_trace_json_round_trips_through_from_json():
    trace = _scripted_trace(instant=lambda r: 0.87, post=lambda r: 0.9)
    restored = DecisionTrace.from_json(trace.to_json())
    assert restored.to_json() == trace.to_json()
