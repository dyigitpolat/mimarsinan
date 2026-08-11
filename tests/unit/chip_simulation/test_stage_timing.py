"""StageTimer (W4.3): host ComputeOp walls are measured, neural stages never.

Pins: monotonic clock source, per-(index, name) accumulation, JSON-safe
export in the ``ComputeOpRecord.wall_s_total`` shape, and the stage-loop
wiring — ``stage_timer=None`` (the default) leaves the loop byte-identical
(the pre-existing tests in ``test_hybrid_stage_runner.py`` run unchanged).
"""

import inspect
import json
import time

import pytest

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
    run_hybrid_stages,
)
from mimarsinan.chip_simulation.hybrid_run.stage_timing import StageTimer


class _FakeClock:
    """Deterministic monotonic stand-in: each read advances by ``step``."""

    def __init__(self, step=1.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


class _Stage:
    def __init__(self, kind, name):
        self.kind = kind
        self.name = name


class _Mapping:
    def __init__(self, stages):
        self.stages = stages


class TestStageTimerUnit:
    def test_default_clock_is_the_monotonic_source(self):
        timer = StageTimer()
        assert timer._clock is time.monotonic

    def test_compute_wall_is_measured_and_exported_json_safe(self):
        timer = StageTimer(clock=_FakeClock(step=1.5))
        with timer.time_compute_stage(3, "fc_head"):
            pass
        walls = timer.compute_stage_walls()
        json.dumps(walls)  # JSON basic entries by construction
        assert walls == [{
            "stage_index": 3, "name": "fc_head",
            "wall_s_total": 1.5, "invocations": 1,
        }]

    def test_same_key_accumulates_distinct_keys_do_not(self):
        timer = StageTimer(clock=_FakeClock(step=1.0))
        with timer.time_compute_stage(1, "op_a"):
            pass
        with timer.time_compute_stage(1, "op_a"):
            pass
        with timer.time_compute_stage(2, "op_b"):
            pass
        assert timer.compute_stage_walls() == [
            {"stage_index": 1, "name": "op_a",
             "wall_s_total": 2.0, "invocations": 2},
            {"stage_index": 2, "name": "op_b",
             "wall_s_total": 1.0, "invocations": 1},
        ]

    def test_wall_accumulates_even_when_the_op_raises(self):
        timer = StageTimer(clock=_FakeClock(step=1.0))
        with pytest.raises(RuntimeError):
            with timer.time_compute_stage(0, "boom"):
                raise RuntimeError("op failed")
        (row,) = timer.compute_stage_walls()
        assert row["wall_s_total"] == 1.0
        assert row["invocations"] == 1


class TestStageLoopWiring:
    def test_compute_timed_neural_untimed(self):
        timer = StageTimer(clock=_FakeClock(step=1.0))
        mapping = _Mapping([
            _Stage("neural", "n1"),
            _Stage("compute", "c1"),
            _Stage("neural", "n2"),
            _Stage("compute", "c2"),
        ])
        run_hybrid_stages(
            mapping, {},
            on_neural=lambda ctx: None,
            on_compute=lambda ctx: None,
            stage_timer=timer,
        )
        walls = timer.compute_stage_walls()
        # Only the ComputeOp stages appear, under their EXECUTION indices;
        # the chip measures neural segments, so they are never timed.
        assert [(w["stage_index"], w["name"]) for w in walls] == [
            (1, "c1"), (3, "c2"),
        ]
        assert all(w["wall_s_total"] == 1.0 for w in walls)

    def test_stage_timer_defaults_to_none(self):
        default = inspect.signature(run_hybrid_stages).parameters["stage_timer"]
        assert default.default is None
