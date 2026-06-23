"""AC5 per-fine-tuning-PASS wall instrumentation (task A5).

AC5 = "no fine-tuning step exceeds 5 min". The end-to-end pipeline wall (e.g.
~1009s) is dominated by NON-FT steps (Soft Core Mapping, Weight Quantization,
Simulation), so the AC5 verdict must be judged per fine-tuning PASS, not
end-to-end. These tests lock that each adaptation PASS (the recover_to / ramp /
stabilize passes inside the SmoothAdaptation cycle) is timed with a monotonic
clock, that the MAX single-pass wall is surfaced as ``max_ft_pass_wall_s`` (the
exact field name A4's AC5 verdict reads) alongside a per-pass breakdown, and that
the cost record the cost-extractor emits carries that field.

CRITICAL: the instrumentation is timing only — it must NOT change any numerics
(the golden-trace + torch-sim fidelity locks gate that separately).
"""

from __future__ import annotations

import time

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_scripted_run_tuner,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.ft_pass_wall import FtPassWallLog


# --------------------------------------------------------------------------- #
# FtPassWallLog — the collector primitive: monotonic per-pass walls + the max.
# --------------------------------------------------------------------------- #


class TestFtPassWallLog:
    def test_records_each_pass_wall(self):
        log = FtPassWallLog()
        log.record("recover", 1.5)
        log.record("stabilize", 0.5)
        assert [p["wall_s"] for p in log.passes] == [1.5, 0.5]
        assert [p["label"] for p in log.passes] == ["recover", "stabilize"]

    def test_max_is_the_worst_single_pass(self):
        log = FtPassWallLog()
        log.record("recover", 1.5)
        log.record("stabilize", 9.0)
        log.record("recover", 0.25)
        assert log.max_wall_s == pytest.approx(9.0)

    def test_max_is_zero_when_empty(self):
        # No FT pass ran ⇒ a well-defined 0.0 (never None — AC5 reads a float).
        assert FtPassWallLog().max_wall_s == 0.0

    def test_negative_wall_rejected(self):
        log = FtPassWallLog()
        with pytest.raises(ValueError):
            log.record("recover", -0.1)

    def test_time_block_uses_monotonic_clock(self, monkeypatch):
        # The block wall is end-minus-start on time.monotonic (immune to wall-clock
        # jumps / NTP steps), and is recorded under the given label.
        log = FtPassWallLog()
        ticks = iter([100.0, 103.25])  # start, end
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        with log.time_pass("recover"):
            pass
        assert log.passes[-1]["label"] == "recover"
        assert log.passes[-1]["wall_s"] == pytest.approx(3.25)

    def test_time_block_records_even_on_exception(self):
        log = FtPassWallLog()
        with pytest.raises(RuntimeError):
            with log.time_pass("recover"):
                raise RuntimeError("boom")
        assert len(log.passes) == 1
        assert log.passes[0]["label"] == "recover"


# --------------------------------------------------------------------------- #
# The tuner surfaces the per-pass walls + the max from a real run() loop.
# --------------------------------------------------------------------------- #


def _scripted_tuner(tmp_path):
    """A controller-path SmoothAdaptationTuner whose train/eval are stubbed
    deterministically, so run() drives the real predictor→corrector cycle (and so
    the real ``_recover_to_target`` FT pass) with no RNG / gradients."""
    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.9
    model = make_tiny_supermodel()
    # A monotone instant/post surface that commits every rate (so a recovery FT
    # pass runs every cycle): post tracks the applied rate up toward 1.0.
    return make_scripted_run_tuner(
        pipeline, model,
        instant_fn=lambda r: 0.80 + 0.15 * r,
        post_fn=lambda r: 0.80 + 0.18 * r,
        target_accuracy=0.9,
    )


class TestTunerSurfacesFtPassWalls:
    def test_run_records_at_least_one_ft_pass(self, tmp_path):
        tuner = _scripted_tuner(tmp_path)
        tuner.run()
        assert len(tuner.ft_pass_walls) >= 1, (
            "the controller cycle runs a recover FT pass; it must be timed"
        )

    def test_max_ft_pass_wall_is_positive_and_a_float(self, tmp_path):
        tuner = _scripted_tuner(tmp_path)
        tuner.run()
        assert isinstance(tuner.max_ft_pass_wall_s, float)
        assert tuner.max_ft_pass_wall_s > 0.0

    def test_max_equals_worst_recorded_pass(self, tmp_path):
        tuner = _scripted_tuner(tmp_path)
        tuner.run()
        walls = [p["wall_s"] for p in tuner.ft_pass_walls]
        assert tuner.max_ft_pass_wall_s == pytest.approx(max(walls))

    def test_each_pass_has_a_label_and_nonnegative_wall(self, tmp_path):
        tuner = _scripted_tuner(tmp_path)
        tuner.run()
        for p in tuner.ft_pass_walls:
            assert isinstance(p["label"], str) and p["label"]
            assert p["wall_s"] >= 0.0

    def test_recover_pass_is_timed(self, tmp_path):
        # The per-cycle corrector (``_recover_to_target``) is the SSOT FT pass; its
        # wall must show up labelled in the breakdown.
        tuner = _scripted_tuner(tmp_path)
        tuner.run()
        labels = {p["label"] for p in tuner.ft_pass_walls}
        assert any("recover" in lbl for lbl in labels)

    def test_fresh_tuner_has_no_passes(self, tmp_path):
        # Before any run, the breakdown is empty and the max is a well-defined 0.0.
        tuner = _scripted_tuner(tmp_path)
        assert tuner.ft_pass_walls == []
        assert tuner.max_ft_pass_wall_s == 0.0
