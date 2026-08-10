"""AC5 per-fine-tuning-PASS wall instrumentation (task A5).

AC5 = "no fine-tuning step exceeds 5 min". The end-to-end pipeline wall (e.g.
~1009s) is dominated by NON-FT steps (Soft Core Mapping, Weight Quantization,
Simulation), so the AC5 verdict must be judged per fine-tuning PASS, not
end-to-end. These tests lock that each adaptation PASS (the recover_to / ramp /
stabilize passes inside the SmoothAdaptation cycle) is timed with a monotonic
clock, that the MAX single-pass wall is surfaced as ``max_ft_pass_wall_s`` (the
exact field name A4's AC5 verdict reads) alongside a per-pass breakdown, and that
the cost record the cost-extractor emits carries that field.

W3-S1 extends the in-memory side with the run-directory writers: the
``ft_pass_walls.json`` accumulator (step-qualified labels, read-modify-write,
atomic replace, readable by ``cost_extraction._ft_pass_walls_from_run``) and the
per-step ``retention_ledger.json``.

CRITICAL: the instrumentation is timing only — it must NOT change any numerics
(the golden-trace + torch-sim fidelity locks gate that separately).
"""

from __future__ import annotations

import json
import os
import time

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_scripted_run_tuner,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.ft_pass_wall import FtPassWallLog
from mimarsinan.tuning.orchestration import run_instrumentation
from mimarsinan.tuning.orchestration.run_ledger import (
    ENDPOINT_STEPS_CACHE_KEY,
    RETENTION_ENVELOPE_CACHE_KEY,
)


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


# --------------------------------------------------------------------------- #
# ft_pass_walls.json writer — step-qualified accumulation, atomic replace,
# readable by the ALREADY-EXISTING cost_extraction reader.
# --------------------------------------------------------------------------- #


def _metrics(*labeled_walls):
    passes = [{"label": lbl, "wall_s": wall} for lbl, wall in labeled_walls]
    return {
        "max_ft_pass_wall_s": max((p["wall_s"] for p in passes), default=0.0),
        "passes": passes,
    }


class TestFtPassWallsWriter:
    def test_accumulates_across_steps_with_qualified_labels(self, tmp_path):
        wd = str(tmp_path)
        run_instrumentation.merge_ft_pass_walls(
            wd, "LIF Adaptation", _metrics(("recover", 3.0))
        )
        merged = run_instrumentation.merge_ft_pass_walls(
            wd, "Weight Quantization", _metrics(("recover", 1.0), ("stabilize", 5.5))
        )
        assert merged is not None
        assert [p["label"] for p in merged["passes"]] == [
            "LIF Adaptation/recover",
            "Weight Quantization/recover",
            "Weight Quantization/stabilize",
        ]
        assert merged["max_ft_pass_wall_s"] == pytest.approx(5.5)
        on_disk = json.loads((tmp_path / "ft_pass_walls.json").read_text())
        assert on_disk == merged

    def test_max_is_over_all_accumulated_passes(self, tmp_path):
        # A later step with SMALLER walls must not shrink the accumulated max.
        wd = str(tmp_path)
        run_instrumentation.merge_ft_pass_walls(wd, "A", _metrics(("recover", 9.0)))
        merged = run_instrumentation.merge_ft_pass_walls(
            wd, "B", _metrics(("recover", 0.25))
        )
        assert merged is not None
        assert merged["max_ft_pass_wall_s"] == pytest.approx(9.0)

    def test_no_passes_writes_nothing(self, tmp_path):
        # A tuner that ran zero FT passes leaves the run byte-identical.
        result = run_instrumentation.merge_ft_pass_walls(
            str(tmp_path), "A", {"max_ft_pass_wall_s": 0.0, "passes": []}
        )
        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_write_is_atomic_replace(self, tmp_path, monkeypatch):
        # The COMPLETE payload lands in a temp file first, then replaces the
        # target in one os.replace; no temp residue survives.
        seen = []
        real_replace = os.replace

        def spy(src, dst):
            with open(src, "r", encoding="utf-8") as fh:
                seen.append((json.load(fh), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        run_instrumentation.merge_ft_pass_walls(
            str(tmp_path), "A", _metrics(("recover", 2.0))
        )
        assert len(seen) == 1
        payload, dst = seen[0]
        assert dst == str(tmp_path / "ft_pass_walls.json")
        assert payload["passes"] == [{"label": "A/recover", "wall_s": 2.0}]
        assert [p.name for p in tmp_path.iterdir()] == ["ft_pass_walls.json"]

    def test_reader_round_trip_via_cost_extraction(self, tmp_path):
        # The written shape MUST be readable by the pre-existing run reader.
        from mimarsinan.chip_simulation.cost_extraction import _ft_pass_walls_from_run

        wd = str(tmp_path)
        run_instrumentation.merge_ft_pass_walls(
            wd, "LIF Adaptation", _metrics(("recover", 3.0))
        )
        run_instrumentation.merge_ft_pass_walls(
            wd, "Weight Quantization", _metrics(("recover", 7.25))
        )
        max_wall, passes = _ft_pass_walls_from_run(wd)
        assert max_wall == pytest.approx(7.25)
        assert passes == (
            {"label": "LIF Adaptation/recover", "wall_s": 3.0},
            {"label": "Weight Quantization/recover", "wall_s": 7.25},
        )

    def test_scripted_tuner_metrics_round_trip(self, tmp_path):
        # The REAL tuner metric bundle (not a hand-built dict) flows through the
        # writer and back out of the cost-extraction reader.
        from mimarsinan.chip_simulation.cost_extraction import _ft_pass_walls_from_run

        tuner = _scripted_tuner(tmp_path / "cache")
        tuner.run()
        run_instrumentation.merge_ft_pass_walls(
            str(tmp_path), "LIF Adaptation", tuner.ft_pass_wall_metrics()
        )
        max_wall, passes = _ft_pass_walls_from_run(str(tmp_path))
        assert max_wall == pytest.approx(tuner.max_ft_pass_wall_s)
        assert len(passes) == len(tuner.ft_pass_walls)
        assert all(p["label"].startswith("LIF Adaptation/") for p in passes)


# --------------------------------------------------------------------------- #
# retention_ledger.json — one appended entry per tuner-hosting step.
# --------------------------------------------------------------------------- #


class _LedgerPipeline:
    """Cache/config duck for the ledger-derived entry fields."""

    def __init__(self, config=None, cache=None):
        self.config = config or {}
        self.cache = {} if cache is None else cache


class TestRetentionLedgerWriter:
    def test_entry_fields(self):
        pipeline = _LedgerPipeline(config={"endpoint_floor_steps": 8000})
        pipeline.cache[RETENTION_ENVELOPE_CACHE_KEY] = 0.97
        pipeline.cache[ENDPOINT_STEPS_CACHE_KEY] = 5000
        entry = run_instrumentation.retention_entry(
            step_name="Weight Quantization",
            entry_metric=0.95,
            exit_metric=0.93,
            pipeline=pipeline,
            consumed_before=2000,
        )
        assert entry == {
            "step": "Weight Quantization",
            "entry_metric": 0.95,
            "exit_metric": 0.93,
            "retention_delta": pytest.approx(-0.02),
            "envelope": 0.97,
            "endpoint_steps_consumed_before": 2000,
            "endpoint_steps_consumed_after": 5000,
            "endpoint_steps_total": 8000,
            "armed_recovery": True,
        }

    def test_endpoint_total_defaults_to_policy(self):
        from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

        entry = run_instrumentation.retention_entry(
            step_name="A", entry_metric=0.9, exit_metric=0.9,
            pipeline=_LedgerPipeline(), consumed_before=0,
        )
        assert entry["endpoint_steps_total"] == TUNING_POLICY.endpoint_floor_steps

    def test_unarmed_when_no_consumption_delta(self):
        # The ledger consume is the arming signature (endpoint_recovery consumes
        # only when armed): no delta => no armed recovery in this step.
        pipeline = _LedgerPipeline()
        pipeline.cache[ENDPOINT_STEPS_CACHE_KEY] = 3000
        entry = run_instrumentation.retention_entry(
            step_name="A", entry_metric=0.9, exit_metric=0.91,
            pipeline=pipeline, consumed_before=3000,
        )
        assert entry["armed_recovery"] is False
        assert entry["retention_delta"] == pytest.approx(0.01)

    def test_missing_entry_metric_yields_null_delta(self):
        entry = run_instrumentation.retention_entry(
            step_name="A", entry_metric=None, exit_metric=0.9,
            pipeline=_LedgerPipeline(), consumed_before=None,
        )
        assert entry["entry_metric"] is None
        assert entry["retention_delta"] is None
        assert entry["endpoint_steps_consumed_before"] is None
        assert entry["armed_recovery"] is False

    def test_entries_append_in_step_order(self, tmp_path):
        wd = str(tmp_path)
        run_instrumentation.append_retention_entry(wd, {"step": "A"})
        run_instrumentation.append_retention_entry(wd, {"step": "B"})
        run_instrumentation.append_retention_entry(wd, {"step": "C"})
        data = json.loads((tmp_path / "retention_ledger.json").read_text())
        assert [e["step"] for e in data["entries"]] == ["A", "B", "C"]

    def test_append_leaves_no_temp_residue(self, tmp_path):
        run_instrumentation.append_retention_entry(str(tmp_path), {"step": "A"})
        assert [p.name for p in tmp_path.iterdir()] == ["retention_ledger.json"]
