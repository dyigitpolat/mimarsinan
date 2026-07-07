"""Buffered live-metric persistence: telemetry I/O must never throttle training.

The trainer reports a metric EVERY optimizer step; the old path did one
open/append/close per record (27k+ per run), which on a network filesystem
costs more wall than the training itself (measured 12-16 steps/s on an H100
whose training rate is ~118). The handle now buffers records and flushes them
in batches (count/age thresholds, step end, shutdown) — same JSONL format,
same reader, no training-semantics change.
"""

from __future__ import annotations

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.persistence import (
    append_live_metric,
    append_live_metrics,
    load_live_metrics,
)


class _Pipeline:
    def __init__(self, working_directory):
        self.working_directory = str(working_directory)


def _handle(tmp_path) -> GUIHandle:
    return GUIHandle(
        _Pipeline(tmp_path), DataCollector(),
        persist_metrics=True, capture_stdio=False,
    )


class TestBatchAppend:
    def test_batch_matches_single_append_format(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        records = [
            {"step": "s1", "name": "LR", "value": 0.1, "seq": 1, "timestamp": 10.0},
            {"step": "s1", "name": "LR", "value": 0.2, "seq": 2, "timestamp": 11.0},
        ]
        for r in records:
            append_live_metric(str(a), r["step"], r["name"], r["value"],
                               r["seq"], r["timestamp"])
        append_live_metrics(str(b), records)
        assert load_live_metrics(str(a)) == load_live_metrics(str(b))

    def test_empty_batch_writes_nothing(self, tmp_path):
        append_live_metrics(str(tmp_path), [])
        assert load_live_metrics(str(tmp_path)) == []


class TestHandleBuffering:
    def test_below_thresholds_defers_the_write(self, tmp_path):
        handle = _handle(tmp_path)
        handle.on_metric("s", "LR", 0.1, 1, 100.0)
        assert load_live_metrics(str(tmp_path)) == []

    def test_count_threshold_flushes_in_one_batch(self, tmp_path):
        handle = _handle(tmp_path)
        n = handle._METRIC_FLUSH_COUNT
        for i in range(n):
            handle.on_metric("s", "LR", float(i), i, 100.0 + i)
        metrics = load_live_metrics(str(tmp_path))
        assert len(metrics) == n
        assert [m["seq"] for m in metrics] == list(range(n)), "order preserved"

    def test_age_threshold_flushes(self, tmp_path, monkeypatch):
        handle = _handle(tmp_path)
        handle.on_metric("s", "LR", 0.1, 1, 100.0)
        handle._metric_flushed_at -= 2 * handle._METRIC_FLUSH_SECONDS
        handle.on_metric("s", "LR", 0.2, 2, 101.0)
        assert len(load_live_metrics(str(tmp_path))) == 2

    def test_step_end_flushes_the_tail(self, tmp_path):
        handle = _handle(tmp_path)
        handle.on_metric("s", "LR", 0.1, 1, 100.0)
        handle.on_step_end("s", object())
        assert len(load_live_metrics(str(tmp_path))) == 1

    def test_shutdown_flushes_the_tail(self, tmp_path):
        handle = _handle(tmp_path)
        handle.on_metric("s", "LR", 0.1, 1, 100.0)
        handle.shutdown()
        assert len(load_live_metrics(str(tmp_path))) == 1

    def test_persist_disabled_buffers_and_writes_nothing(self, tmp_path):
        handle = GUIHandle(
            _Pipeline(tmp_path), DataCollector(),
            persist_metrics=False, capture_stdio=False,
        )
        handle.on_metric("s", "LR", 0.1, 1, 100.0)
        handle.on_step_end("s", object())
        handle.shutdown()
        assert load_live_metrics(str(tmp_path)) == []
