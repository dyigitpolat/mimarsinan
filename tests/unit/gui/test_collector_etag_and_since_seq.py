"""Unit tests for :class:`DataCollector` ETag + ``since_seq`` filtering
and the ``pipeline_overview`` broadcast.

These back the /api/steps/{step} HTTP contract:

* ``snapshot_etag`` is a stable, monotonically-advancing token that
  changes **only** when the step's snapshot (or terminal state) is
  written; re-reading the same step with no state change yields an
  identical ETag, so HTTP handlers can return ``304 Not Modified``.

* ``get_step_detail(step, since_seq=N)`` returns only metrics with
  ``seq > N``, so repeated polls don't re-ship the entire time series.

* Pipeline-level events (step_started, step_completed, step_failed)
  trigger a ``pipeline_overview`` WebSocket broadcast carrying the full
  overview payload so the frontend can drop its periodic REST poll.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from mimarsinan.gui.data_collector import DataCollector


class _CapturingWS:
    """Minimal WebSocket stub captured by the collector's broadcast path."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.messages: list[dict] = []
        self._flush = threading.Event()

    async def send_json(self, msg: dict) -> None:
        self.messages.append(msg)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
        self._loop.close()

    def wait_messages(self, count: int, timeout: float = 1.0) -> bool:
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if len(self.messages) >= count:
                return True
            _time.sleep(0.01)
        return False


class TestSnapshotEtag:
    def test_etag_present_in_step_detail_after_completion(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.step_completed("s1", target_metric=0.5, snapshot={"x": 1})

        detail = c.get_step_detail("s1")
        assert detail is not None
        assert "snapshot_etag" in detail
        assert isinstance(detail["snapshot_etag"], str)
        assert detail["snapshot_etag"]  # non-empty

    def test_etag_stable_when_snapshot_unchanged(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.step_completed("s1", target_metric=0.5, snapshot={"x": 1})
        etag1 = c.get_step_detail("s1")["snapshot_etag"]
        etag2 = c.get_step_detail("s1")["snapshot_etag"]
        assert etag1 == etag2

    def test_etag_stable_when_only_metrics_append(self) -> None:
        """Appending metrics must NOT bump the snapshot ETag — metrics are
        paged separately via since_seq. Bumping on every metric would
        defeat the whole caching mechanism during live training."""
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.step_completed("s1", target_metric=0.5, snapshot={"x": 1})
        etag_before = c.get_step_detail("s1")["snapshot_etag"]
        c.record_metric("train_loss", 0.123)
        etag_after = c.get_step_detail("s1")["snapshot_etag"]
        assert etag_before == etag_after

    def test_etag_changes_on_rerun(self) -> None:
        """Re-running the same step (e.g. start → complete again) must
        bump the ETag so clients re-fetch the fresh snapshot."""
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.step_completed("s1", target_metric=0.5, snapshot={"x": 1})
        etag1 = c.get_step_detail("s1")["snapshot_etag"]
        # Re-run the step.
        c.step_started("s1")
        c.step_completed("s1", target_metric=0.7, snapshot={"x": 2})
        etag2 = c.get_step_detail("s1")["snapshot_etag"]
        assert etag1 != etag2

    def test_etag_differs_between_steps(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1", "s2"], {})
        c.step_completed("s1", snapshot={"x": 1})
        c.step_completed("s2", snapshot={"x": 1})
        e1 = c.get_step_detail("s1")["snapshot_etag"]
        e2 = c.get_step_detail("s2")["snapshot_etag"]
        assert e1 != e2


class TestSinceSeqFiltering:
    def test_since_seq_zero_returns_all_metrics(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        for v in [0.1, 0.2, 0.3]:
            c.record_metric("loss", v)
        detail = c.get_step_detail("s1", since_seq=0)
        assert len(detail["metrics"]) == 3

    def test_since_seq_filters_returns_only_newer(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        for v in [0.1, 0.2, 0.3, 0.4]:
            c.record_metric("loss", v)
        all_metrics = c.get_step_detail("s1", since_seq=0)["metrics"]
        # Use the second metric's seq as the checkpoint.
        checkpoint = all_metrics[1]["seq"]
        new_only = c.get_step_detail("s1", since_seq=checkpoint)["metrics"]
        assert len(new_only) == 2
        assert all(m["seq"] > checkpoint for m in new_only)

    def test_since_seq_equal_to_latest_returns_empty(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.record_metric("loss", 0.5)
        latest = c.get_step_detail("s1")["metrics"][-1]["seq"]
        assert c.get_step_detail("s1", since_seq=latest)["metrics"] == []

    def test_default_since_seq_is_zero(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        c.record_metric("loss", 0.5)
        # No explicit since_seq → full metrics list (backward compat).
        detail = c.get_step_detail("s1")
        assert len(detail["metrics"]) == 1

    def test_detail_always_reports_latest_metric_seq(self) -> None:
        """``latest_metric_seq`` lets the client advance its cursor even
        when ``metrics`` is empty (after since_seq filtering)."""
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        for v in [0.1, 0.2]:
            c.record_metric("loss", v)
        latest = c.get_step_detail("s1")["metrics"][-1]["seq"]
        detail = c.get_step_detail("s1", since_seq=latest)
        assert detail["latest_metric_seq"] == latest


class TestPipelineOverviewBroadcast:
    def test_step_started_broadcasts_pipeline_overview(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        ws = _CapturingWS()
        c.add_ws_listener(ws)
        try:
            c.step_started("s1")
            ws.wait_messages(2, timeout=1.0)
        finally:
            ws.stop()
        kinds = [m.get("type") for m in ws.messages]
        assert "pipeline_overview" in kinds
        overview_msgs = [m for m in ws.messages if m.get("type") == "pipeline_overview"]
        assert overview_msgs
        payload = overview_msgs[0]
        assert "steps" in payload
        assert payload.get("current_step") == "s1"

    def test_step_completed_broadcasts_pipeline_overview(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        ws = _CapturingWS()
        c.add_ws_listener(ws)
        try:
            c.step_started("s1")
            c.step_completed("s1", snapshot={"x": 1})
            ws.wait_messages(4, timeout=1.0)
        finally:
            ws.stop()
        overviews = [m for m in ws.messages if m.get("type") == "pipeline_overview"]
        assert len(overviews) >= 2
        # Final overview should reflect completion.
        final = overviews[-1]
        completed_rows = [s for s in final["steps"] if s["status"] == "completed"]
        assert any(s["name"] == "s1" for s in completed_rows)

    def test_step_failed_broadcasts_pipeline_overview(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        ws = _CapturingWS()
        c.add_ws_listener(ws)
        try:
            c.step_started("s1")
            c.step_failed("s1", error="boom")
            ws.wait_messages(4, timeout=1.0)
        finally:
            ws.stop()
        overviews = [m for m in ws.messages if m.get("type") == "pipeline_overview"]
        assert overviews, "expected at least one pipeline_overview broadcast"
        failed = [s for s in overviews[-1]["steps"] if s["status"] == "failed"]
        assert any(s["name"] == "s1" for s in failed)
