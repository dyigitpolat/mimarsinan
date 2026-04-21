"""Unit tests for mimarsinan.gui.active_run_stream.

Covers the file-tailer + WS-broadcast loop that replaces the 3 s poll
for subprocess-spawned active runs. The tailer-to-subscriber path must:

* Emit a ``metric`` event for every line appended to
  ``live_metrics.jsonl``.
* Emit a ``pipeline_overview`` event whenever ``steps.json`` changes on
  disk.
* Reference-count tailers per run so N subscribers share one tailer and
  tailers stop when the last subscriber disconnects.
* Truncation/rotation of the metrics file resets the read position (so a
  re-spawned subprocess doesn't silently drop its first writes).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from mimarsinan.gui.active_run_stream import ActiveRunHub
from mimarsinan.gui.persistence import (
    _GUI_STATE_DIR,
    append_live_metric,
    save_step_to_persisted,
)


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _FakeWS:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.lock = threading.Lock()

    def send(self, msg: dict) -> None:
        with self.lock:
            self.messages.append(dict(msg))

    def types(self) -> list[str]:
        with self.lock:
            return [m.get("type") for m in self.messages]


@pytest.fixture
def hub_factory():
    """Build a hub pointing at a given working directory."""
    hubs: list[ActiveRunHub] = []

    def make(run_id: str, working_dir: str, overview: dict | None = None):
        def _wd(rid: str):
            return working_dir if rid == run_id else None

        def _ov(rid: str):
            return overview if rid == run_id else None

        hub = ActiveRunHub(get_working_dir=_wd, build_overview=_ov)
        hubs.append(hub)
        return hub

    yield make
    for h in hubs:
        h.shutdown()


class TestMetricsStreaming:
    def test_new_metric_lines_are_broadcast_to_subscriber(self, tmp_path, hub_factory):
        working_dir = str(tmp_path)
        (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)
        hub = hub_factory("run-A", working_dir)

        ws = _FakeWS()
        assert hub.subscribe("run-A", ws, ws.send) is True

        append_live_metric(working_dir, "StepX", "loss", 0.5, seq=1, timestamp=100.0)
        append_live_metric(working_dir, "StepX", "loss", 0.4, seq=2, timestamp=101.0)

        assert _wait_for(lambda: len(ws.messages) >= 2), ws.messages

        metric_msgs = [m for m in ws.messages if m.get("type") == "metric"]
        assert len(metric_msgs) >= 2
        seqs = sorted(m["seq"] for m in metric_msgs if "seq" in m)
        assert 1 in seqs and 2 in seqs

    def test_subscribe_unknown_run_returns_false(self, tmp_path, hub_factory):
        hub = hub_factory("real-run", str(tmp_path))
        ws = _FakeWS()
        assert hub.subscribe("other-run", ws, ws.send) is False

    def test_multiple_subscribers_share_a_single_tailer(self, tmp_path, hub_factory):
        working_dir = str(tmp_path)
        (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)
        hub = hub_factory("run-A", working_dir)

        ws1 = _FakeWS()
        ws2 = _FakeWS()
        assert hub.subscribe("run-A", ws1, ws1.send) is True
        assert hub.subscribe("run-A", ws2, ws2.send) is True

        append_live_metric(working_dir, "StepX", "loss", 0.1, seq=1, timestamp=1.0)

        assert _wait_for(lambda: ws1.messages and ws2.messages)
        assert any(m.get("seq") == 1 for m in ws1.messages)
        assert any(m.get("seq") == 1 for m in ws2.messages)

    def test_unsubscribe_last_subscriber_stops_the_tailer(self, tmp_path, hub_factory):
        working_dir = str(tmp_path)
        (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)
        hub = hub_factory("run-A", working_dir)

        ws = _FakeWS()
        hub.subscribe("run-A", ws, ws.send)
        append_live_metric(working_dir, "StepX", "loss", 0.1, seq=1, timestamp=1.0)
        assert _wait_for(lambda: len(ws.messages) >= 1)

        hub.unsubscribe("run-A", ws)
        count_before = len(ws.messages)

        # Further appends must not reach the disconnected subscriber.
        append_live_metric(working_dir, "StepX", "loss", 0.2, seq=2, timestamp=2.0)
        time.sleep(0.2)
        assert len(ws.messages) == count_before

    def test_truncation_resets_read_position(self, tmp_path, hub_factory):
        """A shrinking file (truncate-then-nothing, or truncate-then-shorter
        append) must cause the tailer to re-read from the start; otherwise
        the stale byte offset skips the new header of the replaced log.
        """
        working_dir = str(tmp_path)
        state_dir = tmp_path / _GUI_STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "live_metrics.jsonl"

        # Seed with several longer lines so the post-truncation file is
        # strictly smaller than the pre-truncation file, giving the
        # tailer a deterministic truncation signal.
        for i in range(1, 6):
            append_live_metric(
                working_dir, "VeryLongStepName", "longmetricname",
                0.12345, seq=i, timestamp=float(i),
            )

        hub = hub_factory("run-A", working_dir)
        ws = _FakeWS()
        hub.subscribe("run-A", ws, ws.send)

        # Wait for the tailer to seek past the existing content.
        assert _wait_for(lambda: any(m.get("seq") == 5 for m in ws.messages))

        # Truncate to empty, then give the tailer a tick to observe the
        # shrink before we append the replacement log.
        path.write_text("")
        time.sleep(_poll_interval() * 2)
        append_live_metric(working_dir, "S", "m", 0.1, seq=42, timestamp=2.0)

        assert _wait_for(lambda: any(m.get("seq") == 42 for m in ws.messages)), ws.messages


def _poll_interval() -> float:
    from mimarsinan.gui.active_run_stream import _POLL_INTERVAL_S
    return _POLL_INTERVAL_S


class TestStepsOverviewStreaming:
    def test_steps_json_change_triggers_overview_event(self, tmp_path, hub_factory):
        working_dir = str(tmp_path)
        (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)

        built_overviews: list[int] = []

        def _overview(rid: str):
            if rid != "run-A":
                return None
            built_overviews.append(1)
            return {"steps": [{"name": "S1", "status": "running"}], "current_step": "S1"}

        hub = ActiveRunHub(
            get_working_dir=lambda rid: working_dir if rid == "run-A" else None,
            build_overview=_overview,
        )
        try:
            ws = _FakeWS()
            hub.subscribe("run-A", ws, ws.send)

            save_step_to_persisted(
                working_dir, step_name="S1",
                start_time=1.0, end_time=None, target_metric=None,
                metrics=[], snapshot=None, snapshot_key_kinds=None,
                status="running",
            )

            assert _wait_for(
                lambda: any(m.get("type") == "pipeline_overview" for m in ws.messages)
            ), ws.messages
        finally:
            hub.shutdown()
