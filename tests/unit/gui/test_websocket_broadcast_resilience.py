"""``WebSocketMixin`` broadcast/replay must degrade a single bad listener
without raising and without disturbing healthy listeners — these paths run
on the pipeline's hot broadcast path (metrics, console lines, step events).
"""

from __future__ import annotations

import asyncio
import threading
import time as _time

from mimarsinan.gui.runtime.collector import DataCollector


class _CapturingWS:
    """Minimal WebSocket stub with a real running event loop."""

    def __init__(self, *, fail: bool = False) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.messages: list[dict] = []
        self._fail = fail

    async def send_json(self, msg: dict) -> None:
        if self._fail:
            raise RuntimeError("send_json boom")
        self.messages.append(msg)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
        self._loop.close()

    def wait_messages(self, count: int, timeout: float = 1.0) -> bool:
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if len(self.messages) >= count:
                return True
            _time.sleep(0.01)
        return False


class _NoLoopWS:
    """Listener with no ``_loop`` attribute at all (never fully connected)."""


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.01)
    return False


class TestBroadcastDropsDeadListeners:
    def test_listener_without_loop_is_removed_without_raising(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        dead = _NoLoopWS()
        c.add_ws_listener(dead)

        c._broadcast({"type": "metric"})  # must not raise

        assert dead not in c._ws_listeners

    def test_listener_whose_send_json_raises_is_removed(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        bad = _CapturingWS(fail=True)
        c.add_ws_listener(bad)
        try:
            c._broadcast({"type": "metric"})  # must not raise
            assert _wait_until(lambda: bad not in c._ws_listeners)
        finally:
            bad.stop()

    def test_healthy_listener_receives_message_and_is_kept(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        good = _CapturingWS()
        c.add_ws_listener(good)
        try:
            c._broadcast({"type": "metric", "value": 1})
            assert good.wait_messages(1)
            assert good in c._ws_listeners
        finally:
            good.stop()

    def test_bad_listener_does_not_prevent_delivery_to_healthy_listener(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        good = _CapturingWS()
        bad = _CapturingWS(fail=True)
        c.add_ws_listener(bad)
        c.add_ws_listener(good)
        try:
            c._broadcast({"type": "metric", "value": 1})
            assert good.wait_messages(1)
            assert _wait_until(lambda: bad not in c._ws_listeners)
            assert good in c._ws_listeners
        finally:
            good.stop()
            bad.stop()


class TestReplayEventsSince:
    def test_replay_pushes_pending_events_then_overview(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")  # buffers a lifecycle event
        ws = _CapturingWS()
        try:
            c.replay_events_since(ws, last_seq=0)
            assert ws.wait_messages(2)
            kinds = [m.get("type") for m in ws.messages]
            assert "pipeline_overview" in kinds
        finally:
            ws.stop()

    def test_replay_aborts_without_raising_when_send_fails(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        c.step_started("s1")
        ws = _CapturingWS(fail=True)
        try:
            c.replay_events_since(ws, last_seq=0)  # must not raise
            _time.sleep(0.1)
            assert ws.messages == []
        finally:
            ws.stop()

    def test_replay_no_pending_events_still_pushes_overview(self) -> None:
        c = DataCollector()
        c.set_pipeline_info(["s1"], {})
        ws = _CapturingWS()
        try:
            latest = c.get_current_event_seq()
            c.replay_events_since(ws, last_seq=latest)
            assert ws.wait_messages(1)
            assert ws.messages[0].get("type") == "pipeline_overview"
        finally:
            ws.stop()
