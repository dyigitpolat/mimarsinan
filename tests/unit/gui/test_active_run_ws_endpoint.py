"""Integration test: the ``/ws/active_runs/{run_id}`` WebSocket endpoint
streams per-line events from a subprocess-spawned run's
``live_metrics.jsonl`` and ``steps.json`` files.

This guards the main user-facing fix for the GUI jank: without this WS
push, the frontend falls back to a 3 s poll and the charts update in
coarse batches rather than butter-smoothly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mimarsinan.gui.data_collector import DataCollector
from mimarsinan.gui.persistence import (
    _GUI_STATE_DIR,
    append_live_metric,
)
from mimarsinan.gui.server import create_app


class _StubProcessManager:
    """Minimal ProcessManager stand-in: the server only needs
    ``get_working_dir`` and ``get_run_detail`` to wire the WS endpoint."""

    def __init__(self, run_id: str, working_dir: str) -> None:
        self._run_id = run_id
        self._working_dir = working_dir

    def get_working_dir(self, run_id: str):
        return self._working_dir if run_id == self._run_id else None

    def get_run_detail(self, run_id: str):
        if run_id != self._run_id:
            return None
        return {"steps": [{"name": "S", "status": "running"}], "current_step": "S"}


def _make_client(run_id: str, working_dir: str) -> TestClient:
    collector = DataCollector()
    app = create_app(collector, process_manager=_StubProcessManager(run_id, working_dir))
    return TestClient(app)


def test_ws_streams_appended_metric_lines(tmp_path: Path) -> None:
    run_id = "run-A"
    working_dir = str(tmp_path)
    (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)
    client = _make_client(run_id, working_dir)

    with client.websocket_connect(f"/ws/active_runs/{run_id}") as ws:
        # The metric writes happen on the test thread; the server-side
        # tailer is polling every 50 ms, so give it a couple of beats.
        append_live_metric(working_dir, "S", "loss", 0.5, seq=1, timestamp=1.0)
        append_live_metric(working_dir, "S", "loss", 0.4, seq=2, timestamp=2.0)

        # Collect messages until we see both metrics (or the test timeout
        # expires). TestClient's recv is blocking; we budget a loose 3 s.
        deadline = time.monotonic() + 3.0
        seqs_seen: set[int] = set()
        while time.monotonic() < deadline and seqs_seen != {1, 2}:
            msg = ws.receive_json()
            if msg.get("type") == "metric" and "seq" in msg:
                seqs_seen.add(msg["seq"])

        assert seqs_seen == {1, 2}, f"did not receive both metric events; got {seqs_seen}"


def test_ws_rejects_unknown_run(tmp_path: Path) -> None:
    client = _make_client("known-run", str(tmp_path))
    (tmp_path / _GUI_STATE_DIR).mkdir(parents=True, exist_ok=True)

    with pytest.raises(Exception):
        # The server closes the socket with a policy-violation code;
        # TestClient surfaces this as a WebSocketDisconnect at recv time.
        with client.websocket_connect("/ws/active_runs/unknown-run") as ws:
            ws.receive_json()
