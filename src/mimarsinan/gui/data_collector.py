"""Central event store for the GUI monitoring system.

Accumulates metric time-series, step lifecycle events, and rich snapshot
data.  Thread-safe so the FastAPI server can read while the pipeline
writes from its own thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mimarsinan.gui.resources import ResourceDescriptor, ResourceStore

logger = logging.getLogger("mimarsinan.gui")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MetricEvent:
    seq: int
    step_name: str
    metric_name: str
    value: Any
    timestamp: float
    global_step: int | None = None


@dataclass
class ConsoleLogEntry:
    seq: int
    stream: str  # "stdout" or "stderr"
    line: str
    ts: float


@dataclass
class StepRecord:
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    target_metric: float | None = None
    snapshot: dict | None = None
    snapshot_key_kinds: dict | None = None  # snapshot key -> "new" | "edited"
    error: str | None = None  # set when step_failed; shown in step detail
    # Monotonic version bumped each time the snapshot or terminal state
    # is written. Used to build a stable HTTP ETag for /api/steps/{name}
    # so the frontend can short-circuit identical polls.
    snapshot_version: int = 0


class DataCollector:
    """Centralized, thread-safe store for pipeline monitoring data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._step_names: list[str] = []
        self._steps: dict[str, StepRecord] = {}
        self._metrics: list[MetricEvent] = []
        self._metric_seq: int = 0
        self._current_step: str | None = None
        self._pipeline_config: dict | None = None

        self._console_logs: list[ConsoleLogEntry] = []
        self._console_seq: int = 0

        self._ws_listeners: list[Any] = []
        self._pipeline_thread: Optional[threading.Thread] = None
        self._metric_callback: Any = None
        self._console_callback: Any = None

        self._resource_store: Optional["ResourceStore"] = None

    # -- Resource store wiring -------------------------------------------------

    def set_resource_store(self, store: "ResourceStore | None") -> None:
        """Attach a lazy :class:`ResourceStore` for step-scoped heavy artifacts.

        Snapshot builders emit ``ResourceDescriptor`` objects alongside the
        lightweight summary dict; when set, the collector routes those
        descriptors to *store* on ``step_completed`` and evicts them on
        ``step_started`` so re-runs never serve stale bytes.
        """
        with self._lock:
            self._resource_store = store

    def get_resource_store(self) -> "ResourceStore | None":
        with self._lock:
            return self._resource_store

    # -- Pipeline thread (for graceful exit) -----------------------------------

    def set_pipeline_thread(self, thread: Optional[threading.Thread]) -> None:
        with self._lock:
            self._pipeline_thread = thread

    def get_pipeline_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._pipeline_thread

    def join_pipeline_thread(self, timeout: float = 30.0) -> bool:
        """Wait for the pipeline thread to finish. Returns True if joined, False if timeout."""
        with self._lock:
            t = self._pipeline_thread
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    # -- Pipeline configuration ------------------------------------------------

    def set_pipeline_info(self, step_names: list[str], config: dict) -> None:
        with self._lock:
            self._step_names = list(step_names)
            self._pipeline_config = dict(config)
            self._steps = {
                name: StepRecord(name=name) for name in step_names
            }

    @property
    def pipeline_config(self) -> dict | None:
        with self._lock:
            return self._pipeline_config

    # -- Step lifecycle --------------------------------------------------------

    def step_started(self, step_name: str) -> None:
        with self._lock:
            self._current_step = step_name
            rec = self._steps.get(step_name)
            if rec is None:
                rec = StepRecord(name=step_name)
                self._steps[step_name] = rec
                if step_name not in self._step_names:
                    self._step_names.append(step_name)
            rec.status = StepStatus.RUNNING
            rec.start_time = time.time()
            rec.end_time = None
            rec.target_metric = None
            rec.snapshot = None
            rec.snapshot_key_kinds = None
            rec.snapshot_version += 1
            self._metrics = [m for m in self._metrics if m.step_name != step_name]
            store = self._resource_store
        if store is not None:
            store.clear_step(step_name)
        self._broadcast({"type": "step_started", "step": step_name})
        self._broadcast_pipeline_overview()

    def step_completed(
        self,
        step_name: str,
        target_metric: float | None = None,
        snapshot: dict | None = None,
        snapshot_key_kinds: dict | None = None,
        resources: Iterable["ResourceDescriptor"] | None = None,
    ) -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.COMPLETED
                rec.end_time = time.time()
                rec.target_metric = target_metric
                if snapshot is not None:
                    rec.snapshot = snapshot
                if snapshot_key_kinds is not None:
                    rec.snapshot_key_kinds = snapshot_key_kinds
                rec.snapshot_version += 1
            if self._current_step == step_name:
                self._current_step = None
            store = self._resource_store
        if resources and store is not None:
            for desc in resources:
                store.put(step_name, desc)
        self._broadcast({
            "type": "step_completed",
            "step": step_name,
            "target_metric": target_metric,
        })
        self._broadcast_pipeline_overview()

    def step_failed(self, step_name: str, error: str = "") -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.FAILED
                rec.end_time = time.time()
                rec.error = error or None
                rec.snapshot_version += 1
            if self._current_step == step_name:
                self._current_step = None
        self._broadcast({"type": "step_failed", "step": step_name, "error": error})
        self._broadcast_pipeline_overview()

    def add_step_from_persisted(
        self,
        step_name: str,
        start_time: float,
        end_time: float,
        target_metric: float | None,
        metrics: list[dict],
        snapshot: dict | None,
        snapshot_key_kinds: dict | None = None,
    ) -> None:
        """Restore a step record and its metrics from persisted state (e.g. backfill)."""
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                rec = StepRecord(name=step_name)
                self._steps[step_name] = rec
                if step_name not in self._step_names:
                    self._step_names.append(step_name)
            rec.status = StepStatus.COMPLETED
            rec.start_time = start_time
            rec.end_time = end_time
            rec.target_metric = target_metric
            rec.snapshot = snapshot
            rec.snapshot_key_kinds = snapshot_key_kinds

            for m in metrics:
                seq = m.get("seq", 0)
                if seq > self._metric_seq:
                    self._metric_seq = seq
                self._metrics.append(MetricEvent(
                    seq=seq,
                    step_name=step_name,
                    metric_name=m.get("name", ""),
                    value=m.get("value"),
                    timestamp=m.get("timestamp", 0.0),
                    global_step=m.get("global_step"),
                ))

    # -- Metrics ---------------------------------------------------------------

    def record_metric(self, metric_name: str, value: Any, step: int | None = None) -> None:
        with self._lock:
            self._metric_seq += 1
            current = self._current_step or ""
            evt = MetricEvent(
                seq=self._metric_seq,
                step_name=current,
                metric_name=metric_name,
                value=_to_json_safe(value),
                timestamp=time.time(),
                global_step=step,
            )
            self._metrics.append(evt)
            cb = self._metric_callback
        self._broadcast({
            "type": "metric",
            "step": current,
            "name": metric_name,
            "value": evt.value,
            "seq": evt.seq,
            "timestamp": evt.timestamp,
        })
        if cb is not None:
            try:
                cb(current, metric_name, evt.value, evt.seq, evt.timestamp)
            except Exception:
                pass

    # -- Console logs ----------------------------------------------------------

    def record_console_log(self, line: str, stream: str) -> None:
        """Record a console output line and broadcast it via WebSocket."""
        with self._lock:
            self._console_seq += 1
            entry = ConsoleLogEntry(
                seq=self._console_seq,
                stream=stream,
                line=line,
                ts=time.time(),
            )
            self._console_logs.append(entry)
            cb = self._console_callback
        self._broadcast({
            "type": "console_log",
            "stream": stream,
            "line": line,
            "ts": entry.ts,
            "seq": entry.seq,
        })
        if cb is not None:
            try:
                cb(stream, line, entry.ts)
            except Exception:
                pass

    def get_console_logs(self, offset: int = 0) -> list[dict]:
        with self._lock:
            return [
                {"seq": e.seq, "stream": e.stream, "line": e.line, "ts": e.ts}
                for e in self._console_logs[offset:]
            ]

    # -- Read API (called by the server) --------------------------------------

    def get_pipeline_overview(self) -> dict:
        with self._lock:
            config = self._pipeline_config or {}
        try:
            from mimarsinan.pipelining.pipelines.deployment_pipeline import (
                get_pipeline_semantic_group_by_step_name,
            )
            groups = get_pipeline_semantic_group_by_step_name(config)
        except Exception:
            groups = {}
        with self._lock:
            steps = []
            for name in self._step_names:
                rec = self._steps.get(name, StepRecord(name=name))
                steps.append({
                    "name": rec.name,
                    "status": rec.status.value,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                    "duration": (rec.end_time - rec.start_time) if rec.start_time and rec.end_time else None,
                    "target_metric": rec.target_metric,
                    "semantic_group": groups.get(rec.name),
                })
            return {
                "steps": steps,
                "current_step": self._current_step,
                "config": self._pipeline_config,
            }

    def get_step_detail(
        self,
        step_name: str,
        *,
        since_seq: int = 0,
    ) -> dict | None:
        """Return step detail, optionally filtering metrics to ``seq > since_seq``.

        The returned payload includes:

        * ``snapshot_etag`` — monotonic token ``W/"{step}-{version}"``.
          The version advances on ``step_started`` / ``step_completed`` /
          ``step_failed`` but *not* on metric appends, so clients can set
          ``If-None-Match`` and get ``304 Not Modified`` during live
          training while still streaming new metrics via ``since_seq``.
        * ``latest_metric_seq`` — highest metric seq for this step (even
          if the filtered ``metrics`` list is empty), so the client can
          advance its cursor without waiting for a non-empty response.
        """
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                return None
            step_metrics = [m for m in self._metrics if m.step_name == step_name]
            latest_metric_seq = max((m.seq for m in step_metrics), default=0)
            metrics = [
                {
                    "seq": m.seq,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "global_step": m.global_step,
                }
                for m in step_metrics
                if m.seq > since_seq
            ]
            return {
                "name": rec.name,
                "status": rec.status.value,
                "start_time": rec.start_time,
                "end_time": rec.end_time,
                "duration": (rec.end_time - rec.start_time) if rec.start_time and rec.end_time else None,
                "target_metric": rec.target_metric,
                "metrics": metrics,
                "latest_metric_seq": latest_metric_seq,
                "snapshot": rec.snapshot,
                "snapshot_key_kinds": rec.snapshot_key_kinds,
                "snapshot_etag": _build_snapshot_etag(rec),
                "error": rec.error,
            }

    def get_step_snapshot_etag(self, step_name: str) -> str | None:
        """Cheap ETag lookup for HTTP handlers doing ``If-None-Match`` checks."""
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                return None
            return _build_snapshot_etag(rec)

    def get_step_metrics(self, step_name: str) -> list[dict]:
        with self._lock:
            return [
                {
                    "seq": m.seq,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "global_step": m.global_step,
                }
                for m in self._metrics
                if m.step_name == step_name
            ]

    def get_all_metrics(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "seq": m.seq,
                    "step": m.step_name,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in self._metrics
            ]

    # -- Pipeline overview broadcasts ------------------------------------------

    def _broadcast_pipeline_overview(self) -> None:
        """Push a full pipeline overview to WS listeners.

        Frontends relied on a 5 s REST poll (``/api/pipeline``) to learn
        about step transitions; piggy-backing an overview message on
        every step_started / step_completed / step_failed lets the UI
        update instantly while still allowing the poll to serve as a
        cheap watchdog.

        We reuse :meth:`get_pipeline_overview` (which handles its own
        locking + semantic-group lookup) and forward the result via the
        existing broadcast path, so the message is sanitized the same
        way as all other WS payloads.
        """
        try:
            overview = self.get_pipeline_overview()
        except Exception:
            logger.debug("Failed to build pipeline overview for broadcast", exc_info=True)
            return
        self._broadcast({"type": "pipeline_overview", **overview})

    # -- WebSocket listener management ----------------------------------------

    def add_ws_listener(self, ws: Any) -> None:
        with self._lock:
            self._ws_listeners.append(ws)

    def remove_ws_listener(self, ws: Any) -> None:
        with self._lock:
            if ws in self._ws_listeners:
                self._ws_listeners.remove(ws)

    def _broadcast(self, message: dict) -> None:
        import asyncio
        import math

        def _ws_sanitize(obj):
            """Recursively replace non-finite floats with None."""
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _ws_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_ws_sanitize(v) for v in obj]
            return obj

        safe_message = _ws_sanitize(message)

        with self._lock:
            listeners = list(self._ws_listeners)
        dead: list[Any] = []
        for ws in listeners:
            try:
                loop = ws._loop if hasattr(ws, "_loop") else None
                if loop is not None and loop.is_running():
                    # Wait for the send to actually complete before the
                    # next broadcast is scheduled.  Without this, two
                    # back-to-back broadcasts from the pipeline thread
                    # (e.g. ``step_completed:A`` then ``step_started:B``)
                    # are submitted as independent asyncio tasks that
                    # can interleave at ``await`` points inside
                    # ``ws.send_json`` — the UI then sees B "start"
                    # before A's "completed" frame arrives and renders
                    # both steps as concurrently running.  A short
                    # timeout bounds the pipeline thread's blocking in
                    # case a listener is stuck.
                    fut = asyncio.run_coroutine_threadsafe(
                        ws.send_json(safe_message), loop,
                    )
                    try:
                        fut.result(timeout=2.0)
                    except Exception:
                        logger.debug(
                            "WebSocket send timed out or failed; dropping listener",
                            exc_info=True,
                        )
                        dead.append(ws)
                else:
                    dead.append(ws)
            except Exception:
                logger.debug("Failed to broadcast to WebSocket, removing listener", exc_info=True)
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._ws_listeners:
                        self._ws_listeners.remove(ws)


def _build_snapshot_etag(rec: StepRecord) -> str:
    """Construct a weak HTTP ETag for a step record.

    Format: ``W/"{step_name}-{status}-{version}"``. The status component
    makes ``pending`` → ``running`` → ``completed``/``failed`` transitions
    visible even if ``snapshot_version`` somehow stays the same
    (defensive; normally the version advances at every lifecycle edge).
    We use a weak validator because the payload is not byte-stable (JSON
    key ordering etc.) — only its logical content version matters.
    """
    return f'W/"{rec.name}-{rec.status.value}-{rec.snapshot_version}"'


def _to_json_safe(value: Any) -> Any:
    """Convert tensors / numpy values to plain Python types.

    Also replaces NaN/Inf with ``None`` so the result is always
    JSON-serialisable.
    """
    import math

    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return str(value)
