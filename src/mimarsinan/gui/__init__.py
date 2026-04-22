"""Mimarsinan Pipeline Monitoring GUI.

Usage::

    from mimarsinan.gui import start_gui

    gui = start_gui(pipeline, port=8501)
    pipeline.register_pre_step_hook(gui.on_step_start)
    pipeline.register_post_step_hook(gui.on_step_end)

The ``GUIHandle`` returned by :func:`start_gui` provides hook callbacks
and exposes the underlying :class:`DataCollector` and :class:`GUIReporter`.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from typing import Any

from mimarsinan.gui.data_collector import DataCollector
from mimarsinan.gui.persistence import (
    load_persisted_steps,
    save_resource_to_disk,
    save_step_to_persisted,
    write_persisted_steps_replace,
    append_live_metric,
    append_console_log,
)
from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.resources import ResourceStore
from mimarsinan.gui.snapshot import build_step_snapshot
from mimarsinan.gui.snapshot_executor import SnapshotExecutor


class _TeeStream(io.RawIOBase):
    """Wraps a writable stream and forwards each write to a callback.

    Lines are buffered so the callback always receives complete newline-terminated
    strings (without the trailing newline), matching what a terminal reader would see.
    """

    def __init__(self, original: Any, callback: Any) -> None:
        self._original = original
        self._callback = callback
        self._buf = ""
        self._lock = threading.Lock()

    # io.RawIOBase interface
    def writable(self) -> bool:
        return True

    def write(self, s: Any) -> int:  # type: ignore[override]
        if isinstance(s, (bytes, bytearray)):
            text = s.decode("utf-8", errors="replace")
        else:
            text = str(s)
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                try:
                    self._callback(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def flush_remaining(self) -> None:
        """Flush any buffered content that has no trailing newline."""
        with self._lock:
            if self._buf:
                try:
                    self._callback(self._buf)
                except Exception:
                    pass
                self._buf = ""

    # Proxy attribute access so third-party code that checks .encoding, .name, etc. works.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class GUIHandle:
    """Facade returned by :func:`start_gui`."""

    def __init__(
        self,
        pipeline: Any,
        collector: DataCollector,
        persist_metrics: bool = False,
        capture_stdio: bool = True,
        snapshot_executor: SnapshotExecutor | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.collector = collector
        self.reporter = GUIReporter(collector)
        self._persist_metrics = persist_metrics
        self._capture_stdio = capture_stdio
        # Owned unless the caller injected one (tests typically inject).
        self._owns_snapshot_executor = snapshot_executor is None
        self._snapshot_executor = snapshot_executor or SnapshotExecutor()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        if capture_stdio:
            self._tee_stdout = _TeeStream(sys.stdout, lambda line: self._on_console(line, "stdout"))
            self._tee_stderr = _TeeStream(sys.stderr, lambda line: self._on_console(line, "stderr"))
            sys.stdout = self._tee_stdout  # type: ignore[assignment]
            sys.stderr = self._tee_stderr  # type: ignore[assignment]
        else:
            self._tee_stdout = None
            self._tee_stderr = None

    def _on_console(self, line: str, stream: str) -> None:
        self.collector.record_console_log(line, stream)
        if self._persist_metrics:
            working_dir = getattr(self.pipeline, "working_directory", None)
            if working_dir:
                append_console_log(working_dir, stream, line, time.time())

    def restore_streams(self) -> None:
        """Restore original stdout/stderr and flush any remaining buffered lines."""
        if self._tee_stdout is not None:
            self._tee_stdout.flush_remaining()
        if self._tee_stderr is not None:
            self._tee_stderr.flush_remaining()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def on_step_start(self, step_name: str, step: Any) -> None:
        self.reporter.prefix = step_name
        self.collector.step_started(step_name)
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            import time
            save_step_to_persisted(
                working_dir, step_name,
                start_time=time.time(), end_time=None,
                target_metric=None, metrics=[], snapshot=None,
                snapshot_key_kinds=None, status="running",
            )

    def on_metric(
        self, step_name: str, metric_name: str, value: float, seq: int, timestamp: float,
    ) -> None:
        if not self._persist_metrics:
            return
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            append_live_metric(working_dir, step_name, metric_name, value, seq, timestamp)

    def on_step_end(self, step_name: str, step: Any) -> None:
        # Read the target metric synchronously while the pipeline is still
        # in its post-step state — after we return, the next step may run
        # on the pipeline thread and mutate ``pipeline.get_target_metric``.
        try:
            raw = self.pipeline.get_target_metric()
            target_metric = float(raw) if raw is not None else None
        except Exception:
            target_metric = None

        # Build the snapshot synchronously on the pipeline thread so the
        # traversal reads a consistent view of pipeline state. Snapshot
        # builders defer the *expensive* parts (matplotlib PNG encoding,
        # connectivity span extraction) into zero-arg ``ResourceDescriptor``
        # closures that capture deep copies of the relevant arrays, so
        # invoking those closures later (on the SnapshotExecutor worker or
        # inside an HTTP handler) is safe even after the pipeline mutates
        # the original tensors.
        try:
            snapshot, snapshot_key_kinds, resource_descriptors = build_step_snapshot(
                self.pipeline, step_name, step=step
            )
        except Exception as e:
            self.collector.step_failed(step_name, error=str(e))
            return

        working_dir = getattr(self.pipeline, "working_directory", None)

        # Broadcast step_completed synchronously so the UI never sees the
        # previous step "still running" while the next step's
        # ``step_started`` has already fired.  The pipeline-thread ordering
        # of broadcasts must be ``started:A → completed:A → started:B``;
        # previously the completion was queued to a background executor
        # so ``started:B`` could be broadcast before ``completed:A``,
        # giving the monitor two concurrently "running" steps.  Disk
        # persistence (PNG/JSON resource materialisation) remains
        # deferred because it's heavy and order-insensitive.
        self.collector.step_completed(
            step_name,
            target_metric=target_metric,
            snapshot=snapshot,
            snapshot_key_kinds=snapshot_key_kinds,
            resources=resource_descriptors,
        )

        def _finalize() -> None:
            if working_dir:
                detail = self.collector.get_step_detail(step_name)
                if detail:
                    save_step_to_persisted(
                        working_dir,
                        step_name,
                        detail.get("start_time"),
                        detail.get("end_time"),
                        detail.get("target_metric"),
                        detail.get("metrics", []),
                        detail.get("snapshot"),
                        detail.get("snapshot_key_kinds"),
                        status="completed",
                    )
                # Materialize heavy resources to disk so the parent GUI
                # server (watching this subprocess via ProcessManager)
                # can serve them via /api/active_runs/.../resources/...
                # without needing in-process access to the ResourceStore.
                for desc in resource_descriptors:
                    try:
                        payload = desc.producer()
                    except Exception:
                        import logging as _logging
                        _logging.getLogger("mimarsinan.gui").debug(
                            "Resource producer failed for %s/%s", desc.kind, desc.rid,
                            exc_info=True,
                        )
                        continue
                    if desc.media_type == "image/png":
                        if isinstance(payload, (bytes, bytearray)):
                            save_resource_to_disk(
                                working_dir, step_name, desc.kind, desc.rid,
                                bytes(payload), media_type=desc.media_type,
                            )
                    elif desc.media_type == "application/json":
                        import json as _json
                        try:
                            encoded = _json.dumps(payload).encode("utf-8")
                        except (TypeError, ValueError):
                            continue
                        save_resource_to_disk(
                            working_dir, step_name, desc.kind, desc.rid,
                            encoded, media_type=desc.media_type,
                        )

        # Offload collector bookkeeping and disk persistence so the
        # pipeline thread returns immediately to run the next step. The
        # executor is single-worker, so step_completed broadcasts remain
        # in submission order (see SnapshotExecutor docstring).
        self._snapshot_executor.submit(_finalize)

    def shutdown(self) -> None:
        """Drain and join the snapshot executor (flush pending persistence)."""
        if self._owns_snapshot_executor:
            self._snapshot_executor.shutdown()

    def wait_snapshots_idle(self, timeout: float | None = None) -> bool:
        """Block until all queued snapshot jobs are flushed.

        Primarily used by tests and shutdown paths that need ``steps.json``
        fully written before inspecting disk.
        """
        return self._snapshot_executor.wait_idle(timeout=timeout)


def start_gui(
    pipeline: Any,
    *,
    port: int = 8501,
    host: str = "0.0.0.0",
    start_step: str | None = None,
) -> GUIHandle:
    """Spin up the GUI server and return a handle for hook registration.

    If start_step is set, steps before it are backfilled from the pipeline cache
    so they can be browsed (metrics, Model, IR Graph, Hardware tabs) even though
    they did not run in this session.
    """
    from mimarsinan.gui.server import start_server

    collector = DataCollector()
    # Attach a resource store so snapshot builders' lazy descriptors are
    # routed to a step-scoped cache that the HTTP resource endpoints
    # (heatmaps, connectivity) can fetch from.
    collector.set_resource_store(ResourceStore())

    step_names = [name for name, _ in pipeline.steps]
    config = getattr(pipeline, "config", {})
    safe_config = _make_json_safe(config)
    collector.set_pipeline_info(step_names, safe_config)

    if start_step is not None:
        _backfill_skipped_steps(pipeline, collector, step_names, start_step)

    start_server(collector, host=host, port=port)
    handle = GUIHandle(pipeline, collector)
    return handle


def _backfill_skipped_steps(
    pipeline: Any,
    collector: DataCollector,
    step_names: list[str],
    start_step: str,
) -> None:
    """Restore steps before start_step from persisted state or cache (step-specific snapshot)."""
    try:
        start_idx = step_names.index(start_step)
    except ValueError:
        return
    working_dir = getattr(pipeline, "working_directory", "")
    persisted = load_persisted_steps(working_dir) if working_dir else {}

    step_by_name = {name: step for name, step in pipeline.steps}

    for i in range(start_idx):
        step_name = step_names[i]
        data = persisted.get(step_name)
        if data is not None:
            collector.add_step_from_persisted(
                step_name,
                data.get("start_time", 0.0),
                data.get("end_time", 0.0),
                data.get("target_metric"),
                data.get("metrics", []),
                data.get("snapshot"),
                data.get("snapshot_key_kinds"),
            )
        else:
            step = step_by_name.get(step_name)
            try:
                snapshot, snapshot_key_kinds, resource_descriptors = build_step_snapshot(
                    pipeline, step_name, step=step
                )
            except Exception:
                snapshot = None
                snapshot_key_kinds = None
                resource_descriptors = []
            collector.step_completed(
                step_name,
                target_metric=None,
                snapshot=snapshot,
                snapshot_key_kinds=snapshot_key_kinds,
                resources=resource_descriptors,
            )

    # Persist skipped steps to disk so the monitor / active-run APIs can browse them.
    # (In-memory backfill alone does not write steps.json; on_step_end only runs for executed steps.)
    if working_dir:
        _persist_skipped_steps_to_steps_json(working_dir, collector, step_names, start_idx)


def _persist_skipped_steps_to_steps_json(
    working_dir: str,
    collector: DataCollector,
    step_names: list[str],
    start_idx: int,
) -> None:
    """Write steps.json containing only steps before start_idx, from collector state."""
    merged: dict[str, Any] = {}
    for i in range(start_idx):
        name = step_names[i]
        detail = collector.get_step_detail(name)
        if not detail:
            continue
        merged[name] = {
            "start_time": detail.get("start_time"),
            "end_time": detail.get("end_time"),
            "target_metric": detail.get("target_metric"),
            "metrics": detail.get("metrics", []),
            "snapshot": detail.get("snapshot"),
            "snapshot_key_kinds": detail.get("snapshot_key_kinds") or {},
            "status": "completed",
        }
    write_persisted_steps_replace(working_dir, merged)


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
