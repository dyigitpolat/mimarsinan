"""GUIHandle facade: pipeline hooks and snapshot persistence."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.resources import ResourceStore
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.persistence import (
    append_console_log,
    append_live_metric,
    save_resource_to_disk,
    save_step_status,
    save_step_to_persisted,
)
from mimarsinan.gui.runtime.snapshot_executor import SnapshotExecutor
from mimarsinan.gui.snapshot import build_step_snapshot
from mimarsinan.gui.tee_stream import TeeStream

logger = logging.getLogger("mimarsinan.gui")


class GUIHandle:
    """Facade returned by :func:`mimarsinan.gui.start.start_gui`."""

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
        self._owns_snapshot_executor = snapshot_executor is None
        self._snapshot_executor = snapshot_executor or SnapshotExecutor()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        if capture_stdio:
            self._tee_stdout = TeeStream(sys.stdout, lambda line: self._on_console(line, "stdout"))
            self._tee_stderr = TeeStream(sys.stderr, lambda line: self._on_console(line, "stderr"))
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
        try:
            raw = self.pipeline.get_target_metric()
            target_metric = float(raw) if raw is not None else None
        except Exception:
            target_metric = None

        try:
            snapshot, snapshot_key_kinds, resource_descriptors = build_step_snapshot(
                self.pipeline, step_name, step=step
            )
        except Exception as e:
            logger.exception("build_step_snapshot failed for %s (pipeline step succeeded)", step_name)
            snapshot = {"step_name": step_name, "snapshot_error": str(e)}
            snapshot_key_kinds = {}
            resource_descriptors = []

        working_dir = getattr(self.pipeline, "working_directory", None)

        self.collector.step_completed(
            step_name,
            target_metric=target_metric,
            snapshot=snapshot,
            snapshot_key_kinds=snapshot_key_kinds,
            resources=resource_descriptors,
        )

        end_time_now = time.time()
        if working_dir:
            detail = self.collector.get_step_detail(step_name)
            if detail:
                try:
                    save_step_to_persisted(
                        working_dir,
                        step_name,
                        detail.get("start_time"),
                        end_time_now,
                        target_metric,
                        detail.get("metrics", []),
                        detail.get("snapshot"),
                        detail.get("snapshot_key_kinds"),
                        status="completed",
                    )
                except Exception:
                    logger.debug(
                        "Synchronous snapshot write failed for %s", step_name,
                        exc_info=True,
                    )
            try:
                save_step_status(
                    working_dir,
                    step_name,
                    status="completed",
                    end_time=end_time_now,
                    target_metric=target_metric,
                )
            except Exception:
                logger.debug(
                    "Synchronous status=completed write failed for %s", step_name,
                    exc_info=True,
                )

        def _persist_resources() -> None:
            store = self.collector.get_resource_store()
            for desc in resource_descriptors:
                payload = None
                if store is not None:
                    payload = store.prewarm(step_name, desc.kind, desc.rid)
                if payload is None:
                    try:
                        payload = desc.producer()
                    except Exception:
                        logger.debug(
                            "Resource producer failed for %s/%s", desc.kind, desc.rid,
                            exc_info=True,
                        )
                        continue
                if not working_dir:
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

        self._snapshot_executor.submit(_persist_resources)

    def shutdown(self) -> None:
        if self._owns_snapshot_executor:
            self._snapshot_executor.shutdown()

    def wait_snapshots_idle(self, timeout: float | None = None) -> bool:
        return self._snapshot_executor.wait_idle(timeout=timeout)
