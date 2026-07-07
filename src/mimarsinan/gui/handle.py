"""GUIHandle facade: pipeline hooks and snapshot persistence."""

from __future__ import annotations

import json as _json
import logging
import sys
import time
from typing import Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runtime.events import PipelineEvent
from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.persistence import (
    append_event,
    append_live_metrics,
    append_console_log,
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
        # Buffered live-metric persistence: the trainer reports EVERY optimizer
        # step, and a per-record open/append/close on a network filesystem
        # throttles training itself (measured 12-16 steps/s vs ~118). Records
        # flush in batches; step end and shutdown flush the tail.
        self._metric_buffer: list = []
        self._metric_flushed_at = time.monotonic()
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

    _METRIC_FLUSH_COUNT = 256
    _METRIC_FLUSH_SECONDS = 2.0

    def on_metric(
        self, step_name: str, metric_name: str, value: float, seq: int, timestamp: float,
    ) -> None:
        if not self._persist_metrics:
            return
        self._metric_buffer.append({
            "step": step_name, "name": metric_name, "value": value,
            "seq": seq, "timestamp": timestamp,
        })
        if (
            len(self._metric_buffer) >= self._METRIC_FLUSH_COUNT
            or time.monotonic() - self._metric_flushed_at >= self._METRIC_FLUSH_SECONDS
        ):
            self._flush_live_metrics()

    def _flush_live_metrics(self) -> None:
        records, self._metric_buffer = self._metric_buffer, []
        self._metric_flushed_at = time.monotonic()
        if not records:
            return
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            append_live_metrics(working_dir, records)

    def on_event(self, event: PipelineEvent) -> None:
        """Persist one structured pipeline event (events.jsonl); low-rate stream."""
        if not self._persist_metrics:
            return
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            append_event(working_dir, event.to_record())

    def on_step_end(self, step_name: str, step: Any) -> None:
        self._flush_live_metrics()
        target_metric = None
        with best_effort(f"read target metric for step {step_name}", logger=logger):
            raw = self.pipeline.get_target_metric()
            target_metric = float(raw) if raw is not None else None

        # Honest metric semantics: verdict-only steps carry the previous metric
        # forward; record the kind so no view ever plots a carried value as a
        # measurement, and record the step's pass/fail verdict when it has one.
        metric_kind = None
        verdict = None
        with best_effort(f"read metric kind/verdict for step {step_name}", logger=logger):
            metric_kind = step.pipeline_metric_kind()
            verdict = step.step_verdict()

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
            metric_kind=metric_kind,
            verdict=verdict,
        )

        end_time_now = time.time()
        if working_dir:
            detail = self.collector.get_step_detail(step_name)
            if detail:
                with best_effort(f"synchronous snapshot write for {step_name}", logger=logger):
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
                        metric_kind=metric_kind,
                        verdict=verdict,
                    )
            with best_effort(f"synchronous status=completed write for {step_name}", logger=logger):
                save_step_status(
                    working_dir,
                    step_name,
                    status="completed",
                    end_time=end_time_now,
                    target_metric=target_metric,
                    metric_kind=metric_kind,
                    verdict=verdict,
                )

        def _persist_resources() -> None:
            store = self.collector.get_resource_store()
            for desc in resource_descriptors:
                payload = None
                if store is not None:
                    payload = store.prewarm(step_name, desc.kind, desc.rid)
                if payload is None:
                    produced = False
                    with best_effort(f"resource producer for {desc.kind}/{desc.rid}", logger=logger):
                        payload = desc.producer()
                        produced = True
                    if not produced:
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
        self._flush_live_metrics()
        if self._owns_snapshot_executor:
            self._snapshot_executor.shutdown()

    def wait_snapshots_idle(self, timeout: float | None = None) -> bool:
        return self._snapshot_executor.wait_idle(timeout=timeout)
