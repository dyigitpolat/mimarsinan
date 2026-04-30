"""Persistence of GUI state for backfill and process-based monitoring.

Files written under ``<working_dir>/_GUI_STATE/``:

- ``steps.json``          – per-step snapshot (metrics, snapshot, status)
- ``run_info.json``       – run metadata (pid, step_names, status, config_summary)
- ``live_metrics.jsonl``  – append-only stream of metric events for live monitoring
- ``console.jsonl``       – append-only stream of stdout/stderr log lines
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

_GUI_STATE_DIR = "_GUI_STATE"
_STEPS_FILENAME = "steps.json"
_RUN_INFO_FILENAME = "run_info.json"
_LIVE_METRICS_FILENAME = "live_metrics.jsonl"
_CONSOLE_LOG_FILENAME = "console.jsonl"
_RESOURCES_DIRNAME = "resources"
_RESOURCE_EXT_FOR_MEDIA_TYPE = {
    "image/png": ".png",
    "application/json": ".json",
}

# Serialises read-modify-write access to ``steps.json``. Without this lock,
# the pipeline thread (``on_step_start`` → status=running) and the snapshot
# executor thread (``_finalize`` → status=completed) can interleave their
# load-merge-atomic_rename cycles and clobber each other's writes, making
# fast steps (Model Configuration, Model Building) intermittently appear
# never to finish because their "completed" entry gets overwritten with a
# stale view from a concurrent writer. The lock is per-process and
# per-path (via ``_steps_file_lock``) so tests using ``tmp_path`` can run
# in parallel without serialising on a single global lock.
_STEPS_FILE_LOCKS_LOCK = threading.Lock()
_STEPS_FILE_LOCKS: dict[str, threading.Lock] = {}


def _steps_file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve() if path.exists() else path.absolute())
    with _STEPS_FILE_LOCKS_LOCK:
        lock = _STEPS_FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STEPS_FILE_LOCKS[key] = lock
        return lock


def _gui_state_dir(working_directory: str) -> Path:
    return Path(working_directory) / _GUI_STATE_DIR


def _state_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _STEPS_FILENAME


def _run_info_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _RUN_INFO_FILENAME


def _live_metrics_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _LIVE_METRICS_FILENAME


def _console_log_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _CONSOLE_LOG_FILENAME


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ── steps.json ────────────────────────────────────────────────────────────

# Bound the cache so browsing many historical runs cannot grow it without
# limit. 64 distinct runs is generous for typical workflows.
_LOAD_PERSISTED_STEPS_CACHE_MAXSIZE = 64


@dataclass
class _PersistedStepsCacheInfo:
    hits: int
    misses: int
    currsize: int
    maxsize: int


class _PersistedStepsCache:
    """LRU cache keyed on ``(abspath, mtime_ns, size)``.

    Poll-heavy GUI endpoints (``list_active``, ``get_run_detail``,
    ``get_run_step_detail``) call :func:`load_persisted_steps` on every
    REST hit. For long runs the ``steps.json`` file grows to tens of
    MB and re-parsing it at several Hz saturates a CPU core.

    Keying on ``(mtime_ns, size)`` as well as the path lets us serve
    cached parses when the file hasn't changed while guaranteeing a
    fresh read whenever the snapshot executor rewrites the file.

    Thread-safety: a module-level :class:`threading.Lock` serialises
    reads/writes. The critical section is small (dict ops only — file
    I/O happens outside the lock) so contention is negligible.
    """

    def __init__(self, maxsize: int = _LOAD_PERSISTED_STEPS_CACHE_MAXSIZE) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple[str, int, int]) -> dict[str, Any] | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: tuple[str, int, int], value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> _PersistedStepsCacheInfo:
        with self._lock:
            return _PersistedStepsCacheInfo(
                hits=self._hits,
                misses=self._misses,
                currsize=len(self._store),
                maxsize=self._maxsize,
            )


_PERSISTED_STEPS_CACHE = _PersistedStepsCache()


def load_persisted_steps_cache_clear() -> None:
    """Reset the LRU cache. Primarily for tests."""
    _PERSISTED_STEPS_CACHE.clear()


def load_persisted_steps_cache_info() -> _PersistedStepsCacheInfo:
    """Expose cache statistics for tests and instrumentation."""
    return _PERSISTED_STEPS_CACHE.info()


def load_persisted_steps(working_directory: str) -> dict[str, Any]:
    """Load persisted step data from working_directory/_GUI_STATE/steps.json.

    Returns a dict keyed by step name; each value has start_time, end_time,
    target_metric, metrics, snapshot, snapshot_key_kinds. Returns {} if file
    is missing or invalid.

    The parsed result is cached on ``(abspath, mtime_ns, size)`` so repeated
    polls from the GUI server short-circuit the expensive ``json.load`` when
    the file has not changed. Missing files do **not** enter the cache, so
    a later write is picked up on the next call.
    """
    path = _state_path(working_directory)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    except OSError as e:
        logger.debug("Failed to stat persisted steps at %s: %s", path, e)
        return {}

    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _PERSISTED_STEPS_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = data.get("steps", {})
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to load persisted GUI steps from %s: %s", path, e)
        return {}

    _PERSISTED_STEPS_CACHE.put(key, result)
    return result


def write_persisted_steps_replace(working_directory: str, steps: dict[str, Any]) -> None:
    """Replace ``steps.json`` with exactly the given ``steps`` mapping (atomic write).

    Used after backfilling skipped steps so the monitor and REST APIs see completed
    step snapshots on disk, not only in-memory state.

    Holds the per-file lock so it serialises against concurrent
    :func:`save_step_to_persisted` calls.
    """
    path = _state_path(working_directory)
    with _steps_file_lock(path):
        _atomic_write_json(path, {"steps": steps})


def save_step_to_persisted(
    working_directory: str,
    step_name: str,
    start_time: float | None,
    end_time: float | None,
    target_metric: float | None,
    metrics: list[dict],
    snapshot: dict | None,
    snapshot_key_kinds: dict | None,
    *,
    status: str | None = None,
) -> None:
    """Merge one step's data into persisted steps.json and write atomically.

    Thread-safe: the read-modify-write is protected by a per-file lock so
    the pipeline thread (``on_step_start`` → ``status=running``) and the
    snapshot executor thread (``_finalize`` → ``status=completed``) cannot
    clobber each other. Without this lock, a fast step pair like
    ``Model Configuration → Model Building`` could intermittently end up
    with no ``completed`` entry on disk because the executor wrote a stale
    view back over the pipeline thread's update.
    """
    path = _state_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _steps_file_lock(path):
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                existing = data.get("steps", {})
            except (OSError, json.JSONDecodeError):
                pass

        entry: dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "target_metric": target_metric,
            "metrics": metrics,
            "snapshot": snapshot,
            "snapshot_key_kinds": snapshot_key_kinds or {},
        }
        if status is not None:
            entry["status"] = status

        existing[step_name] = entry
        _atomic_write_json(path, {"steps": existing})


def save_step_status(
    working_directory: str,
    step_name: str,
    *,
    status: str,
    end_time: float | None = None,
    target_metric: float | None = None,
) -> None:
    """Field-wise update of a single step's lifecycle fields in steps.json.

    Unlike :func:`save_step_to_persisted` this never touches the heavy
    ``snapshot`` / ``metrics`` / ``snapshot_key_kinds`` fields — it only
    upserts the listed lifecycle fields. Used by ``on_step_end`` to mark
    a step ``status="completed"`` synchronously *before* the next
    ``on_step_start`` writes ``status="running"`` for the following step;
    without this synchronous mark, the active-run watcher reads the file
    during the gap and sees two steps in ``running`` simultaneously,
    showing the previous step as still active in the pipeline bar.

    Holds the same per-file lock as :func:`save_step_to_persisted` so the
    later heavy write from the snapshot executor merges cleanly on top.
    """
    path = _state_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _steps_file_lock(path):
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                existing = data.get("steps", {})
            except (OSError, json.JSONDecodeError):
                pass

        entry = dict(existing.get(step_name) or {})
        entry["status"] = status
        if end_time is not None:
            entry["end_time"] = end_time
        if target_metric is not None:
            entry["target_metric"] = target_metric
        existing[step_name] = entry
        _atomic_write_json(path, {"steps": existing})


# ── resources (lazy heatmaps / connectivity blobs) ─────────────────────────

def _resource_root(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _RESOURCES_DIRNAME


_SAFE_SEGMENT_CHAR_RE = __import__("re").compile(r"[^A-Za-z0-9 ._()+,@=-]")


def _sanitize_path_segment(segment: str) -> str:
    """Produce a filesystem-safe, path-traversal-proof version of *segment*.

    Step names (``"Hard Core Mapping"``), resource kinds (``"ir_core_heatmap"``)
    and ``rid`` components (``"core/5"`` split on ``/``) all flow through
    this function. Spaces and common punctuation are kept verbatim so
    human-readable step names survive a disk round-trip. Any character
    outside the safe set is replaced with ``_``, which is idempotent and
    keeps the mapping stable across URL lookups and disk writes.

    The function is deliberately normalising, not validating: rejecting
    legitimate step names (e.g. anything with a space) would crash the
    snapshot executor and leave the monitor unable to persist the step.
    """
    if not segment or segment in (".", ".."):
        raise ValueError(f"Invalid path segment: {segment!r}")
    if "/" in segment or "\\" in segment or "\x00" in segment:
        raise ValueError(f"Invalid path segment: {segment!r}")
    sanitized = _SAFE_SEGMENT_CHAR_RE.sub("_", segment)
    if sanitized in (".", "..", ""):
        raise ValueError(f"Invalid path segment: {segment!r}")
    return sanitized


def _resource_disk_path(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    media_type: str,
) -> Path:
    """Build the on-disk location for a lazy resource.

    Layout: ``_GUI_STATE/resources/{step}/{kind}/{rid}.{ext}``.
    Each ``rid`` component (split on ``/``) is validated to prevent
    directory escape. Unknown media types are refused.
    """
    ext = _RESOURCE_EXT_FOR_MEDIA_TYPE.get(media_type)
    if ext is None:
        raise ValueError(f"Unsupported resource media_type: {media_type!r}")
    safe_step = _sanitize_path_segment(step_name)
    safe_kind = _sanitize_path_segment(kind)
    safe_rid_parts = [_sanitize_path_segment(p) for p in rid.split("/") if p]
    if not safe_rid_parts:
        raise ValueError(f"Empty rid: {rid!r}")
    # Append extension to the final rid segment.
    safe_rid_parts[-1] = safe_rid_parts[-1] + ext
    return _resource_root(working_directory).joinpath(safe_step, safe_kind, *safe_rid_parts)


def save_resource_to_disk(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    payload: bytes,
    *,
    media_type: str,
) -> None:
    """Write a materialized resource to ``_GUI_STATE/resources/...``.

    This lets cross-process consumers (the parent GUI server monitoring
    a subprocess-spawned run) serve heatmaps and connectivity blobs that
    were produced inside the child's ``ResourceStore``.

    The write is atomic: we write to a sibling temp file and rename, so
    an HTTP reader never sees a half-written PNG.
    """
    path = _resource_disk_path(working_directory, step_name, kind, rid, media_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write resource %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_resource_from_disk(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    *,
    media_type: str,
) -> bytes | None:
    """Read a persisted resource; returns ``None`` if absent or invalid."""
    try:
        path = _resource_disk_path(working_directory, step_name, kind, rid, media_type)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


# ── run_info.json ─────────────────────────────────────────────────────────

def save_run_info(
    working_directory: str,
    pid: int,
    step_names: list[str],
    config_summary: dict[str, Any] | None = None,
) -> None:
    """Write initial run_info.json for a headless process."""
    path = _run_info_path(working_directory)
    info = {
        "pid": pid,
        "step_names": step_names,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "config_summary": config_summary or {},
        "error": None,
    }
    _atomic_write_json(path, info)


def update_run_status(
    working_directory: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Update the status field in run_info.json."""
    path = _run_info_path(working_directory)
    info: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    info["status"] = status
    info["finished_at"] = time.time()
    if error is not None:
        info["error"] = error
    _atomic_write_json(path, info)


def load_run_info(working_directory: str) -> dict[str, Any] | None:
    """Load run_info.json. Returns None if missing."""
    path = _run_info_path(working_directory)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── live_metrics.jsonl ────────────────────────────────────────────────────

def append_live_metric(
    working_directory: str,
    step_name: str,
    metric_name: str,
    value: float,
    seq: int,
    timestamp: float,
) -> None:
    """Append a single metric event to live_metrics.jsonl."""
    path = _live_metrics_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step_name,
        "name": metric_name,
        "value": value,
        "seq": seq,
        "timestamp": timestamp,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("Failed to append metric to %s: %s", path, e)


def load_live_metrics(
    working_directory: str,
    *,
    step_name: str | None = None,
) -> list[dict[str, Any]]:
    """Read live_metrics.jsonl, optionally filtering by step."""
    path = _live_metrics_path(working_directory)
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if step_name is not None and record.get("step") != step_name:
                    continue
                results.append(record)
    except OSError:
        pass
    return results


# ── console.jsonl ──────────────────────────────────────────────────────────

def append_console_log(
    working_directory: str,
    stream: str,
    line: str,
    ts: float,
) -> None:
    """Append a single console log line to console.jsonl.

    ``stream`` is ``"stdout"`` or ``"stderr"``.
    """
    path = _console_log_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stream": stream, "line": line, "ts": ts}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("Failed to append console log to %s: %s", path, e)


def load_console_logs(
    working_directory: str,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Read console.jsonl, skipping the first ``offset`` entries."""
    path = _console_log_path(working_directory)
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    idx = 0
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                if idx < offset:
                    idx += 1
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    idx += 1
                    continue
                results.append(record)
                idx += 1
    except OSError:
        pass
    return results
