"""Load persisted GUI state from disk."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimarsinan.gui.runtime.persistence.paths import (
    console_log_path,
    events_path,
    live_metrics_path,
    run_info_path,
    steps_path,
)
from mimarsinan.gui.resources import encode_resource_payload
from mimarsinan.gui.runtime.persistence.resource_paths import resource_disk_path
from mimarsinan.gui.runtime.persistence.resource_sources import load_resource_source
from mimarsinan.gui.runtime.persistence.store import save_resource_to_disk

logger = logging.getLogger("mimarsinan.gui")

_LOAD_PERSISTED_STEPS_CACHE_MAXSIZE = 64


@dataclass
class PersistedStepsCacheInfo:
    hits: int
    misses: int
    currsize: int
    maxsize: int


class _PersistedStepsCache:
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

    def info(self) -> PersistedStepsCacheInfo:
        with self._lock:
            return PersistedStepsCacheInfo(
                hits=self._hits,
                misses=self._misses,
                currsize=len(self._store),
                maxsize=self._maxsize,
            )


_PERSISTED_STEPS_CACHE = _PersistedStepsCache()


def load_persisted_steps_cache_clear() -> None:
    _PERSISTED_STEPS_CACHE.clear()


def load_persisted_steps_cache_info() -> PersistedStepsCacheInfo:
    return _PERSISTED_STEPS_CACHE.info()


def load_persisted_steps(working_directory: str) -> dict[str, Any]:
    path = steps_path(working_directory)
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


def load_run_info(working_directory: str) -> dict[str, Any] | None:
    path = run_info_path(working_directory)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_resource_from_disk(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    *,
    media_type: str,
    variant: str = "ui",
) -> bytes | None:
    """Bytes for one persisted resource, rendering from its saved source on a miss.

    A run with no monitor attached renders nothing and persists source data
    instead, so the first fetch of a resource is also what produces it. The
    rendered bytes are written back under the normal resource path, which makes
    every later fetch -- and every later attach -- a plain file read.

    ``variant="full"`` serves the on-demand near-native render (cached as a
    sibling ``.full`` file); it needs the persisted SOURCE and returns ``None``
    without one, so callers fall back to the UI-resolution artifact.
    """
    try:
        path = resource_disk_path(working_directory, step_name, kind, rid, media_type)
    except ValueError:
        return None
    if variant == "full":
        return _load_full_variant(
            working_directory, step_name, kind, rid, path,
        )
    if path.is_file():
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None
    return _render_persisted_source(
        working_directory, step_name, kind, rid, media_type=media_type,
    )


def _load_full_variant(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    artifact_path: "Path",
) -> bytes | None:
    full_path = artifact_path.with_name(artifact_path.stem + ".full" + artifact_path.suffix)
    if full_path.is_file():
        try:
            with open(full_path, "rb") as f:
                return f.read()
        except OSError:
            return None
    source = load_resource_source(working_directory, step_name, kind, rid)
    if source is None:
        return None
    payload = source.render_full()
    if not isinstance(payload, (bytes, bytearray)):
        return None
    _write_variant_file(full_path, bytes(payload))
    return bytes(payload)


def _write_variant_file(path: "Path", payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(payload)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write resource variant %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _render_persisted_source(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    *,
    media_type: str,
) -> bytes | None:
    source = load_resource_source(working_directory, step_name, kind, rid)
    if source is None:
        return None
    payload = source.render()
    encoded = encode_resource_payload(payload, media_type)
    if encoded is None:
        logger.debug(
            "Resource source %s/%s/%s rendered a payload no %s encoder accepts",
            step_name, kind, rid, media_type,
        )
        return None
    save_resource_to_disk(
        working_directory, step_name, kind, rid, encoded, media_type=media_type,
    )
    return encoded


def load_live_metrics(
    working_directory: str,
    *,
    step_name: str | None = None,
) -> list[dict[str, Any]]:
    path = live_metrics_path(working_directory)
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


def load_console_logs(
    working_directory: str,
    offset: int = 0,
) -> list[dict[str, Any]]:
    path = console_log_path(working_directory)
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


def load_events(
    working_directory: str,
    *,
    since_seq: int = 0,
    step_name: str | None = None,
) -> list[dict[str, Any]]:
    """Read persisted structured events from events.jsonl (empty when absent)."""
    path = events_path(working_directory)
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
                if record.get("seq", 0) <= since_seq:
                    continue
                if step_name is not None and record.get("step") != step_name:
                    continue
                results.append(record)
    except OSError:
        pass
    return results
