"""Step-scoped lazy resource store for the GUI monitor.

Heavy artifacts (heatmap PNGs, per-core connectivity span lists) are not
embedded in step snapshots. Instead, snapshot builders emit
:class:`ResourceDescriptor` objects that carry a zero-argument ``producer``
closure; the store materialises bytes/JSON on the first HTTP request and
caches the result for subsequent fetches.

Lifecycle:

* ``step_started(step)`` in the collector should call :meth:`clear_step` so
  a re-run of the same step does not serve stale bytes.
* ``step_completed(step, ...)`` registers any resource descriptors via
  :meth:`put`. The producer is *not* invoked here, which keeps the pipeline
  thread free of matplotlib calls.
* ``get_bytes`` / ``get_json`` are called by the FastAPI handlers in
  ``gui.server``; they run on the event loop's thread pool and are the only
  places producers execute.

The store is thread-safe and uses per-key locks so concurrent gets for the
same resource invoke the producer exactly once while independent keys can
materialise in parallel.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("mimarsinan.gui")


@dataclass(frozen=True)
class ResourceDescriptor:
    """Metadata + lazy producer for a single resource.

    Attributes:
        kind: High-level resource kind (``"heatmap"``, ``"connectivity"``,
            ``"bank_heatmap"``, ...). Used as part of the URL path.
        rid: Stable resource id within a step (e.g. ``"core/17"`` or
            ``"seg/0"``). Used as the trailing URL path segment.
        producer: Zero-argument closure returning ``bytes`` (for binary
            resources like PNG) or ``dict`` / JSON-safe values.
        media_type: HTTP ``Content-Type`` for the response. Binary producers
            must use ``image/png`` or similar; JSON producers must use
            ``application/json``.
    """

    kind: str
    rid: str
    producer: Callable[[], Any]
    media_type: str


class _Entry:
    """Internal per-resource slot with its own lock and cached payload."""

    __slots__ = ("descriptor", "_lock", "_materialised", "_payload", "_failed")

    def __init__(self, descriptor: ResourceDescriptor) -> None:
        self.descriptor = descriptor
        self._lock = threading.Lock()
        self._materialised = False
        self._payload: Any = None
        self._failed = False

    def materialise(self) -> Any:
        """Invoke the producer at most once (per entry) and return the payload.

        Returns ``None`` if the producer raised. Subsequent calls also return
        ``None`` — failures are sticky per entry so callers are not charged
        the matplotlib cost repeatedly for a broken input.
        """
        with self._lock:
            if self._materialised:
                return self._payload
            try:
                self._payload = self.descriptor.producer()
            except Exception:  # pragma: no cover - logged and surfaced as None
                logger.debug(
                    "Resource producer failed for %s/%s",
                    self.descriptor.kind,
                    self.descriptor.rid,
                    exc_info=True,
                )
                self._payload = None
                self._failed = True
            self._materialised = True
            return self._payload


class ResourceStore:
    """Thread-safe step-scoped lazy resource cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, dict[tuple[str, str], _Entry]] = {}
        self._versions: dict[str, int] = {}

    def put(self, step: str, desc: ResourceDescriptor) -> None:
        """Register a resource for *step*. Overwrites any existing entry with the
        same ``(kind, rid)`` and bumps the step's version counter.

        The producer is **not** invoked here; materialisation happens lazily
        inside :meth:`get_bytes` / :meth:`get_json`.
        """
        key = (desc.kind, desc.rid)
        with self._lock:
            bucket = self._store.setdefault(step, {})
            bucket[key] = _Entry(desc)
            self._versions[step] = self._versions.get(step, 0) + 1

    def has(self, step: str, kind: str, rid: str) -> bool:
        with self._lock:
            bucket = self._store.get(step)
            if bucket is None:
                return False
            return (kind, rid) in bucket

    def get_bytes(self, step: str, kind: str, rid: str) -> tuple[bytes, str] | None:
        """Return ``(bytes, media_type)`` for a binary resource, or ``None``.

        Returns ``None`` when the resource is not registered or when the
        registered producer returned a non-bytes payload.
        """
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = entry.materialise()
        if not isinstance(payload, (bytes, bytearray)):
            return None
        return bytes(payload), entry.descriptor.media_type

    def get_json(self, step: str, kind: str, rid: str) -> Any | None:
        """Return a JSON-safe Python object for a JSON resource, or ``None``.

        Returns ``None`` when the resource is not registered or when the
        registered producer returned bytes (callers must use
        :meth:`get_bytes` for binary payloads).
        """
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = entry.materialise()
        if isinstance(payload, (bytes, bytearray)):
            return None
        return payload

    def clear_step(self, step: str) -> None:
        """Evict every resource for *step* and bump the version counter.

        Called on ``step_started`` so a re-run of the same step never serves
        stale bytes. The version bump invalidates any cached URLs the client
        may still be holding onto.
        """
        with self._lock:
            self._store.pop(step, None)
            self._versions[step] = self._versions.get(step, 0) + 1

    def step_version(self, step: str) -> int:
        """Monotonically increasing version counter for *step*.

        Incremented on every :meth:`put` and :meth:`clear_step`. Used by the
        collector to compose ETags so a re-run (or late resource registration)
        invalidates any cached HTTP response.
        """
        with self._lock:
            return self._versions.get(step, 0)

    def _lookup(self, step: str, kind: str, rid: str) -> _Entry | None:
        with self._lock:
            bucket = self._store.get(step)
            if bucket is None:
                return None
            return bucket.get((kind, rid))


__all__ = ["ResourceDescriptor", "ResourceStore"]
