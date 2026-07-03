"""Step-scoped lazy resource store for the GUI monitor."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from mimarsinan.common.best_effort import best_effort

logger = logging.getLogger("mimarsinan.gui")


@dataclass(frozen=True)
class ResourceDescriptor:
    """Metadata + lazy producer for a single resource.

    ``kind``/``rid`` compose the URL path; ``producer`` returns bytes (binary)
    or JSON-safe values, tagged by ``media_type``.
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
        """Invoke the producer at most once and return the payload; failures are sticky (return ``None``)."""
        with self._lock:
            if self._materialised:
                return self._payload
            produced = False
            with best_effort(
                f"resource producer for {self.descriptor.kind}/{self.descriptor.rid}", logger=logger,
            ):
                self._payload = self.descriptor.producer()
                produced = True
            if not produced:
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
        """Register a resource for *step*, overwriting any existing ``(kind, rid)`` and bumping the version.

        The producer is not invoked here; materialisation is lazy.
        """
        key = (desc.kind, desc.rid)
        with self._lock:
            bucket = self._store.setdefault(step, {})
            bucket[key] = _Entry(desc)
            self._versions[step] = self._versions.get(step, 0) + 1

    def prewarm(self, step: str, kind: str, rid: str) -> Any:
        """Force materialisation of an already-registered descriptor off the request thread.

        Returns the payload (``None`` on failure/missing) so the first HTTP fetch hits a hot cache.
        """
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        return entry.materialise()

    def has(self, step: str, kind: str, rid: str) -> bool:
        with self._lock:
            bucket = self._store.get(step)
            if bucket is None:
                return False
            return (kind, rid) in bucket

    def get_bytes(self, step: str, kind: str, rid: str) -> tuple[bytes, str] | None:
        """Return ``(bytes, media_type)`` for a binary resource, or ``None`` if unregistered or non-bytes."""
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = entry.materialise()
        if not isinstance(payload, (bytes, bytearray)):
            return None
        return bytes(payload), entry.descriptor.media_type

    def get_json(self, step: str, kind: str, rid: str) -> Any | None:
        """Return a JSON-safe object for a JSON resource, or ``None`` if unregistered or bytes-valued."""
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = entry.materialise()
        if isinstance(payload, (bytes, bytearray)):
            return None
        return payload

    def clear_step(self, step: str) -> None:
        """Evict every resource for *step* and bump the version; called on ``step_started`` to avoid stale bytes."""
        with self._lock:
            self._store.pop(step, None)
            self._versions[step] = self._versions.get(step, 0) + 1

    def step_version(self, step: str) -> int:
        """Monotonic version counter for *step*, bumped on every :meth:`put`/:meth:`clear_step` for ETag composition."""
        with self._lock:
            return self._versions.get(step, 0)

    def _lookup(self, step: str, kind: str, rid: str) -> _Entry | None:
        with self._lock:
            bucket = self._store.get(step)
            if bucket is None:
                return None
            return bucket.get((kind, rid))


__all__ = ["ResourceDescriptor", "ResourceStore"]
