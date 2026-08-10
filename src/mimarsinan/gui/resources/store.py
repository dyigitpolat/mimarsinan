"""Step-scoped lazy resource store for the GUI monitor, with a bounded payload cache."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.resources.descriptor import ResourceDescriptor

logger = logging.getLogger("mimarsinan.gui")

# Cap on RESIDENT rendered bytes (PNG payloads). A GUI-attached run registers
# hundreds of heatmaps per step across many steps; descriptors and their host
# sources stay registered forever, but cached render output beyond this cap is
# evicted least-recently-used and re-materialises on the next fetch.
RESOURCE_STORE_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


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
        """Invoke the producer if no payload is cached; failures are sticky (return ``None``)."""
        with self._lock:
            if self._materialised:
                return self._payload
            produced = False
            with best_effort(
                f"resource producer for {self.descriptor.kind}/{self.descriptor.rid}", logger=logger,
            ):
                self._payload = self.descriptor.source.render()
                produced = True
            if not produced:
                self._payload = None
                self._failed = True
            self._materialised = True
            return self._payload

    def drop_payload(self) -> None:
        """Release the cached payload so the next fetch re-materialises; sticky failures stay."""
        with self._lock:
            if self._failed:
                return
            self._materialised = False
            self._payload = None


class ResourceStore:
    """Thread-safe step-scoped lazy resource cache with an LRU byte bound.

    Only rendered BYTES payloads count toward ``max_payload_bytes`` (JSON
    payloads are small and held by reference). Lock order: an entry lock is
    never held while taking another entry's lock, and the store lock is only
    held for bookkeeping, so materialisation and eviction cannot deadlock.
    """

    def __init__(self, max_payload_bytes: int = RESOURCE_STORE_MAX_PAYLOAD_BYTES) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, dict[tuple[str, str], _Entry]] = {}
        self._versions: dict[str, int] = {}
        self._max_payload_bytes = int(max_payload_bytes)
        self._resident: OrderedDict[tuple[str, str, str], int] = OrderedDict()

    def put(self, step: str, desc: ResourceDescriptor) -> None:
        """Register a resource for *step*, overwriting any existing ``(kind, rid)`` and bumping the version.

        The producer is not invoked here; materialisation is lazy.
        """
        key = (desc.kind, desc.rid)
        with self._lock:
            bucket = self._store.setdefault(step, {})
            bucket[key] = _Entry(desc)
            self._resident.pop((step, desc.kind, desc.rid), None)
            self._versions[step] = self._versions.get(step, 0) + 1

    def prewarm(self, step: str, kind: str, rid: str) -> Any:
        """Force materialisation of an already-registered descriptor off the request thread.

        Returns the payload (``None`` on failure/missing) so the first HTTP fetch hits a hot cache.
        """
        return self._materialise(step, kind, rid)

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
        payload = self._materialise(step, kind, rid, entry=entry)
        if not isinstance(payload, (bytes, bytearray)):
            return None
        return bytes(payload), entry.descriptor.media_type

    def get_json(self, step: str, kind: str, rid: str) -> Any | None:
        """Return a JSON-safe object for a JSON resource, or ``None`` if unregistered or bytes-valued."""
        entry = self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = self._materialise(step, kind, rid, entry=entry)
        if isinstance(payload, (bytes, bytearray)):
            return None
        return payload

    def clear_step(self, step: str) -> None:
        """Evict every resource for *step* and bump the version; called on ``step_started`` to avoid stale bytes."""
        with self._lock:
            self._store.pop(step, None)
            for key in [k for k in self._resident if k[0] == step]:
                self._resident.pop(key, None)
            self._versions[step] = self._versions.get(step, 0) + 1

    def step_version(self, step: str) -> int:
        """Monotonic version counter for *step*, bumped on every :meth:`put`/:meth:`clear_step` for ETag composition."""
        with self._lock:
            return self._versions.get(step, 0)

    def resident_payload_bytes(self) -> int:
        """Bytes of rendered payloads currently held (the quantity the LRU bounds)."""
        with self._lock:
            return sum(self._resident.values())

    def _lookup(self, step: str, kind: str, rid: str) -> _Entry | None:
        with self._lock:
            bucket = self._store.get(step)
            if bucket is None:
                return None
            return bucket.get((kind, rid))

    def _materialise(
        self, step: str, kind: str, rid: str, *, entry: _Entry | None = None,
    ) -> Any:
        entry = entry if entry is not None else self._lookup(step, kind, rid)
        if entry is None:
            return None
        payload = entry.materialise()
        if isinstance(payload, (bytes, bytearray)):
            self._account_and_evict((step, kind, rid), len(payload))
        return payload

    def _account_and_evict(self, key: tuple[str, str, str], size: int) -> None:
        """Note *key*'s resident bytes, then drop LRU payloads beyond the cap.

        The just-touched key is never its own victim: a single oversize payload
        stays resident (and evicts everything else) rather than thrashing.
        """
        victims: list[tuple[str, str, str]] = []
        with self._lock:
            self._resident[key] = size
            self._resident.move_to_end(key)
            while sum(self._resident.values()) > self._max_payload_bytes and len(self._resident) > 1:
                victim, _ = self._resident.popitem(last=False)
                victims.append(victim)
        for step, kind, rid in victims:
            victim_entry = self._lookup(step, kind, rid)
            if victim_entry is not None:
                victim_entry.drop_payload()


__all__ = ["RESOURCE_STORE_MAX_PAYLOAD_BYTES", "ResourceStore"]
