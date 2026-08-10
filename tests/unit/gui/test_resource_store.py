"""Unit tests for the step-scoped lazy resource store."""

import threading

import pytest

from mimarsinan.gui.resources import ResourceDescriptor, ResourceStore
from resource_source_doubles import CallableSource


def _png_descriptor(rid: str, payload: bytes, calls: list[str]) -> ResourceDescriptor:
    def produce():
        calls.append(rid)
        return payload

    return ResourceDescriptor(
        kind="heatmap",
        rid=rid,
        source=CallableSource(produce),
        media_type="image/png",
    )


def _json_descriptor(rid: str, payload: dict, calls: list[str]) -> ResourceDescriptor:
    def produce():
        calls.append(rid)
        return payload

    return ResourceDescriptor(
        kind="connectivity",
        rid=rid,
        source=CallableSource(produce),
        media_type="application/json",
    )


class TestResourceStorePutAndGet:
    def test_get_bytes_returns_payload_and_media_type(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"PNGPAYLOAD", calls))

        result = store.get_bytes("Step A", "heatmap", "core/1")
        assert result is not None
        data, media_type = result
        assert data == b"PNGPAYLOAD"
        assert media_type == "image/png"

    def test_get_json_returns_dict_and_media_type(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _json_descriptor("seg/0", {"spans": [1, 2, 3]}, calls))

        result = store.get_json("Step A", "connectivity", "seg/0")
        assert result == {"spans": [1, 2, 3]}

    def test_returns_none_when_missing(self):
        store = ResourceStore()
        assert store.get_bytes("nope", "heatmap", "core/1") is None
        assert store.get_json("nope", "connectivity", "seg/0") is None

    def test_has_reports_presence(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"X", calls))
        assert store.has("Step A", "heatmap", "core/1") is True
        assert store.has("Step A", "heatmap", "core/2") is False
        assert store.has("Other", "heatmap", "core/1") is False


class TestResourceStoreLazyProducer:
    def test_producer_not_invoked_on_put(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"A", calls))
        assert calls == []

    def test_producer_invoked_on_first_get_only(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"A", calls))

        store.get_bytes("Step A", "heatmap", "core/1")
        store.get_bytes("Step A", "heatmap", "core/1")
        store.get_bytes("Step A", "heatmap", "core/1")

        assert calls == ["core/1"]

    def test_bytes_and_json_use_separate_endpoints(self):
        """Serving a JSON resource via get_bytes returns None (wrong kind match on media_type)."""
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _json_descriptor("seg/0", {"x": 1}, calls))
        assert store.get_bytes("Step A", "connectivity", "seg/0") is None
        assert store.get_json("Step A", "connectivity", "seg/0") == {"x": 1}


class TestResourceStoreClearStep:
    def test_clear_step_evicts_only_that_step(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"A1", calls))
        store.put("Step B", _png_descriptor("core/1", b"B1", calls))

        store.clear_step("Step A")

        assert store.get_bytes("Step A", "heatmap", "core/1") is None
        b = store.get_bytes("Step B", "heatmap", "core/1")
        assert b is not None and b[0] == b"B1"

    def test_clear_step_bumps_version(self):
        store = ResourceStore()
        v0 = store.step_version("Step A")
        store.put("Step A", _png_descriptor("core/1", b"A", []))
        v1 = store.step_version("Step A")
        assert v1 > v0
        store.clear_step("Step A")
        v2 = store.step_version("Step A")
        assert v2 > v1

    def test_put_after_clear_reinvokes_producer(self):
        store = ResourceStore()
        calls: list[str] = []
        store.put("Step A", _png_descriptor("core/1", b"OLD", calls))
        store.get_bytes("Step A", "heatmap", "core/1")
        assert calls == ["core/1"]

        store.clear_step("Step A")
        store.put("Step A", _png_descriptor("core/1", b"NEW", calls))
        data, _ = store.get_bytes("Step A", "heatmap", "core/1")  # type: ignore[misc]
        assert data == b"NEW"
        assert calls == ["core/1", "core/1"]


class TestResourceStoreProducerFailure:
    def test_failing_producer_returns_none_and_may_retry(self):
        store = ResourceStore()

        class _Boom:
            calls = 0

            def __call__(self):
                _Boom.calls += 1
                raise RuntimeError("nope")

        boom = _Boom()
        store.put(
            "Step A",
            ResourceDescriptor(
                kind="heatmap",
                rid="core/1",
                source=CallableSource(boom),
                media_type="image/png",
            ),
        )
        assert store.get_bytes("Step A", "heatmap", "core/1") is None
        assert boom.calls == 1


class TestResourceStoreLruBound:
    """The payload cache is BOUNDED: rendered bytes beyond the byte cap are
    evicted least-recently-used and re-materialise on demand (descriptors and
    their sources stay registered — only cached payload bytes are dropped)."""

    def test_default_cap_is_the_documented_constant(self):
        from mimarsinan.gui.resources.store import RESOURCE_STORE_MAX_PAYLOAD_BYTES

        assert RESOURCE_STORE_MAX_PAYLOAD_BYTES == 64 * 1024 * 1024
        assert ResourceStore()._max_payload_bytes == RESOURCE_STORE_MAX_PAYLOAD_BYTES

    def test_eviction_respects_cap_and_rematerialises_on_demand(self):
        store = ResourceStore(max_payload_bytes=100)
        calls: list[str] = []
        for rid in ("core/0", "core/1", "core/2"):
            store.put("S", _png_descriptor(rid, b"x" * 40, calls))

        for rid in ("core/0", "core/1", "core/2"):
            store.get_bytes("S", "heatmap", rid)
        assert calls == ["core/0", "core/1", "core/2"]
        assert store.resident_payload_bytes() <= 100

        # core/0 was least recently used -> evicted -> next get re-renders it.
        store.get_bytes("S", "heatmap", "core/0")
        assert calls == ["core/0", "core/1", "core/2", "core/0"]

    def test_recently_used_entry_survives_eviction(self):
        store = ResourceStore(max_payload_bytes=100)
        calls: list[str] = []
        for rid in ("core/0", "core/1"):
            store.put("S", _png_descriptor(rid, b"x" * 40, calls))
        store.get_bytes("S", "heatmap", "core/0")
        store.get_bytes("S", "heatmap", "core/1")
        store.get_bytes("S", "heatmap", "core/0")  # refresh recency
        store.put("S", _png_descriptor("core/2", b"x" * 40, calls))
        store.get_bytes("S", "heatmap", "core/2")  # evicts core/1, not core/0

        store.get_bytes("S", "heatmap", "core/0")
        store.get_bytes("S", "heatmap", "core/1")
        assert calls == ["core/0", "core/1", "core/2", "core/1"]

    def test_single_oversize_payload_is_still_served(self):
        store = ResourceStore(max_payload_bytes=10)
        calls: list[str] = []
        store.put("S", _png_descriptor("core/0", b"x" * 64, calls))
        hit = store.get_bytes("S", "heatmap", "core/0")
        assert hit is not None and hit[0] == b"x" * 64

    def test_json_payloads_are_exempt_from_the_byte_cap(self):
        """Connectivity JSON is small and served by reference; only rendered
        BYTES payloads count toward the cap."""
        store = ResourceStore(max_payload_bytes=10)
        calls: list[str] = []
        store.put("S", _json_descriptor("seg/0", {"spans": list(range(100))}, calls))
        store.get_json("S", "connectivity", "seg/0")
        store.get_json("S", "connectivity", "seg/0")
        assert calls == ["seg/0"]
        assert store.resident_payload_bytes() == 0

    def test_clear_step_releases_the_accounted_bytes(self):
        store = ResourceStore(max_payload_bytes=1000)
        calls: list[str] = []
        store.put("S", _png_descriptor("core/0", b"x" * 40, calls))
        store.get_bytes("S", "heatmap", "core/0")
        assert store.resident_payload_bytes() == 40
        store.clear_step("S")
        assert store.resident_payload_bytes() == 0


class TestResourceStoreThreadSafety:
    def test_concurrent_gets_invoke_producer_once(self):
        store = ResourceStore()
        barrier = threading.Barrier(8)

        call_count = {"n": 0}
        count_lock = threading.Lock()

        def slow_produce():
            with count_lock:
                call_count["n"] += 1
            return b"P"

        store.put(
            "Step A",
            ResourceDescriptor(
                kind="heatmap",
                rid="core/1",
                source=CallableSource(slow_produce),
                media_type="image/png",
            ),
        )

        results: list[bytes | None] = []

        def worker():
            barrier.wait()
            r = store.get_bytes("Step A", "heatmap", "core/1")
            results.append(r[0] if r else None)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == b"P" for r in results)
        assert call_count["n"] == 1
