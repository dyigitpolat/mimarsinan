"""HTTP contract tests for the lazy resource endpoints + ETag/304 on
``/api/steps/{step}``.

These tests pin the observable behavior the frontend depends on:

* ``/api/steps/{step}`` carries an ``ETag`` header. Re-requesting with a
  matching ``If-None-Match`` returns ``304 Not Modified`` with no body.
* ``?since_seq=N`` prunes the ``metrics`` array to ``seq > N`` and still
  reports ``latest_metric_seq`` so the client advances its cursor.
* ``/api/steps/{step}/resources/{kind}/{rid}`` serves bytes from the
  in-memory :class:`ResourceStore` with the correct ``Content-Type``.
* The ``/api/runs/{run_id}/steps/.../resources/...`` mirror serves
  resources from disk (the snapshot executor persists them to
  ``_GUI_STATE/resources/``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mimarsinan.gui.data_collector import DataCollector
from mimarsinan.gui.persistence import save_resource_to_disk
from mimarsinan.gui.resources import ResourceDescriptor, ResourceStore
from mimarsinan.gui.server import create_app


@pytest.fixture
def collector_with_store() -> DataCollector:
    c = DataCollector()
    c.set_pipeline_info(["s1"], {})
    c.set_resource_store(ResourceStore())
    return c


@pytest.fixture
def client(collector_with_store: DataCollector) -> TestClient:
    app = create_app(collector_with_store)
    return TestClient(app)


def _png_bytes() -> bytes:
    # Minimal valid PNG signature + IEND chunk.
    return b"\x89PNG\r\n\x1a\nfakepngdata"


class TestStepDetailEtag:
    def test_first_request_returns_etag_header(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        collector_with_store.step_completed("s1", snapshot={"x": 1})
        r = client.get("/api/steps/s1")
        assert r.status_code == 200
        assert r.headers.get("etag"), "expected ETag header on step detail"
        body = r.json()
        assert body["snapshot_etag"] == r.headers["etag"]

    def test_if_none_match_returns_304(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        collector_with_store.step_completed("s1", snapshot={"x": 1})
        r1 = client.get("/api/steps/s1")
        etag = r1.headers["etag"]
        r2 = client.get("/api/steps/s1", headers={"If-None-Match": etag})
        assert r2.status_code == 304
        assert r2.content == b""
        assert r2.headers.get("etag") == etag

    def test_etag_changes_on_rerun(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        collector_with_store.step_completed("s1", snapshot={"x": 1})
        etag1 = client.get("/api/steps/s1").headers["etag"]
        collector_with_store.step_started("s1")
        collector_with_store.step_completed("s1", snapshot={"x": 2})
        etag2 = client.get("/api/steps/s1").headers["etag"]
        assert etag1 != etag2

    def test_404_when_step_unknown(self, client: TestClient) -> None:
        r = client.get("/api/steps/does_not_exist")
        assert r.status_code == 404


class TestStepDetailSinceSeq:
    def test_since_seq_zero_returns_all_metrics(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        collector_with_store.step_started("s1")
        collector_with_store.record_metric("loss", 0.1)
        collector_with_store.record_metric("loss", 0.2)
        body = client.get("/api/steps/s1?since_seq=0").json()
        assert len(body["metrics"]) == 2
        assert body["latest_metric_seq"] == body["metrics"][-1]["seq"]

    def test_since_seq_nonzero_filters(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        collector_with_store.step_started("s1")
        for v in [0.1, 0.2, 0.3]:
            collector_with_store.record_metric("loss", v)
        full = client.get("/api/steps/s1").json()
        checkpoint = full["metrics"][0]["seq"]
        filtered = client.get(f"/api/steps/s1?since_seq={checkpoint}").json()
        assert len(filtered["metrics"]) == 2
        assert all(m["seq"] > checkpoint for m in filtered["metrics"])
        assert filtered["latest_metric_seq"] == full["latest_metric_seq"]


class TestResourceEndpoints:
    def test_serves_png_from_store(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        store = collector_with_store.get_resource_store()
        assert store is not None
        desc = ResourceDescriptor(
            kind="ir_core_heatmap",
            rid="core/0",
            producer=_png_bytes,
            media_type="image/png",
        )
        store.put("s1", desc)
        r = client.get("/api/steps/s1/resources/ir_core_heatmap/core/0")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == _png_bytes()
        assert "cache-control" in r.headers
        assert "max-age" in r.headers["cache-control"]

    def test_serves_json_connectivity(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        store = collector_with_store.get_resource_store()
        assert store is not None
        payload = [{"core_id": 0, "dst": 1}]
        desc = ResourceDescriptor(
            kind="connectivity",
            rid="seg/0",
            producer=lambda: payload,
            media_type="application/json",
        )
        store.put("s1", desc)
        r = client.get("/api/steps/s1/resources/connectivity/seg/0")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == payload

    def test_404_on_missing_resource(self, client: TestClient) -> None:
        r = client.get("/api/steps/s1/resources/ir_core_heatmap/core/999")
        assert r.status_code == 404

    def test_404_on_unknown_kind(self, client: TestClient) -> None:
        r = client.get("/api/steps/s1/resources/not_a_kind/whatever")
        assert r.status_code == 404

    def test_producer_invoked_lazily_only_on_fetch(
        self, collector_with_store: DataCollector, client: TestClient
    ) -> None:
        store = collector_with_store.get_resource_store()
        calls: list[int] = []

        def prod() -> bytes:
            calls.append(1)
            return _png_bytes()

        desc = ResourceDescriptor(
            kind="ir_core_heatmap",
            rid="core/7",
            producer=prod,
            media_type="image/png",
        )
        store.put("s1", desc)
        assert calls == []  # put must not invoke producer
        client.get("/api/steps/s1/resources/ir_core_heatmap/core/7")
        client.get("/api/steps/s1/resources/ir_core_heatmap/core/7")
        assert calls == [1]  # cached after first fetch


class TestHistoricalRunResources:
    def test_serves_png_from_disk(self, tmp_path: Path, monkeypatch) -> None:
        # Stand up a minimal "historical run" layout.
        runs_root = tmp_path / "generated"
        run_id = "hist_run"
        run_dir = runs_root / run_id
        (run_dir / "_RUN_CONFIG").mkdir(parents=True)
        # get_runs_root is read from env.
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(runs_root))

        save_resource_to_disk(
            str(run_dir), "s1", "ir_core_heatmap", "core/0",
            _png_bytes(), media_type="image/png",
        )

        c = DataCollector()
        app = create_app(c)
        client = TestClient(app)

        r = client.get(f"/api/runs/{run_id}/steps/s1/resources/ir_core_heatmap/core/0")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == _png_bytes()

    def test_404_when_resource_not_on_disk(self, tmp_path: Path, monkeypatch) -> None:
        runs_root = tmp_path / "generated"
        run_dir = runs_root / "hist_run"
        (run_dir / "_RUN_CONFIG").mkdir(parents=True)
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(runs_root))

        c = DataCollector()
        app = create_app(c)
        client = TestClient(app)
        r = client.get("/api/runs/hist_run/steps/s1/resources/ir_core_heatmap/core/0")
        assert r.status_code == 404

    def test_rejects_invalid_run_id(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        c = DataCollector()
        app = create_app(c)
        client = TestClient(app)
        # Path traversal should be rejected before touching the filesystem.
        r = client.get("/api/runs/..%2Fetc/steps/s1/resources/ir_core_heatmap/core/0")
        assert r.status_code in (400, 404)
