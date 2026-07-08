"""Artifacts section: run-directory listing + guarded file downloads."""

import json

import pytest
from fastapi.testclient import TestClient

from mimarsinan.gui.runs import (
    classify_artifact,
    list_dir_artifacts,
    resolve_artifact_file,
)


def _make_run_dir(tmp_path, name="myrun_deployment_run"):
    run = tmp_path / name
    run.mkdir()
    (run / "_RUN_CONFIG").mkdir()
    (run / "_RUN_CONFIG" / "config.json").write_text(
        json.dumps({"experiment_name": "myrun", "pipeline_mode": "phased"})
    )
    (run / "_GUI_STATE").mkdir()
    (run / "_GUI_STATE" / "steps.json").write_text("{}")
    (run / "Pretraining.model.pt").write_bytes(b"x" * 64)
    (run / "Model Configuration.model_config.json").write_text("{}")
    (run / "cost_record.json").write_text("{}")
    seg = run / "segment_0"
    (seg / "weights").mkdir(parents=True)
    (seg / "weights" / "w.txt").write_bytes(b"y" * 10)
    return run


class TestListDirArtifacts:
    def test_lists_top_level_entries_with_size_and_kind(self, tmp_path):
        run = _make_run_dir(tmp_path)
        entries = list_dir_artifacts(str(run))
        by_path = {e["path"]: e for e in entries}
        assert by_path["Pretraining.model.pt"]["kind"] == "torch"
        assert by_path["Pretraining.model.pt"]["size"] == 64
        assert by_path["cost_record.json"]["kind"] == "json"
        seg = by_path["segment_0"]
        assert seg["kind"] == "dir"
        assert seg["files"] == 1
        assert seg["size"] == 10

    def test_step_cache_files_carry_their_step_name(self, tmp_path):
        run = _make_run_dir(tmp_path)
        entries = {e["path"]: e for e in list_dir_artifacts(str(run))}
        assert entries["Pretraining.model.pt"]["group"] == "step cache"
        assert entries["Pretraining.model.pt"]["step"] == "Pretraining"
        assert entries["Model Configuration.model_config.json"]["step"] == "Model Configuration"
        assert entries["_GUI_STATE"]["group"] == "monitor state"
        assert entries["_RUN_CONFIG"]["group"] == "config"
        assert entries["segment_0"]["group"] == "segments"
        assert entries["cost_record.json"]["group"] == "run outputs"

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_dir_artifacts(str(tmp_path / "nope")) == []


class TestClassifyArtifact:
    @pytest.mark.parametrize("name,kind", [
        ("a.pt", "torch"), ("a.pickle", "pickle"), ("a.json", "json"),
        ("a.jsonl", "jsonl"), ("a.png", "image"), ("a.txt", "text"),
        ("a.bin", "other"),
    ])
    def test_kind_by_suffix(self, name, kind):
        assert classify_artifact(name)["kind"] == kind


class TestResolveArtifactFile:
    def test_resolves_a_real_file(self, tmp_path):
        run = _make_run_dir(tmp_path)
        p = resolve_artifact_file(str(run), "cost_record.json")
        assert p is not None and p.name == "cost_record.json"

    def test_rejects_traversal(self, tmp_path):
        run = _make_run_dir(tmp_path)
        (tmp_path / "secret.txt").write_text("no")
        assert resolve_artifact_file(str(run), "../secret.txt") is None
        assert resolve_artifact_file(str(run), "/etc/passwd") is None

    def test_rejects_directories_and_missing(self, tmp_path):
        run = _make_run_dir(tmp_path)
        assert resolve_artifact_file(str(run), "segment_0") is None
        assert resolve_artifact_file(str(run), "ghost.json") is None


@pytest.fixture()
def client_with_run(tmp_path, monkeypatch):
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.server.app import create_app

    run = _make_run_dir(tmp_path)
    monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
    collector = DataCollector()
    collector.set_working_directory(str(run))
    app = create_app(collector)
    return TestClient(app), run


class TestArtifactRoutes:
    def test_live_listing_uses_the_collector_working_dir(self, client_with_run):
        client, _run = client_with_run
        entries = client.get("/api/artifacts").json()
        assert any(e["path"] == "Pretraining.model.pt" for e in entries)

    def test_historical_listing(self, client_with_run):
        client, run = client_with_run
        entries = client.get(f"/api/runs/{run.name}/artifacts").json()
        assert any(e["path"] == "cost_record.json" for e in entries)

    def test_historical_listing_404s_for_unknown_run(self, client_with_run):
        client, _run = client_with_run
        response = client.get("/api/runs/not_a_run/artifacts")
        assert response.status_code == 404

    def test_download_serves_the_file(self, client_with_run):
        client, run = client_with_run
        response = client.get(
            f"/api/runs/{run.name}/artifact_file",
            params={"path": "cost_record.json"},
        )
        assert response.status_code == 200
        assert response.content == b"{}"

    def test_download_rejects_traversal(self, client_with_run):
        client, run = client_with_run
        response = client.get(
            f"/api/runs/{run.name}/artifact_file",
            params={"path": "../secret.txt"},
        )
        assert response.status_code == 404

    def test_live_download(self, client_with_run):
        client, _run = client_with_run
        response = client.get("/api/artifact_file", params={"path": "cost_record.json"})
        assert response.status_code == 200
