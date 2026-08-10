"""Unit tests for mimarsinan.gui.runs."""

import json

import pytest

from mimarsinan.gui.runtime.persistence import (
    save_run_info,
    save_step_status,
    save_step_to_persisted,
    update_run_status,
)
from mimarsinan.gui.runs import (
    get_runs_root,
    list_runs,
    get_run_config,
    get_run_pipeline,
    get_run_step_detail,
)


class TestGetRunsRoot:
    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "/custom/runs")
        assert get_runs_root() == "/custom/runs"

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_RUNS_ROOT", raising=False)
        assert get_runs_root() == "./generated"


class TestListRuns:
    def test_empty_dir_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        assert list_runs() == []

    def test_valid_runs_returned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        (run_a / "_RUN_CONFIG").mkdir()
        (run_a / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps({"experiment_name": "Exp A", "pipeline_mode": "phased"}),
            encoding="utf-8",
        )
        run_b = tmp_path / "run_b"
        run_b.mkdir()
        (run_b / "_RUN_CONFIG").mkdir()
        (run_b / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps({"experiment_name": "Exp B", "pipeline_mode": "vanilla"}),
            encoding="utf-8",
        )
        results = list_runs()
        assert len(results) == 2
        run_ids = {r["run_id"] for r in results}
        assert run_ids == {"run_a", "run_b"}
        for r in results:
            assert "experiment_name" in r
            assert "pipeline_mode" in r
            assert "created_at" in r

    def test_list_runs_with_include_steps(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = tmp_path / "run_with_steps"
        run_dir.mkdir()
        (run_dir / "_RUN_CONFIG").mkdir()
        (run_dir / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps({"experiment_name": "With Steps", "pipeline_mode": "phased"}),
            encoding="utf-8",
        )
        save_step_to_persisted(
            str(run_dir),
            step_name="Step1",
            start_time=1.0,
            end_time=2.0,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        save_step_to_persisted(
            str(run_dir),
            step_name="Step2",
            start_time=3.0,
            end_time=None,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        results = list_runs(include_steps=True)
        assert len(results) == 1
        entry = results[0]
        assert entry["run_id"] == "run_with_steps"
        assert set(entry["steps"]) == {"Step1", "Step2"}
        assert entry["total_steps"] == 2
        assert entry["completed_steps"] == 1

    def test_skips_dirs_without_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        (tmp_path / "no_config").mkdir()
        (tmp_path / "has_config").mkdir()
        (tmp_path / "has_config" / "_RUN_CONFIG").mkdir()
        (tmp_path / "has_config" / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps({"experiment_name": "OK"}),
            encoding="utf-8",
        )
        results = list_runs()
        assert len(results) == 1
        assert results[0]["run_id"] == "has_config"


class TestGetRunConfig:
    def test_existing_run_returns_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        (run_dir / "_RUN_CONFIG").mkdir()
        config = {"experiment_name": "Test", "pipeline_mode": "phased", "seed": 42}
        (run_dir / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        assert get_run_config("my_run") == config

    def test_missing_run_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        assert get_run_config("nonexistent_run") is None


class TestGetRunPipeline:
    def test_existing_run_returns_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = tmp_path / "pipeline_run"
        run_dir.mkdir()
        (run_dir / "_RUN_CONFIG").mkdir()
        (run_dir / "_RUN_CONFIG" / "config.json").write_text(
            json.dumps({"experiment_name": "Pipeline Test"}),
            encoding="utf-8",
        )
        save_step_to_persisted(
            str(run_dir),
            step_name="Build",
            start_time=10.0,
            end_time=15.0,
            target_metric=0.95,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        result = get_run_pipeline("pipeline_run")
        assert result is not None
        assert "steps" in result
        assert "config" in result
        assert result["current_step"] is None
        assert len(result["steps"]) == 1
        step = result["steps"][0]
        assert step["name"] == "Build"
        assert step["status"] == "completed"
        assert step["start_time"] == 10.0
        assert step["end_time"] == 15.0
        assert step["duration"] == 5.0
        assert step["target_metric"] == 0.95
        assert "config_view" in result
        assert result["config_view"]["summary"]["experiment_name"] == "Pipeline Test"


def _make_run_dir(root, run_id, config=None):
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "_RUN_CONFIG").mkdir()
    (run_dir / "_RUN_CONFIG" / "config.json").write_text(
        json.dumps(config or {"experiment_name": run_id}), encoding="utf-8",
    )
    return run_dir


class TestFailureStates:
    """A dead run's persisted failure state must reach the historical payload."""

    def test_failed_step_and_run_info_surface_in_pipeline_payload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = _make_run_dir(tmp_path, "crashed_run")
        wd = str(run_dir)
        save_step_to_persisted(
            wd, step_name="Build", start_time=1.0, end_time=2.0,
            target_metric=0.9, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="completed",
        )
        save_step_to_persisted(
            wd, step_name="Train", start_time=2.0, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="running",
        )
        save_step_status(wd, "Train", status="failed", end_time=3.0, error="boom")
        save_run_info(wd, pid=999999999, step_names=["Build", "Train"])
        update_run_status(wd, "failed", error="boom")

        result = get_run_pipeline("crashed_run")
        assert result is not None
        by_name = {s["name"]: s for s in result["steps"]}
        assert by_name["Build"]["status"] == "completed"
        assert by_name["Train"]["status"] == "failed"
        assert by_name["Train"]["error"] == "boom"
        assert result["status"] == "failed"
        assert result["error"] == "boom"
        assert result["is_alive"] is False
        # The payload stays additive: every pre-existing key is still served.
        for key in ("steps", "current_step", "config", "config_view", "overview_chart"):
            assert key in result

    def test_legacy_dir_without_status_still_derives_completed_pending(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = _make_run_dir(tmp_path, "legacy_run")
        wd = str(run_dir)
        save_step_to_persisted(
            wd, step_name="Done", start_time=1.0, end_time=2.0,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
        )
        save_step_to_persisted(
            wd, step_name="NeverRan", start_time=None, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
        )
        result = get_run_pipeline("legacy_run")
        assert result is not None
        by_name = {s["name"]: s for s in result["steps"]}
        assert by_name["Done"]["status"] == "completed"
        assert by_name["NeverRan"]["status"] == "pending"
        assert result["is_alive"] is False
        assert result["error"] is None

    def test_stale_running_step_in_dead_run_reads_failed(self, tmp_path, monkeypatch):
        """SIGKILL leaves status='running' behind; a historical run is dead by
        definition, so the payload must not present the step as live."""
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = _make_run_dir(tmp_path, "sigkilled_run")
        wd = str(run_dir)
        save_step_to_persisted(
            wd, step_name="Train", start_time=1.0, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="running",
        )
        save_run_info(wd, pid=999999999, step_names=["Train"])

        result = get_run_pipeline("sigkilled_run")
        assert result is not None
        assert result["steps"][0]["status"] == "failed"
        assert result["status"] == "failed"
        assert result["is_alive"] is False

    def test_step_detail_honors_persisted_failed_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = _make_run_dir(tmp_path, "detail_run")
        wd = str(run_dir)
        save_step_to_persisted(
            wd, step_name="Train", start_time=1.0, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="running",
        )
        save_step_status(wd, "Train", status="failed", end_time=2.0, error="boom")
        detail = get_run_step_detail("detail_run", "Train")
        assert detail is not None
        assert detail["status"] == "failed"
        assert detail["error"] == "boom"

    def test_list_runs_does_not_count_failed_steps_as_completed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
        run_dir = _make_run_dir(tmp_path, "mixed_run")
        wd = str(run_dir)
        save_step_to_persisted(
            wd, step_name="Build", start_time=1.0, end_time=2.0,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="completed",
        )
        save_step_to_persisted(
            wd, step_name="Train", start_time=2.0, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="running",
        )
        save_step_status(wd, "Train", status="failed", end_time=3.0, error="boom")
        entry = next(r for r in list_runs(include_steps=True) if r["run_id"] == "mixed_run")
        assert entry["total_steps"] == 2
        assert entry["completed_steps"] == 1


class TestInvalidRunId:
    def test_path_traversal_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "/some/root")
        with pytest.raises(ValueError, match="Invalid run_id"):
            get_run_config("../foo")

    def test_slash_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "/some/root")
        with pytest.raises(ValueError, match="Invalid run_id"):
            get_run_config("foo/bar")

    def test_dot_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", "/some/root")
        with pytest.raises(ValueError, match="Invalid run_id"):
            get_run_config(".")
