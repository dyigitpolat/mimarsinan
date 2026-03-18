"""Unit tests for mimarsinan.gui.persistence."""

import pytest

from mimarsinan.gui.persistence import (
    save_run_info,
    load_run_info,
    update_run_status,
    append_live_metric,
    load_live_metrics,
    save_step_to_persisted,
    load_persisted_steps,
)


class TestSaveLoadRunInfo:
    def test_save_and_load_run_info(self, tmp_path):
        working_dir = str(tmp_path)
        save_run_info(
            working_dir,
            pid=12345,
            step_names=["Step A", "Step B"],
            config_summary={"model": "mlp_mixer"},
        )
        loaded = load_run_info(working_dir)
        assert loaded is not None
        assert loaded["pid"] == 12345
        assert loaded["step_names"] == ["Step A", "Step B"]
        assert loaded["status"] == "running"
        assert loaded["config_summary"] == {"model": "mlp_mixer"}
        assert loaded["started_at"] is not None
        assert loaded["finished_at"] is None
        assert loaded["error"] is None

    def test_load_run_info_missing_returns_none(self, tmp_path):
        working_dir = str(tmp_path)
        loaded = load_run_info(working_dir)
        assert loaded is None


class TestUpdateRunStatus:
    def test_update_run_status(self, tmp_path):
        working_dir = str(tmp_path)
        save_run_info(working_dir, pid=999, step_names=["X"])
        update_run_status(working_dir, "completed")
        loaded = load_run_info(working_dir)
        assert loaded is not None
        assert loaded["status"] == "completed"
        assert loaded["finished_at"] is not None

    def test_update_run_status_with_error(self, tmp_path):
        working_dir = str(tmp_path)
        save_run_info(working_dir, pid=999, step_names=["X"])
        update_run_status(working_dir, "failed", error="Something went wrong")
        loaded = load_run_info(working_dir)
        assert loaded is not None
        assert loaded["status"] == "failed"
        assert loaded["error"] == "Something went wrong"


class TestAppendLoadLiveMetrics:
    def test_append_and_load_live_metrics(self, tmp_path):
        working_dir = str(tmp_path)
        append_live_metric(working_dir, "step1", "accuracy", 0.95, seq=1, timestamp=100.0)
        append_live_metric(working_dir, "step1", "loss", 0.1, seq=2, timestamp=101.0)
        append_live_metric(working_dir, "step2", "accuracy", 0.92, seq=1, timestamp=102.0)
        all_metrics = load_live_metrics(working_dir)
        assert len(all_metrics) == 3
        assert all_metrics[0] == {
            "step": "step1",
            "name": "accuracy",
            "value": 0.95,
            "seq": 1,
            "timestamp": 100.0,
        }
        assert all_metrics[1]["name"] == "loss"
        assert all_metrics[2]["step"] == "step2"

    def test_load_live_metrics_filter_by_step(self, tmp_path):
        working_dir = str(tmp_path)
        append_live_metric(working_dir, "step_a", "m1", 1.0, seq=1, timestamp=1.0)
        append_live_metric(working_dir, "step_b", "m2", 2.0, seq=1, timestamp=2.0)
        append_live_metric(working_dir, "step_a", "m3", 3.0, seq=2, timestamp=3.0)
        step_a_only = load_live_metrics(working_dir, step_name="step_a")
        assert len(step_a_only) == 2
        assert all(r["step"] == "step_a" for r in step_a_only)
        assert step_a_only[0]["name"] == "m1"
        assert step_a_only[1]["name"] == "m3"

    def test_load_live_metrics_missing_returns_empty(self, tmp_path):
        working_dir = str(tmp_path)
        metrics = load_live_metrics(working_dir)
        assert metrics == []


class TestSaveLoadPersistedSteps:
    def test_save_and_load_persisted_steps(self, tmp_path):
        working_dir = str(tmp_path)
        save_step_to_persisted(
            working_dir,
            step_name="Model Building",
            start_time=100.0,
            end_time=105.0,
            target_metric=0.98,
            metrics=[{"name": "acc", "value": 0.95}],
            snapshot={"layers": 3},
            snapshot_key_kinds={"layers": "int"},
        )
        steps = load_persisted_steps(working_dir)
        assert "Model Building" in steps
        entry = steps["Model Building"]
        assert entry["start_time"] == 100.0
        assert entry["end_time"] == 105.0
        assert entry["target_metric"] == 0.98
        assert entry["metrics"] == [{"name": "acc", "value": 0.95}]
        assert entry["snapshot"] == {"layers": 3}
        assert entry["snapshot_key_kinds"] == {"layers": "int"}

    def test_save_step_with_status(self, tmp_path):
        working_dir = str(tmp_path)
        save_step_to_persisted(
            working_dir,
            step_name="Pretraining",
            start_time=1.0,
            end_time=2.0,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
            status="completed",
        )
        steps = load_persisted_steps(working_dir)
        assert steps["Pretraining"]["status"] == "completed"

    def test_save_step_merges_into_existing(self, tmp_path):
        working_dir = str(tmp_path)
        save_step_to_persisted(
            working_dir,
            step_name="Step1",
            start_time=1.0,
            end_time=2.0,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        save_step_to_persisted(
            working_dir,
            step_name="Step2",
            start_time=3.0,
            end_time=4.0,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        steps = load_persisted_steps(working_dir)
        assert len(steps) == 2
        assert steps["Step1"]["start_time"] == 1.0
        assert steps["Step2"]["start_time"] == 3.0

    def test_load_persisted_steps_missing_returns_empty(self, tmp_path):
        working_dir = str(tmp_path)
        steps = load_persisted_steps(working_dir)
        assert steps == {}
