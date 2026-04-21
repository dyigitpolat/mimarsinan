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
    write_persisted_steps_replace,
    save_resource_to_disk,
    load_resource_from_disk,
)
from mimarsinan.gui.persistence import _sanitize_path_segment, _resource_disk_path


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

    def test_save_step_running_then_completed(self, tmp_path):
        """Simulate on_step_start writing status='running', then on_step_end writing status='completed'."""
        working_dir = str(tmp_path)
        save_step_to_persisted(
            working_dir,
            step_name="Train",
            start_time=10.0,
            end_time=None,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
            status="running",
        )
        steps = load_persisted_steps(working_dir)
        assert steps["Train"]["status"] == "running"

        save_step_to_persisted(
            working_dir,
            step_name="Train",
            start_time=10.0,
            end_time=20.0,
            target_metric=0.95,
            metrics=[{"name": "acc", "value": 0.95}],
            snapshot={"layers": 4},
            snapshot_key_kinds={"layers": "int"},
            status="completed",
        )
        steps = load_persisted_steps(working_dir)
        assert steps["Train"]["status"] == "completed"
        assert steps["Train"]["end_time"] == 20.0
        assert steps["Train"]["target_metric"] == 0.95

    def test_load_persisted_steps_missing_returns_empty(self, tmp_path):
        working_dir = str(tmp_path)
        steps = load_persisted_steps(working_dir)
        assert steps == {}


class TestResourceDiskPath:
    """Regression: step names with spaces (``"Hard Core Mapping"``) must
    round-trip through the sanitizer. Previously the sanitizer rejected
    anything outside ``[A-Za-z0-9._-]``, which crashed the snapshot
    executor mid-run."""

    def test_sanitize_allows_spaces_in_step_name(self):
        assert _sanitize_path_segment("Hard Core Mapping") == "Hard Core Mapping"
        assert _sanitize_path_segment("Soft Core Mapping") == "Soft Core Mapping"

    def test_sanitize_allows_parentheses_and_punctuation(self):
        # Real-world step names occasionally contain these characters.
        assert _sanitize_path_segment("Model Building (1)") == "Model Building (1)"
        assert _sanitize_path_segment("IR+Graph,step") == "IR+Graph,step"

    def test_sanitize_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            _sanitize_path_segment("..")
        with pytest.raises(ValueError):
            _sanitize_path_segment(".")
        with pytest.raises(ValueError):
            _sanitize_path_segment("")
        with pytest.raises(ValueError):
            _sanitize_path_segment("a/b")
        with pytest.raises(ValueError):
            _sanitize_path_segment("a\\b")
        with pytest.raises(ValueError):
            _sanitize_path_segment("a\x00b")

    def test_sanitize_replaces_unsafe_chars_with_underscore(self):
        # Non-path-separator but otherwise unsafe characters are normalised
        # rather than rejected, so unknown future step names do not crash
        # the snapshot executor.
        assert _sanitize_path_segment("weird:name") == "weird_name"
        assert _sanitize_path_segment("tab\there") == "tab_here"

    def test_sanitize_is_idempotent(self):
        """Crucial: disk writes (sanitized) and URL-path reads (sanitized
        again by the server) must agree, so the mapping must be stable."""
        for raw in ["Hard Core Mapping", "Model Building", "weird:name", "tab\there"]:
            once = _sanitize_path_segment(raw)
            twice = _sanitize_path_segment(once)
            assert once == twice, raw

    def test_resource_disk_path_with_spaces(self, tmp_path):
        path = _resource_disk_path(
            str(tmp_path),
            step_name="Hard Core Mapping",
            kind="connectivity",
            rid="seg/2",
            media_type="application/json",
        )
        assert "Hard Core Mapping" in str(path)
        assert path.suffix == ".json"

    def test_save_and_load_resource_roundtrip_with_spaces(self, tmp_path):
        """End-to-end: saving under a space-containing step name works and
        the same name loads back via the lookup path."""
        payload = b"\x89PNG\r\n\x1a\nfake"
        save_resource_to_disk(
            str(tmp_path),
            step_name="Hard Core Mapping",
            kind="ir_core_heatmap",
            rid="core/0",
            payload=payload,
            media_type="image/png",
        )
        loaded = load_resource_from_disk(
            str(tmp_path),
            step_name="Hard Core Mapping",
            kind="ir_core_heatmap",
            rid="core/0",
            media_type="image/png",
        )
        assert loaded == payload


class TestSaveStepToPersistedThreadSafety:
    """Regression: ``save_step_to_persisted`` does a read-modify-write on
    ``steps.json``. The pipeline thread (``on_step_start``) and the
    snapshot executor thread (``_finalize``) both call it, so the
    function **must** serialise writes or else one call's update gets
    overwritten by the other's stale view. Symptom: fast steps like
    ``Model Configuration`` / ``Model Building`` intermittently stayed
    in ``running`` status forever because their ``completed`` write was
    clobbered by a concurrent ``running`` write for the next step.
    """

    def test_concurrent_writes_preserve_every_step_entry(self, tmp_path):
        import threading
        working_dir = str(tmp_path)
        step_names = [f"Step{i}" for i in range(40)]

        def writer(name: str, status: str) -> None:
            save_step_to_persisted(
                working_dir,
                step_name=name,
                start_time=1.0,
                end_time=2.0 if status == "completed" else None,
                target_metric=0.5 if status == "completed" else None,
                metrics=[],
                snapshot=None,
                snapshot_key_kinds=None,
                status=status,
            )

        threads = []
        for name in step_names:
            threads.append(threading.Thread(target=writer, args=(name, "running")))
            threads.append(threading.Thread(target=writer, args=(name, "completed")))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = load_persisted_steps(working_dir)
        for name in step_names:
            assert name in loaded, f"step {name} lost during concurrent writes"

    def test_interleaved_running_then_completed_is_preserved(self, tmp_path):
        """Simulate the exact scenario from the bug report: running-then-completed
        for step A while step B's running write happens concurrently. All three
        writes must be reflected in the final on-disk state."""
        import threading

        working_dir = str(tmp_path)
        barrier = threading.Barrier(3)

        def running(name: str) -> None:
            barrier.wait()
            save_step_to_persisted(
                working_dir, step_name=name,
                start_time=1.0, end_time=None, target_metric=None,
                metrics=[], snapshot=None, snapshot_key_kinds=None,
                status="running",
            )

        def completed(name: str) -> None:
            barrier.wait()
            save_step_to_persisted(
                working_dir, step_name=name,
                start_time=1.0, end_time=2.0, target_metric=0.9,
                metrics=[], snapshot=None, snapshot_key_kinds=None,
                status="completed",
            )

        # Many repetitions to smoke out the race without the lock.
        for _ in range(20):
            # Seed the file so there is something to read-modify-write against.
            save_step_to_persisted(
                working_dir, step_name="Seed",
                start_time=0.0, end_time=0.0, target_metric=None,
                metrics=[], snapshot=None, snapshot_key_kinds=None,
                status="completed",
            )
            barrier.reset()
            threads = [
                threading.Thread(target=running, args=("A",)),
                threading.Thread(target=completed, args=("A",)),
                threading.Thread(target=running, args=("B",)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            loaded = load_persisted_steps(working_dir)
            # Regardless of which order wins, *every* name must be present.
            assert "A" in loaded
            assert "B" in loaded
            assert "Seed" in loaded


class TestWritePersistedStepsReplace:
    def test_replace_overwrites_entire_steps_dict(self, tmp_path):
        working_dir = str(tmp_path)
        save_step_to_persisted(
            working_dir,
            step_name="OldStep",
            start_time=1.0,
            end_time=2.0,
            target_metric=None,
            metrics=[],
            snapshot=None,
            snapshot_key_kinds=None,
        )
        write_persisted_steps_replace(
            working_dir,
            {"OnlyStep": {"start_time": 3.0, "end_time": 4.0, "target_metric": 0.9, "metrics": [], "snapshot": {}, "snapshot_key_kinds": {}, "status": "completed"}},
        )
        steps = load_persisted_steps(working_dir)
        assert list(steps.keys()) == ["OnlyStep"]
        assert "OldStep" not in steps
