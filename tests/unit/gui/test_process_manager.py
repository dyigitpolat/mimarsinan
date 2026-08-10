"""Unit tests for mimarsinan.gui.runtime.process_manager."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mimarsinan.gui.runtime.proc_identity import (
    parse_stat_starttime,
    read_proc_starttime,
)
from mimarsinan.gui.runtime.process_manager import ManagedRun, ProcessManager


class TestListActive:
    def test_list_active_empty_manager(self, tmp_path):
        """list_active returns empty list when manager has no runs."""
        manager = ProcessManager(generated_files_root=str(tmp_path))
        assert manager.list_active() == []


class TestGetRunDetail:
    def test_get_run_detail_unknown_run_id_returns_none(self, tmp_path):
        """get_run_detail returns None for unknown run_id."""
        manager = ProcessManager(generated_files_root=str(tmp_path))
        assert manager.get_run_detail("nonexistent_run_id") is None

    def test_get_run_detail_end_time_fallback_for_status(self, tmp_path):
        """Steps with end_time but no explicit status should be inferred as 'completed'."""
        from mimarsinan.gui.runtime.persistence import (
            save_run_info, save_step_to_persisted,
        )
        run_id = "test_exp_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)
        working_dir = str(run_dir)

        save_run_info(working_dir, pid=999999999, step_names=["StepA", "StepB"])

        save_step_to_persisted(
            working_dir, "StepA",
            start_time=10.0, end_time=20.0,
            target_metric=0.9, metrics=[], snapshot=None, snapshot_key_kinds=None,
        )
        save_step_to_persisted(
            working_dir, "StepB",
            start_time=20.0, end_time=None,
            target_metric=None, metrics=[], snapshot=None, snapshot_key_kinds=None,
            status="running",
        )

        manager = ProcessManager(generated_files_root=str(tmp_path))
        detail = manager.get_run_detail(run_id)
        assert detail is not None
        step_a = next(s for s in detail["steps"] if s["name"] == "StepA")
        step_b = next(s for s in detail["steps"] if s["name"] == "StepB")
        assert step_a["status"] == "completed"
        assert step_a["target_metric"] == 0.9
        assert step_b["status"] == "failed"

    def test_get_run_step_detail_end_time_fallback(self, tmp_path):
        """get_run_step_detail uses end_time fallback when status key is missing."""
        from mimarsinan.gui.runtime.persistence import (
            save_run_info, save_step_to_persisted,
        )
        run_id = "test2_exp_phased_deployment_run_20240101_130000"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)
        working_dir = str(run_dir)

        save_run_info(working_dir, pid=999999999, step_names=["StepX"])
        save_step_to_persisted(
            working_dir, "StepX",
            start_time=10.0, end_time=15.0,
            target_metric=0.88, metrics=[], snapshot=None, snapshot_key_kinds=None,
        )

        manager = ProcessManager(generated_files_root=str(tmp_path))
        sd = manager.get_run_step_detail(run_id, "StepX")
        assert sd is not None
        assert sd["status"] == "completed"
        assert sd["target_metric"] == 0.88


class TestKillRun:
    def test_kill_run_unknown_run_id_returns_false(self, tmp_path):
        """kill_run returns False for unknown run_id."""
        manager = ProcessManager(generated_files_root=str(tmp_path))
        assert manager.kill_run("nonexistent_run_id") is False


class TestRecoverOrphanedRuns:
    def test_recover_orphaned_run_with_nonexistent_pid_is_alive_false(self, tmp_path):
        """Create fake run with non-existent pid and status 'running'; verify recovered but is_alive returns False."""
        run_id = "my_exp_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id / "_GUI_STATE"
        run_dir.mkdir(parents=True)
        run_info = {
            "pid": 999999999,  # Non-existent PID
            "status": "running",
            "started_at": time.time() - 60,
            "finished_at": time.time() - 60,  # Recent so _cleanup does not remove it
            "config_summary": {"experiment_name": "my_exp"},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info))

        manager = ProcessManager(generated_files_root=str(tmp_path))

        active = manager.list_active()
        assert len(active) == 1
        assert active[0]["run_id"] == run_id
        assert active[0]["is_alive"] is False

        managed = manager._runs[run_id]
        assert managed.is_alive() is False

    def test_list_active_handles_missing_finished_at_for_dead_run(self, tmp_path):
        """Recovered dead runs with ``finished_at: null`` must not crash cleanup."""
        from mimarsinan.gui.runtime.persistence import save_run_info

        run_id = "null_finished_at_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)
        working_dir = str(run_dir)

        # ``save_run_info`` writes ``finished_at=None`` for newly started runs.
        save_run_info(working_dir, pid=999999999, step_names=["StepA"])

        manager = ProcessManager(generated_files_root=str(tmp_path))

        active = manager.list_active()
        assert len(active) == 1
        assert active[0]["run_id"] == run_id
        assert active[0]["is_alive"] is False

    def test_recover_orphaned_runs_skips_finished_over_one_hour_ago(self, tmp_path):
        """Create run_info with finished_at over 1 hour ago; verify it is NOT recovered."""
        run_id = "old_exp_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id / "_GUI_STATE"
        run_dir.mkdir(parents=True)
        run_info = {
            "pid": 12345,
            "status": "completed",
            "started_at": time.time() - 7200,
            "finished_at": time.time() - 7200,  # 2 hours ago
            "config_summary": {"experiment_name": "old_exp"},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info))

        manager = ProcessManager(generated_files_root=str(tmp_path))

        active = manager.list_active()
        assert len(active) == 0
        assert run_id not in manager._runs


class TestProcStarttime:
    """PID-reuse honesty: liveness = signal-0 probe AND kernel starttime match."""

    def test_parse_stat_starttime_field_22(self):
        # Field 22 (starttime) is the 20th token after the comm field.
        rest = "R " + " ".join(str(i) for i in range(18)) + " 4242 999"
        stat = f"123 (python) {rest}".encode()
        assert parse_stat_starttime(stat) == 4242

    def test_parse_handles_spaces_and_parens_in_comm(self):
        # comm may contain spaces AND parens; fields parse from the LAST ')'.
        rest = "S " + " ".join(str(i) for i in range(18)) + " 777 0"
        stat = f"42 (weird name) with (parens) {rest}".encode()
        assert parse_stat_starttime(stat) == 777

    def test_parse_malformed_returns_none(self):
        assert parse_stat_starttime(b"garbage without parens") is None
        assert parse_stat_starttime(b"1 (x) R 2 3") is None

    def test_read_proc_starttime_of_self_matches_stat_file(self):
        pid = os.getpid()
        value = read_proc_starttime(pid)
        assert isinstance(value, int) and value > 0
        with open(f"/proc/{pid}/stat", "rb") as f:
            assert value == parse_stat_starttime(f.read())

    def test_read_proc_starttime_missing_pid_returns_none(self):
        assert read_proc_starttime(999999999) is None

    def test_starttime_mismatch_means_not_alive(self, tmp_path):
        pid = os.getpid()
        actual = read_proc_starttime(pid)
        assert actual is not None
        reused = ManagedRun(
            run_id="r", working_dir=str(tmp_path), pid=pid,
            started_at=time.time(), starttime=actual + 1,
        )
        assert reused.is_alive() is False
        same = ManagedRun(
            run_id="r", working_dir=str(tmp_path), pid=pid,
            started_at=time.time(), starttime=actual,
        )
        assert same.is_alive() is True

    def test_missing_proc_falls_back_to_bare_probe(self, tmp_path, monkeypatch):
        from mimarsinan.gui.runtime import process_spawn

        monkeypatch.setattr(process_spawn, "read_proc_starttime", lambda pid: None)
        managed = ManagedRun(
            run_id="r", working_dir=str(tmp_path), pid=os.getpid(),
            started_at=time.time(), starttime=123456,
        )
        assert managed.is_alive() is True

    def test_save_run_info_records_starttime(self, tmp_path):
        from mimarsinan.gui.runtime.persistence import load_run_info, save_run_info

        save_run_info(str(tmp_path), pid=os.getpid(), step_names=["A"])
        info = load_run_info(str(tmp_path))
        assert info is not None
        assert info["starttime"] == read_proc_starttime(os.getpid())

    def test_recovered_run_with_reused_pid_is_not_alive(self, tmp_path):
        """A run_info pointing at a live pid whose kernel starttime differs is a
        DEAD run wearing a recycled pid; recovery must not call it alive."""
        pid = os.getpid()
        actual = read_proc_starttime(pid)
        assert actual is not None
        run_id = "reused_pid_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id / "_GUI_STATE"
        run_dir.mkdir(parents=True)
        run_info = {
            "pid": pid,
            "starttime": actual + 1,
            "status": "running",
            "started_at": time.time() - 60,
            "finished_at": None,
            "config_summary": {"experiment_name": "reused"},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info))

        manager = ProcessManager(generated_files_root=str(tmp_path))
        active = manager.list_active()
        assert len(active) == 1
        assert active[0]["is_alive"] is False
        assert manager.is_run_alive(run_id) is False

    def test_recovered_run_with_matching_starttime_is_alive(self, tmp_path):
        pid = os.getpid()
        actual = read_proc_starttime(pid)
        run_id = "same_pid_phased_deployment_run_20240101_120000"
        run_dir = tmp_path / run_id / "_GUI_STATE"
        run_dir.mkdir(parents=True)
        run_info = {
            "pid": pid,
            "starttime": actual,
            "status": "running",
            "started_at": time.time() - 60,
            "finished_at": None,
            "config_summary": {"experiment_name": "same"},
        }
        (run_dir / "run_info.json").write_text(json.dumps(run_info))

        manager = ProcessManager(generated_files_root=str(tmp_path))
        assert manager.is_run_alive(run_id) is True


class TestSpawnRun:
    @patch("mimarsinan.gui.runtime.process_spawn.subprocess.Popen")
    @patch("mimarsinan.gui.runtime.process_spawn.time.strftime")
    def test_spawn_run_appears_in_list_active_with_correct_format(
        self, mock_strftime, mock_popen, tmp_path
    ):
        """Mock Popen; verify run appears in list_active with correct run_id format and _working_directory in config."""
        mock_strftime.return_value = "20240318_143000"
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None  # Process is alive
        mock_popen.return_value = mock_proc

        gen_root = str(tmp_path / "generated")
        config = {
            "experiment_name": "my_experiment",
            "pipeline_mode": "phased",
            "generated_files_path": gen_root,
        }

        manager = ProcessManager(generated_files_root=gen_root)
        run_id = manager.spawn_run(config)

        expected_prefix = "my_experiment_phased_deployment_run_20240318_143000"
        assert run_id == expected_prefix

        active = manager.list_active()
        assert len(active) == 1
        assert active[0]["run_id"] == run_id
        assert active[0]["is_alive"] is True

        config_path = Path(gen_root) / run_id / "_RUN_CONFIG" / "config.json"
        assert config_path.exists()
        with open(config_path, encoding="utf-8") as f:
            saved_config = json.load(f)
        assert saved_config["_working_directory"] == str(Path(gen_root) / run_id)
