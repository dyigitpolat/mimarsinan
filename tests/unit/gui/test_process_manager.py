"""Unit tests for mimarsinan.gui.process_manager."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mimarsinan.gui.process_manager import ManagedRun, ProcessManager


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


class TestSpawnRun:
    @patch("mimarsinan.gui.process_manager.subprocess.Popen")
    @patch("mimarsinan.gui.process_manager.time.strftime")
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
