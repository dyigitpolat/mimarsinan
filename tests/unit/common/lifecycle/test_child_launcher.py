"""Launching a long-lived child: completion is the child's exit, never pipe EOF."""

import subprocess
import sys
import time

import pytest

from mimarsinan.common.lifecycle.child_launcher import ChildResult, run_child

# Exits immediately but leaves a detached grandchild holding the inherited
# stdout/stderr write ends -- the leaked forkserver/dataloader cohort in
# miniature. Pipe EOF cannot arrive until the grandchild dies.
_LEAKS_A_PIPE_HOLDER = [
    sys.executable, "-c",
    "import os, subprocess, sys;"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
    "print('DONE', flush=True);"
    "os._exit(0)",
]


class TestStdlibCaptureIsEofGated:
    """Control: this is the defect the launcher exists to route around."""

    def test_subprocess_run_blocks_its_whole_timeout_on_a_leaked_holder(self):
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(_LEAKS_A_PIPE_HOLDER, capture_output=True, timeout=3.0)
        assert time.monotonic() - started >= 2.5


class TestCompletionIsGatedOnChildExit:
    def test_returns_at_child_exit_even_though_a_grandchild_holds_the_pipe(self):
        started = time.monotonic()
        result = run_child(_LEAKS_A_PIPE_HOLDER, timeout_s=30.0)
        elapsed = time.monotonic() - started

        assert isinstance(result, ChildResult)
        assert not result.timed_out
        assert result.returncode == 0
        assert "DONE" in result.stdout
        assert elapsed < 15.0, (
            f"run_child waited {elapsed:.1f}s: completion is still gated on pipe "
            "EOF held by a descendant cohort, not on the child's own exit"
        )

    def test_the_leaked_cohort_is_killed_with_the_session(self):
        result = run_child(_LEAKS_A_PIPE_HOLDER, timeout_s=30.0)
        assert result.session_pid > 0
        assert result.session_gone, (
            "the launcher returned while the child's session still had live "
            "members; a leaked cohort survives every later run"
        )


class TestCapture:
    def test_stdout_and_stderr_are_captured_separately(self):
        result = run_child([
            sys.executable, "-c",
            "import sys; sys.stdout.write('out-side'); sys.stderr.write('err-side')",
        ], timeout_s=30.0)
        assert result.returncode == 0
        assert result.stdout.strip() == "out-side"
        assert result.stderr.strip() == "err-side"

    def test_child_returncode_is_reported(self):
        result = run_child([sys.executable, "-c", "raise SystemExit(3)"], timeout_s=30.0)
        assert result.returncode == 3
        assert not result.timed_out

    def test_stderr_survives_a_crash_so_failures_are_diagnosable(self):
        result = run_child([
            sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); raise SystemExit(1)",
        ], timeout_s=30.0)
        assert result.returncode == 1
        assert "boom" in result.stderr


class TestTimeout:
    def test_timeout_reports_timed_out_and_kills_the_whole_session(self):
        started = time.monotonic()
        result = run_child([
            sys.executable, "-c",
            "import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']);"
            "print('UP', flush=True);"
            "time.sleep(120)",
        ], timeout_s=2.0)
        elapsed = time.monotonic() - started

        assert result.timed_out
        assert result.returncode != 0
        assert elapsed < 20.0, f"timeout path took {elapsed:.1f}s"
        assert "UP" in result.stdout, "partial output must survive the timeout kill"
        assert result.session_gone, "the timed-out child's cohort was left running"

    def test_wall_seconds_are_measured(self):
        result = run_child([sys.executable, "-c", "import time; time.sleep(0.3)"],
                           timeout_s=30.0)
        assert result.wall_s >= 0.3
