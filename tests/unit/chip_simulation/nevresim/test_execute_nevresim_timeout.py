"""execute_simulator wall cap: hung binaries are killed, retried once, then fail loud."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.execution_bounds import SimulationTimeoutError
from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator


def _write_script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _assert_pid_dead(pid: int, deadline_s: float = 5.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still alive")


def test_hung_simulator_is_killed_retried_once_then_fails_loud(tmp_path):
    pids = tmp_path / "pids"
    script = _write_script(
        tmp_path / "fake_sim",
        f"echo $$ >> {pids}\nexec sleep 60\n",
    )

    t0 = time.monotonic()
    with pytest.raises(SimulationTimeoutError, match="twice"):
        execute_simulator(
            str(script), input_count=2, num_proc=1,
            expected_values=2, timeout_s=0.5,
        )
    assert time.monotonic() - t0 < 20.0

    launched = [int(p) for p in pids.read_text().split()]
    assert len(launched) == 2, "expected exactly one retry"
    for pid in launched:
        _assert_pid_dead(pid)


def test_normal_completion_results_unchanged(tmp_path):
    script = _write_script(
        tmp_path / "fake_sim",
        'i=$1\nwhile [ "$i" -lt "$2" ]; do\n  echo "1.0 2.0 3.0"\n  i=$((i+1))\ndone\n',
    )
    expected = [1.0, 2.0, 3.0] * 4

    out_default = execute_simulator(
        str(script), input_count=4, num_proc=2, expected_values=12,
    )
    assert out_default == expected

    out_capped = execute_simulator(
        str(script), input_count=4, num_proc=2, expected_values=12, timeout_s=60.0,
    )
    assert out_capped == expected


def test_worker_failure_still_raises_runtime_error(tmp_path):
    script = _write_script(
        tmp_path / "fake_sim",
        "echo boom >&2\nexit 3\n",
    )
    with pytest.raises(RuntimeError, match="exit 3"):
        execute_simulator(str(script), input_count=2, num_proc=1, timeout_s=30.0)
