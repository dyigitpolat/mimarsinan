"""Wall-clock bounds for external-simulator invocations (execution_bounds SSOT)."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.execution_bounds import (
    DEFAULT_SIMULATION_STEP_TIMEOUT_S,
    SimulationTimeoutError,
    kill_process_group,
    resolve_simulation_step_timeout_s,
    retry_once_on_timeout,
    run_bounded,
    run_tasks_in_pool_bounded,
)
from mimarsinan.common.env import SIMULATION_STEP_TIMEOUT_VAR

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


# ---------------------------------------------------------------------------
# resolve_simulation_step_timeout_s
# ---------------------------------------------------------------------------


class TestResolver:
    def test_default_is_900s(self, monkeypatch):
        monkeypatch.delenv(SIMULATION_STEP_TIMEOUT_VAR, raising=False)
        assert resolve_simulation_step_timeout_s() == DEFAULT_SIMULATION_STEP_TIMEOUT_S
        assert DEFAULT_SIMULATION_STEP_TIMEOUT_S == 900.0

    def test_config_value_wins_over_default(self, monkeypatch):
        monkeypatch.delenv(SIMULATION_STEP_TIMEOUT_VAR, raising=False)
        assert resolve_simulation_step_timeout_s(120) == 120.0
        assert resolve_simulation_step_timeout_s(1.5) == 1.5

    def test_env_override_wins_over_config_and_default(self, monkeypatch):
        monkeypatch.setenv(SIMULATION_STEP_TIMEOUT_VAR, "123.5")
        assert resolve_simulation_step_timeout_s() == 123.5
        assert resolve_simulation_step_timeout_s(50.0) == 123.5

    def test_empty_env_is_unset(self, monkeypatch):
        monkeypatch.setenv(SIMULATION_STEP_TIMEOUT_VAR, "  ")
        assert resolve_simulation_step_timeout_s(75.0) == 75.0

    def test_nonpositive_values_fail_loud(self, monkeypatch):
        monkeypatch.delenv(SIMULATION_STEP_TIMEOUT_VAR, raising=False)
        with pytest.raises(ValueError, match="simulation_step_timeout_s"):
            resolve_simulation_step_timeout_s(0)
        with pytest.raises(ValueError, match="simulation_step_timeout_s"):
            resolve_simulation_step_timeout_s(-5.0)
        monkeypatch.setenv(SIMULATION_STEP_TIMEOUT_VAR, "0")
        with pytest.raises(ValueError, match="simulation_step_timeout_s"):
            resolve_simulation_step_timeout_s(30.0)

    def test_resolution_is_idempotent(self, monkeypatch):
        monkeypatch.delenv(SIMULATION_STEP_TIMEOUT_VAR, raising=False)
        once = resolve_simulation_step_timeout_s(33.0)
        assert resolve_simulation_step_timeout_s(once) == once


# ---------------------------------------------------------------------------
# retry_once_on_timeout
# ---------------------------------------------------------------------------


class TestRetryOnce:
    def test_success_runs_exactly_once(self):
        calls = []

        def attempt(i):
            calls.append(i)
            return "ok"

        assert retry_once_on_timeout(attempt, description="x") == "ok"
        assert calls == [0]

    def test_timeout_then_success_retries_exactly_once(self):
        calls = []

        def attempt(i):
            calls.append(i)
            if i == 0:
                raise SimulationTimeoutError("stuck")
            return "recovered"

        assert retry_once_on_timeout(attempt, description="x") == "recovered"
        assert calls == [0, 1]

    def test_second_expiry_fails_loud(self):
        calls = []

        def attempt(i):
            calls.append(i)
            raise SimulationTimeoutError("stuck again")

        with pytest.raises(SimulationTimeoutError, match="twice"):
            retry_once_on_timeout(attempt, description="my step")
        assert calls == [0, 1]

    def test_non_timeout_errors_propagate_without_retry(self):
        calls = []

        def attempt(i):
            calls.append(i)
            raise ValueError("real bug")

        with pytest.raises(ValueError, match="real bug"):
            retry_once_on_timeout(attempt, description="x")
        assert calls == [0]


# ---------------------------------------------------------------------------
# run_bounded (in-process watchdog for native calls)
# ---------------------------------------------------------------------------


class TestRunBounded:
    def test_normal_completion_returns_result_unchanged(self):
        payload = {"spikes": 3}
        assert run_bounded(lambda: payload, timeout_s=5.0, description="x") is payload

    def test_callee_exception_propagates(self):
        def boom():
            raise ValueError("callee bug")

        with pytest.raises(ValueError, match="callee bug"):
            run_bounded(boom, timeout_s=5.0, description="x")

    def test_hung_call_raises_timeout_promptly(self):
        t0 = time.monotonic()
        with pytest.raises(SimulationTimeoutError, match="wall cap"):
            run_bounded(lambda: time.sleep(30.0), timeout_s=0.2, description="hung sim")
        assert time.monotonic() - t0 < 5.0


# ---------------------------------------------------------------------------
# kill_process_group
# ---------------------------------------------------------------------------


def _assert_pid_dead(pid: int, deadline_s: float = 5.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still alive")


class TestKillProcessGroup:
    def test_kills_leader_and_grandchildren(self, tmp_path):
        grandchild_file = tmp_path / "grandchild_pid"
        proc = subprocess.Popen(
            ["sh", "-c", f"sleep 30 & echo $! > {grandchild_file}; wait"],
            start_new_session=True,
        )
        end = time.monotonic() + 5.0
        while not grandchild_file.exists() or not grandchild_file.read_text().strip():
            assert time.monotonic() < end, "grandchild never started"
            time.sleep(0.05)
        grandchild_pid = int(grandchild_file.read_text().strip())

        kill_process_group(proc.pid)
        assert proc.wait(timeout=5.0) != 0
        _assert_pid_dead(grandchild_pid)

    def test_already_dead_pid_is_a_noop(self):
        proc = subprocess.Popen(["true"], start_new_session=True)
        proc.wait(timeout=5.0)
        kill_process_group(proc.pid)


# ---------------------------------------------------------------------------
# run_tasks_in_pool_bounded (the hybrid segment emit+compile pool seam)
# ---------------------------------------------------------------------------


def _echo_args(*args):
    return args


def _mark_and_sleep(marker_path: str, key: int, sleep_s: float):
    with open(marker_path, "a") as f:
        f.write(f"{key}:{os.getpid()}\n")
        f.flush()
    time.sleep(sleep_s)
    return key


def _boom(_key: int):
    raise ValueError("worker boom")


def _read_marker(marker_path: Path) -> list[tuple[int, int]]:
    lines = [ln for ln in marker_path.read_text().splitlines() if ln.strip()]
    return [(int(k), int(p)) for k, p in (ln.split(":") for ln in lines)]


class TestPoolBounded:
    def test_normal_completion_results_unchanged(self):
        results = run_tasks_in_pool_bounded(
            _echo_args,
            {i: (i, i * 2) for i in range(3)},
            max_workers=2,
            timeout_s=60.0,
            description="echo pool",
        )
        assert results == {i: (i, i * 2) for i in range(3)}

    def test_stuck_workers_are_killed_retried_once_then_fail_loud(self, tmp_path):
        marker = tmp_path / "marker"
        marker.touch()
        t0 = time.monotonic()
        with pytest.raises(SimulationTimeoutError, match="twice"):
            run_tasks_in_pool_bounded(
                _mark_and_sleep,
                {0: (str(marker), 0, 60.0), 1: (str(marker), 1, 60.0)},
                max_workers=2,
                timeout_s=0.75,
                description="stuck pool",
            )
        assert time.monotonic() - t0 < 30.0
        entries = _read_marker(marker)
        # Two tasks x (initial + one retry) = four launches.
        assert sorted(k for k, _ in entries) == [0, 0, 1, 1]
        for _, pid in entries:
            _assert_pid_dead(pid)

    def test_retry_reruns_only_the_stuck_tasks(self, tmp_path):
        marker = tmp_path / "marker"
        marker.touch()
        with pytest.raises(SimulationTimeoutError):
            run_tasks_in_pool_bounded(
                _mark_and_sleep,
                {"fast": (str(marker), 7, 0.0), "stuck": (str(marker), 8, 60.0)},
                max_workers=2,
                timeout_s=1.0,
                description="partial pool",
            )
        keys = sorted(k for k, _ in _read_marker(marker))
        assert keys == [7, 8, 8]

    def test_worker_exception_fails_loud_without_retry(self, tmp_path):
        marker = tmp_path / "marker"
        marker.touch()
        with pytest.raises(ValueError, match="worker boom"):
            run_tasks_in_pool_bounded(
                _boom,
                {0: (0,)},
                max_workers=1,
                timeout_s=60.0,
                description="boom pool",
            )


# ---------------------------------------------------------------------------
# The three external-simulator seams stay bounded (source-level guard)
# ---------------------------------------------------------------------------


class TestSeamsStayBounded:
    """Each unbounded-wait seam must go through execution_bounds primitives."""

    def test_hybrid_pool_uses_bounded_pool(self):
        src = (SRC_ROOT / "chip_simulation/simulation_runner/hybrid.py").read_text()
        assert "run_tasks_in_pool_bounded" in src
        assert "as_completed" not in src

    def test_execute_nevresim_has_no_unbounded_communicate(self):
        src = (SRC_ROOT / "chip_simulation/nevresim/execute_nevresim.py").read_text()
        assert "resolve_simulation_step_timeout_s" in src
        assert ".communicate()" not in src

    def test_compile_nevresim_has_no_unbounded_wait(self):
        src = (SRC_ROOT / "chip_simulation/nevresim/compile_nevresim.py").read_text()
        assert "retry_once_on_timeout" in src
        assert ".wait()" not in src

    def test_sanafe_neural_stage_sims_through_the_watchdog(self):
        src = (SRC_ROOT / "chip_simulation/sanafe/runner/neural_stage.py").read_text()
        assert "simulate_chip_bounded" in src
        assert "chip.sim(" not in src
