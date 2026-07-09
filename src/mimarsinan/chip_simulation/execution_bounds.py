"""Wall-clock bounds for external-simulator invocations: resolve, kill, retry-once, fail loud."""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, Mapping, Tuple, TypeVar

from mimarsinan.common.env import simulation_step_timeout_override

DEFAULT_SIMULATION_STEP_TIMEOUT_S = 900.0

# Pool workers must not FORK the parent: a run's main process carries live CUDA
# and OpenMP state by simulation time, and forked children crash abruptly on it.
_POOL_MP_CONTEXT = multiprocessing.get_context("spawn")

R = TypeVar("R")
K = TypeVar("K")


class SimulationTimeoutError(RuntimeError):
    """An external-simulator invocation exceeded its wall cap."""


def resolve_simulation_step_timeout_s(config_value: float | None = None) -> float:
    """Effective wall cap for one simulator invocation: env override > config value > 900s default."""
    override = simulation_step_timeout_override()
    value = override if override is not None else config_value
    if value is None:
        value = DEFAULT_SIMULATION_STEP_TIMEOUT_S
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"simulation_step_timeout_s must be > 0; got {value}")
    return value


def kill_process_group(pid: int) -> None:
    """SIGKILL the process group led by ``pid`` (grandchildren included), falling back to the single process; already-dead targets are a no-op."""
    try:
        os.killpg(pid, signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def retry_once_on_timeout(run_attempt: Callable[[int], R], *, description: str) -> R:
    """Run ``run_attempt(attempt_index)``; on SimulationTimeoutError retry exactly once; a second expiry raises loud."""
    try:
        return run_attempt(0)
    except SimulationTimeoutError as first_expiry:
        print(
            f"[execution-bounds] {description}: wall cap expired, retrying once "
            f"({first_expiry})"
        )
        try:
            return run_attempt(1)
        except SimulationTimeoutError as second_expiry:
            raise SimulationTimeoutError(
                f"{description}: wall cap expired twice (initial run + one retry); "
                f"failing the step: {second_expiry}"
            ) from first_expiry


def run_bounded(fn: Callable[[], R], *, timeout_s: float, description: str) -> R:
    """Run ``fn`` under a wall cap on a daemon watchdog thread.

    A native call that never returns cannot be killed in-process: on expiry the
    daemon thread is abandoned and SimulationTimeoutError raises in the caller.
    """
    outcome: list[R] = []
    error: list[BaseException] = []

    def _target() -> None:
        # Cross-thread exception transport: everything re-raises in the caller.
        try:
            outcome.append(fn())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(
        target=_target, name=f"bounded[{description}]", daemon=True,
    )
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise SimulationTimeoutError(
            f"{description}: no result after the {timeout_s:.0f}s wall cap "
            "(in-process native call; watchdog thread abandoned)"
        )
    if error:
        raise error[0]
    return outcome[0]


def _pool_worker_snapshot(executor: ProcessPoolExecutor) -> list[Any]:
    """The live pool worker processes; MUST be taken before shutdown() drops ``_processes``."""
    # _processes is the only seam exposing worker pids on ProcessPoolExecutor.
    return list((getattr(executor, "_processes", None) or {}).values())


def _kill_pool_workers(workers: list[Any]) -> None:
    """Kill every pool worker's process group (workers ``setpgrp`` at start, so compile grandchildren die too)."""
    for worker in workers:
        kill_process_group(worker.pid)


def _harvest_completed(
    futures: Mapping["Future[R]", K], results: Dict[K, R],
) -> None:
    """Keep results that finished before the deadline so a retry only re-runs stuck tasks."""
    for future, key in futures.items():
        if key in results or not future.done() or future.cancelled():
            continue
        if future.exception() is None:
            results[key] = future.result()


def run_tasks_in_pool_bounded(
    fn: Callable[..., R],
    task_args: Mapping[K, Tuple[Any, ...]],
    *,
    max_workers: int,
    timeout_s: float,
    description: str,
) -> Dict[K, R]:
    """Run ``fn(*args)`` per task under a pool-wide deadline.

    On expiry: cancel pending futures, kill the worker process groups, retry the
    still-incomplete tasks once, and fail loud on a second expiry.
    """
    results: Dict[K, R] = {}

    def attempt(attempt_index: int) -> None:
        pending = {key: args for key, args in task_args.items() if key not in results}
        if not pending:
            return
        executor = ProcessPoolExecutor(
            max_workers=max(1, min(max_workers, len(pending))),
            initializer=os.setpgrp,
            mp_context=_POOL_MP_CONTEXT,
        )
        futures: Dict["Future[R]", K] = {
            executor.submit(fn, *args): key for key, args in pending.items()
        }
        try:
            try:
                for future in as_completed(futures, timeout=timeout_s):
                    results[futures[future]] = future.result()
            except FuturesTimeoutError:
                workers = _pool_worker_snapshot(executor)
                executor.shutdown(wait=False, cancel_futures=True)
                _kill_pool_workers(workers)
                _harvest_completed(futures, results)
                stuck = sorted(str(key) for key in pending if key not in results)
                raise SimulationTimeoutError(
                    f"{description}: {len(stuck)} task(s) still running after the "
                    f"{timeout_s:.0f}s wall cap (attempt {attempt_index + 1}); "
                    f"killed the pool workers; stuck: {stuck[:8]}"
                ) from None
            else:
                executor.shutdown(wait=True)
        finally:
            # Idempotent after a normal shutdown; reaps the pool on error paths.
            executor.shutdown(wait=False, cancel_futures=True)

    retry_once_on_timeout(attempt, description=description)
    return results
