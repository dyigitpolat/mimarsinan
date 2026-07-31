"""Die-with-owner contract: a spawned worker must not outlive the run that owns it."""

import os
import signal
import subprocess
import sys
import time

from mimarsinan.common.lifecycle.owner import (
    ORPHAN_EXIT_CODE,
    owner_token,
    watch_owner,
)
from mimarsinan.common.lifecycle.process_tree import process_identity

from .test_process_tree import repo_env, run_script


def _wait_for(predicate, timeout_s, interval_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


class TestOwnerToken:
    def test_default_owner_is_this_process(self):
        assert owner_token() == process_identity(os.getpid())

    def test_explicit_pid_is_honoured(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        try:
            assert owner_token(proc.pid) == process_identity(proc.pid)
        finally:
            proc.kill()
            proc.wait()


class TestWatchOwner:
    def test_watch_is_a_daemon_thread_that_never_blocks_exit(self):
        thread = watch_owner(owner_token(), poll_interval_s=0.05)
        assert thread.daemon
        assert thread.is_alive()

    def test_watch_of_a_dead_owner_calls_back_immediately(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        token = process_identity(proc.pid)
        assert token is not None
        proc.wait()

        fired = []
        thread = watch_owner(token, poll_interval_s=0.02, on_owner_death=fired.append)
        thread.join(timeout=5.0)
        assert fired == [token]


class TestOrphanedWorkerSelfTerminates:
    """RC-06 oracle: killing the owner must terminate the worker it spawned.

    Under a forkserver start method a worker's real parent is the forkserver,
    whose pid never changes when the run dies, so no ``getppid`` watchdog in the
    worker can ever fire. The owner token closes that hole.
    """

    def test_worker_dies_when_its_owner_is_sigkilled(self, tmp_path):
        script = tmp_path / "owned_worker.py"
        script.write_text(
            "import os, subprocess, sys, time\n"
            "from mimarsinan.common.lifecycle.owner import owner_token\n"
            "if __name__ == '__main__':\n"
            "    token = owner_token()\n"
            "    worker = subprocess.Popen([sys.executable, '-c',\n"
            "        'import sys, time;'\n"
            "        'from mimarsinan.common.lifecycle.owner import install_owner_watch;'\n"
            "        'install_owner_watch(sys.argv[1], poll_interval_s=0.2);'\n"
            "        'time.sleep(300)', token])\n"
            "    print('WORKER %d' % worker.pid, flush=True)\n"
            "    time.sleep(300)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE, text=True,
            start_new_session=True, env=repo_env(),
        )
        try:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            assert line.startswith("WORKER "), f"owner never spawned a worker: {line!r}"
            worker_pid = int(line.split()[1])
            worker_token = process_identity(worker_pid)
            assert worker_token is not None

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5.0)

            from mimarsinan.common.lifecycle.process_tree import process_is_alive
            assert _wait_for(
                lambda: not process_is_alive(worker_token), timeout_s=15.0
            ), f"worker {worker_pid} outlived its SIGKILLed owner"
        finally:
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass

    def test_orphaned_worker_exit_code_names_the_cause(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "orphan_exit_code.py",
            """
            import os, subprocess, sys

            from mimarsinan.common.lifecycle.owner import owner_token

            if __name__ == "__main__":
                dead = subprocess.Popen([sys.executable, "-c", "pass"])
                token = "%d:%d" % (dead.pid, 1)
                dead.wait()
                worker = subprocess.Popen([sys.executable, "-c",
                    "import sys, time;"
                    "from mimarsinan.common.lifecycle.owner import install_owner_watch;"
                    "install_owner_watch(sys.argv[1], poll_interval_s=0.05);"
                    "time.sleep(30)", token])
                print("RC %d" % worker.wait(), flush=True)
            """,
        )
        assert rc == 0, f"harness failed ({rc}); stderr:\n{err}"
        rc_lines = [ln for ln in out.splitlines() if ln.startswith("RC ")]
        assert rc_lines, f"worker never exited; stdout:\n{out}\n{err}"
        assert int(rc_lines[0].split()[1]) == ORPHAN_EXIT_CODE
