"""Process-tree enumeration and reaping: prompt, orphan-free, resource-free exits."""

import os
import subprocess
import sys
import textwrap
import threading
import time

from mimarsinan.common.lifecycle.process_tree import (
    iter_descendants,
    process_identity,
    process_is_alive,
    reap_descendants,
)

_SLEEP_CHILD = [sys.executable, "-c", "import time; time.sleep(300)"]


def _spawn_child_with_grandchild():
    """Child that spawns a long-lived grandchild, then sleeps itself."""
    return subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']);"
        "time.sleep(300)",
    ])


def _state_of(pid):
    """The /proc state character (``Z`` for an exited-but-unreaped process)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
    except OSError:
        return "gone"
    return raw[raw.rindex(")") + 2:].split()[0]


def _wait_for(predicate, timeout_s, interval_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def repo_env():
    """Child environment that can import mimarsinan from the worktree."""
    env = dict(os.environ)
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
    env["PYTHONPATH"] = os.path.abspath(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


def run_script(tmp_path, name, *body_parts, timeout_s=60.0):
    """Run the dedented concatenation of ``body_parts`` as a real process file."""
    script = tmp_path / name
    script.write_text("".join(textwrap.dedent(part) for part in body_parts))
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True, env=repo_env(),
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
    return proc.returncode, out, err


class TestIterDescendants:
    def test_finds_direct_child(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        try:
            assert _wait_for(
                lambda: proc.pid in iter_descendants(os.getpid()), timeout_s=5.0
            )
        finally:
            proc.kill()
            proc.wait()

    def test_finds_grandchild(self):
        proc = _spawn_child_with_grandchild()
        try:
            def _has_grandchild():
                descendants = iter_descendants(os.getpid())
                return proc.pid in descendants and len(
                    [p for p in descendants if p != proc.pid]
                ) >= 1 and any(
                    p in iter_descendants(proc.pid) for p in descendants
                )
            assert _wait_for(_has_grandchild, timeout_s=10.0)
        finally:
            reap_descendants(root_pid=proc.pid, term_grace_s=2.0)
            proc.kill()
            proc.wait()

    def test_no_children_yields_empty(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        try:
            assert _wait_for(lambda: proc.pid in iter_descendants(os.getpid()), 5.0)
            assert iter_descendants(proc.pid) == []
        finally:
            proc.kill()
            proc.wait()

    def test_dead_root_yields_empty(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        proc.kill()
        proc.wait()
        assert iter_descendants(proc.pid) == []


class TestProcessIdentity:
    def test_identity_is_stable_for_a_live_process(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        try:
            first = process_identity(proc.pid)
            assert first is not None
            assert process_identity(proc.pid) == first
            assert process_is_alive(first)
        finally:
            proc.kill()
            proc.wait()

    def test_identity_carries_start_time_so_pid_reuse_cannot_alias(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        try:
            token = process_identity(proc.pid)
            assert token is not None
            pid_part, _, start_part = token.partition(":")
            assert int(pid_part) == proc.pid
            assert int(start_part) > 0
            # A same-pid token with a different start time is a DIFFERENT process.
            assert not process_is_alive(f"{proc.pid}:{int(start_part) + 1}")
        finally:
            proc.kill()
            proc.wait()

    def test_an_exited_but_unreaped_process_is_not_alive(self):
        """A zombie holds no fd and no child and can spawn none, so a watcher must
        not keep waiting on it just because its parent has yet to call wait()."""
        proc = subprocess.Popen(_SLEEP_CHILD)
        token = process_identity(proc.pid)
        assert token is not None
        proc.kill()
        try:
            assert _wait_for(lambda: _state_of(proc.pid) == "Z", timeout_s=5.0), (
                f"process never became a zombie (state={_state_of(proc.pid)})"
            )
            assert not process_is_alive(token)
        finally:
            proc.wait()

    def test_dead_process_has_no_identity_and_is_not_alive(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        token = process_identity(proc.pid)
        assert token is not None
        proc.kill()
        proc.wait()
        assert _wait_for(lambda: not process_is_alive(token), timeout_s=5.0)


class TestReapDescendants:
    def test_kills_child_and_grandchild(self):
        proc = _spawn_child_with_grandchild()
        assert _wait_for(lambda: len(iter_descendants(proc.pid)) >= 1, 10.0)
        grandchildren = iter_descendants(proc.pid)

        report = reap_descendants(root_pid=os.getpid(), term_grace_s=3.0)
        proc.wait(timeout=5.0)

        signalled = set(report["terminated"]) | set(report["killed"])
        assert proc.pid in signalled
        assert set(grandchildren) <= signalled

        def _all_gone():
            live = iter_descendants(os.getpid())
            return proc.pid not in live and not (set(grandchildren) & set(live))
        assert _wait_for(_all_gone, timeout_s=5.0)

    def test_noop_without_children_never_raises(self):
        proc = subprocess.Popen(_SLEEP_CHILD)
        try:
            assert _wait_for(lambda: proc.pid in iter_descendants(os.getpid()), 5.0)
            report = reap_descendants(root_pid=proc.pid, term_grace_s=0.5)
            assert report["terminated"] == [] and report["killed"] == []
        finally:
            proc.kill()
            proc.wait()

    def test_reaped_grandchild_releases_inherited_stdout_pipe(self):
        """The launcher-hang oracle: EOF must arrive even though a grandchild
        inherited our pipe, because reap_descendants kills it before exit."""
        script = (
            "import os, subprocess, sys;"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']);"
            "from mimarsinan.common.lifecycle.process_tree import reap_descendants;"
            "reap_descendants(term_grace_s=2.0);"
            "os._exit(0)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, env=repo_env(),
        )
        try:
            eof = {}

            def _read_to_eof():
                assert proc.stdout is not None
                proc.stdout.read()
                eof["reached"] = True

            reader = threading.Thread(target=_read_to_eof, daemon=True)
            reader.start()
            reader.join(timeout=20.0)
            assert eof.get("reached"), "stdout EOF never arrived: leaked process holds the pipe"
            assert proc.wait(timeout=5.0) == 0

            def _pgroup_gone():
                try:
                    os.killpg(proc.pid, 0)
                    return False
                except ProcessLookupError:
                    return True
            assert _wait_for(_pgroup_gone, timeout_s=10.0)
        finally:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass


def shm_path(sem_name: str) -> str:
    return "/dev/shm/sem." + sem_name.lstrip("/")


class TestReapReleasesNamedResources:
    """RC-05 oracle: the reap must not destroy the resource tracker's unlink pass.

    ``multiprocessing.resource_tracker`` sets SIGTERM to SIG_IGN and unlinks its
    cache only from the ``finally:`` of its read loop, which is reached on pipe
    EOF. A blanket SIGKILL therefore orphans every named semaphore forever.
    """

    def test_hard_exit_leaves_no_orphaned_named_semaphores(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "named_sem_child.py",
            """
            import multiprocessing as mp
            import os

            from mimarsinan.common.lifecycle.process_tree import reap_descendants

            if __name__ == "__main__":
                ctx = mp.get_context("spawn")
                locks = [ctx.Lock() for _ in range(3)]
                print("NAMES " + " ".join(l._semlock.name for l in locks), flush=True)

                reap_descendants(term_grace_s=2.0)
                os._exit(0)
            """,
        )
        assert rc == 0, f"child failed ({rc}); stderr:\n{err}"
        name_lines = [ln for ln in out.splitlines() if ln.startswith("NAMES ")]
        assert name_lines, f"child never reported semaphore names; stdout:\n{out}\n{err}"
        names = name_lines[0].split()[1:]
        assert len(names) == 3

        leaked = [
            n for n in names
            if not _wait_for(lambda n=n: not os.path.exists(shm_path(n)), timeout_s=5.0)
        ]
        assert not leaked, (
            "named semaphores survived the exit; the resource tracker was SIGKILLed "
            f"before it could unlink them: {leaked}"
        )

    def test_reap_returns_promptly_when_only_the_tracker_ignores_sigterm(self, tmp_path):
        """The tracker can never die from SIGTERM, so it must not pin the grace loop."""
        rc, out, err = run_script(
            tmp_path, "grace_timing_child.py",
            """
            import multiprocessing as mp
            import os
            import time

            from mimarsinan.common.lifecycle.process_tree import (
                iter_descendants, reap_descendants,
            )

            if __name__ == "__main__":
                ctx = mp.get_context("spawn")
                lock = ctx.Lock()
                assert iter_descendants(os.getpid()), "no resource tracker spawned"

                started = time.monotonic()
                reap_descendants(term_grace_s=5.0)
                print("ELAPSED %.3f" % (time.monotonic() - started), flush=True)
                os._exit(0)
            """,
        )
        assert rc == 0, f"child failed ({rc}); stderr:\n{err}"
        elapsed_lines = [ln for ln in out.splitlines() if ln.startswith("ELAPSED ")]
        assert elapsed_lines, f"child never timed the reap; stdout:\n{out}\n{err}"
        elapsed = float(elapsed_lines[0].split()[1])
        assert elapsed < 2.5, (
            f"reap_descendants burned {elapsed:.2f}s of its 5s grace waiting on a "
            "process that ignores SIGTERM; the grace must end when the cohort is gone"
        )
