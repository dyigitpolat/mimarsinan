"""Process-tree enumeration and reaping: prompt, orphan-free process exits."""

import os
import subprocess
import sys
import threading
import time

from mimarsinan.common.process_tree import iter_descendants, reap_descendants

_SLEEP_CHILD = [sys.executable, "-c", "import time; time.sleep(300)"]


def _spawn_child_with_grandchild():
    """Child that spawns a long-lived grandchild, then sleeps itself."""
    return subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']);"
        "time.sleep(300)",
    ])


def _wait_for(predicate, timeout_s, interval_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


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
            "from mimarsinan.common.process_tree import reap_descendants;"
            "reap_descendants(term_grace_s=2.0);"
            "os._exit(0)"
        )
        env = dict(os.environ)
        repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
        env["PYTHONPATH"] = os.path.abspath(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
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
