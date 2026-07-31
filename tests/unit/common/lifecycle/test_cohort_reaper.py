"""SIGKILL containment: the cohort dies with the owner even when no handler runs."""

import os
import signal
import subprocess
import sys
import time

from mimarsinan.common.lifecycle.process_tree import process_identity, process_is_alive

from .test_process_tree import _state_of, _wait_for, repo_env, run_script

# The reaper polls the owner at 2s; a SIGTERM/grace/SIGKILL sweep follows. Generous
# multiples of that so a loaded machine cannot make these flaky.
COLLAPSE_BUDGET_S = 30.0

_PRELUDE = """
    import os, signal, subprocess, sys, time

    from mimarsinan.common.lifecycle.exit_contract import (
        exit_process, install_exit_contract,
    )
    from mimarsinan.common.lifecycle.owner import owner_token
"""


def _start_owner(script_text, tmp_path, name):
    """Run ``script_text`` as a session-leading owner we can SIGKILL from outside."""
    script = tmp_path / name
    script.write_text(script_text)
    return subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, text=True,
        start_new_session=True, env=repo_env(),
    )


def _read_marker(proc, prefix, timeout_s=60.0):
    """The first stdout line starting with ``prefix``, without its prefix."""
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"owner never printed {prefix!r}")


def _kill_session(pid):
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class TestOwnerSigkillCollapsesTheWholeCohort:
    """RC-07 oracle: SIGKILL of the owner must still take the cohort with it.

    Nothing in this cohort installs an owner watch, exactly like the two process
    kinds that leaked in production: ``forkserver`` and ``resource_tracker`` are
    spawned by CPython itself and never run a pool initializer or
    ``worker_init_fn``. And the exit contract's reap cannot run under SIGKILL, so
    a defence that lives inside the owner is no defence at all here.
    """

    def test_children_that_never_ran_an_initializer_die_with_a_sigkilled_owner(
        self, tmp_path,
    ):
        proc = _start_owner(
            "import subprocess, sys, time\n"
            "from mimarsinan.common.lifecycle.exit_contract import install_exit_contract\n"
            "if __name__ == '__main__':\n"
            "    install_exit_contract(term_grace_s=1.0)\n"
            "    kids = [subprocess.Popen([sys.executable, '-c',\n"
            "        'import time; time.sleep(300)']) for _ in range(3)]\n"
            "    print('COHORT %s' % ' '.join(str(k.pid) for k in kids), flush=True)\n"
            "    time.sleep(300)\n",
            tmp_path, "unwatched_cohort.py",
        )
        try:
            pids = [int(p) for p in _read_marker(proc, "COHORT ").split()]
            tokens = [process_identity(pid) for pid in pids]
            assert all(token is not None for token in tokens)

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10.0)

            assert _wait_for(
                lambda: not any(process_is_alive(t) for t in tokens if t),
                timeout_s=COLLAPSE_BUDGET_S,
            ), (
                f"{sum(process_is_alive(t) for t in tokens if t)} of {len(pids)} "
                "cohort members outlived their SIGKILLed owner; nothing outside "
                "the owner is bound to the cohort"
            )
        finally:
            _kill_session(proc.pid)

    def test_a_sigkilled_owner_leaks_no_named_semaphore(self, tmp_path):
        """The resource tracker only unlinks its cache once EVERY holder is gone.

        A surviving forkserver keeps that pipe open forever, so the semaphores the
        run registered stay in ``/dev/shm`` until the machine reboots.
        """
        proc = _start_owner(
            "import multiprocessing as mp\n"
            "import time\n"
            "from mimarsinan.common.lifecycle.exit_contract import install_exit_contract\n"
            "def _sleep_forever():\n"
            "    time.sleep(300)\n"
            "if __name__ == '__main__':\n"
            "    install_exit_contract(term_grace_s=1.0)\n"
            "    ctx = mp.get_context('forkserver')\n"
            "    lock = ctx.Lock()\n"
            "    worker = ctx.Process(target=_sleep_forever)\n"
            "    worker.start()\n"
            "    print('SEM %s' % lock._semlock.name, flush=True)\n"
            "    print('WORKER %d' % worker.pid, flush=True)\n"
            "    time.sleep(300)\n",
            tmp_path, "forkserver_cohort.py",
        )
        try:
            sem_path = "/dev/shm/sem." + _read_marker(proc, "SEM ").lstrip("/")
            worker_pid = int(_read_marker(proc, "WORKER "))
            worker_token = process_identity(worker_pid)
            assert worker_token is not None
            assert os.path.exists(sem_path), "the owner never registered a semaphore"

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10.0)

            assert _wait_for(
                lambda: not process_is_alive(worker_token), COLLAPSE_BUDGET_S
            ), f"forkserver worker {worker_pid} outlived its SIGKILLed owner"
            assert _wait_for(
                lambda: not os.path.exists(sem_path), COLLAPSE_BUDGET_S
            ), (
                f"{sem_path} leaked: a surviving cohort member still holds the "
                "resource tracker's pipe open, so its unlink pass never ran"
            )
        finally:
            _kill_session(proc.pid)

    def test_an_unreaped_zombie_owner_does_not_stall_the_teardown(self, tmp_path):
        """The launcher that SIGKILLs a run rarely ``wait()``s for it afterwards.

        The owner then sits as a zombie for as long as that launcher lives. Nothing
        of the run is left in it -- no fd, no child -- so a teardown that treats it
        as still running would simply never happen.
        """
        proc = _start_owner(
            "import subprocess, sys, time\n"
            "from mimarsinan.common.lifecycle.exit_contract import install_exit_contract\n"
            "if __name__ == '__main__':\n"
            "    install_exit_contract(term_grace_s=1.0)\n"
            "    kid = subprocess.Popen([sys.executable, '-c',\n"
            "        'import time; time.sleep(300)'])\n"
            "    print('KID %d' % kid.pid, flush=True)\n"
            "    time.sleep(300)\n",
            tmp_path, "zombie_owner.py",
        )
        try:
            kid_pid = int(_read_marker(proc, "KID "))
            kid_token = process_identity(kid_pid)
            assert kid_token is not None

            os.kill(proc.pid, signal.SIGKILL)
            # Deliberately NOT reaped: proc stays a zombie for the whole assertion.
            assert _wait_for(lambda: _state_of(proc.pid) == "Z", timeout_s=10.0), (
                f"owner never became a zombie (state={_state_of(proc.pid)})"
            )
            assert _wait_for(
                lambda: not process_is_alive(kid_token), timeout_s=COLLAPSE_BUDGET_S
            ), (
                f"cohort member {kid_pid} outlived a SIGKILLed owner that its "
                "launcher never reaped"
            )
            assert _state_of(proc.pid) == "Z", "the owner was reaped after all"
        finally:
            _kill_session(proc.pid)
            proc.wait(timeout=10.0)

    def test_the_reaper_reports_what_it_swept(self, tmp_path):
        """A silent safety net cannot be audited after an OOM kill."""
        log = tmp_path / "reaper.log"
        proc = _start_owner(
            "import subprocess, sys, time\n"
            "from mimarsinan.common.lifecycle.cohort_reaper import spawn_cohort_reaper\n"
            "from mimarsinan.common.lifecycle.owner import owner_token, stamp_cohort\n"
            "if __name__ == '__main__':\n"
            "    token = owner_token()\n"
            "    stamp_cohort(token)\n"
            f"    spawn_cohort_reaper(token, log_path={str(log)!r})\n"
            "    kid = subprocess.Popen([sys.executable, '-c',\n"
            "        'import time; time.sleep(300)'])\n"
            "    print('KID %d' % kid.pid, flush=True)\n"
            "    time.sleep(300)\n",
            tmp_path, "reaper_reports.py",
        )
        try:
            kid_pid = int(_read_marker(proc, "KID "))
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10.0)
            assert _wait_for(
                lambda: log.exists() and str(kid_pid) in log.read_text(),
                timeout_s=COLLAPSE_BUDGET_S,
            ), f"the reaper never reported the sweep; log={log.read_text() if log.exists() else '<missing>'}"
            assert "died without running its exit contract" in log.read_text()
        finally:
            _kill_session(proc.pid)


class TestCohortStamp:
    """Membership is inherited, not observed: every exec carries the owner's mark.

    A watcher that instead had to see the tree grow would miss anything spawned in
    its last poll window, and would lose every member the moment the owner died and
    the kernel reparented them.
    """

    def test_a_stamped_process_claims_its_children_and_grandchildren(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "stamped_children.py", _PRELUDE, """
            from mimarsinan.common.lifecycle.owner import cohort_members, stamp_cohort

            if __name__ == "__main__":
                token = owner_token()
                stamp_cohort(token)
                child = subprocess.Popen([sys.executable, "-c",
                    "import subprocess, sys, time;"
                    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
                    "time.sleep(30)"])
                time.sleep(1.0)
                members = cohort_members(token, exclude=(os.getpid(),))
                print("CHILD_CLAIMED %d" % (child.pid in members), flush=True)
                print("MEMBERS %d" % len(members), flush=True)
                exit_process(0, term_grace_s=1.0)
            """,
        )
        assert rc == 0, f"harness failed ({rc}); stderr:\n{err}"
        assert "CHILD_CLAIMED 1" in out, f"stamped child not claimed; stdout:\n{out}"
        counts = [ln for ln in out.splitlines() if ln.startswith("MEMBERS ")]
        assert counts and int(counts[0].split()[1]) >= 2, (
            f"the grandchild is part of the cohort too; stdout:\n{out}"
        )

    def test_an_unstamped_process_is_never_claimed(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "unstamped_bystander.py", _PRELUDE, """
            from mimarsinan.common.lifecycle.owner import cohort_members, stamp_cohort

            if __name__ == "__main__":
                bystander = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"]
                )
                time.sleep(0.5)
                token = owner_token()
                stamp_cohort(token)
                print("BYSTANDER_CLAIMED %d" % (
                    bystander.pid in cohort_members(token, exclude=(os.getpid(),))
                ), flush=True)
                exit_process(0, term_grace_s=1.0)
            """,
        )
        assert rc == 0, f"harness failed ({rc}); stderr:\n{err}"
        assert "BYSTANDER_CLAIMED 0" in out, (
            "a process started before the stamp carries another cohort's mark (or "
            f"none) and must never be swept; stdout:\n{out}"
        )


class TestReaperIsNotTheNextOrphan:
    def test_the_reaper_exits_when_the_run_ends_normally(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "reaper_normal_exit.py", _PRELUDE, """
            from mimarsinan.common.lifecycle.cohort_reaper import spawn_cohort_reaper
            from mimarsinan.common.lifecycle.owner import stamp_cohort

            if __name__ == "__main__":
                token = owner_token()
                stamp_cohort(token)
                reaper = spawn_cohort_reaper(token)
                print("REAPER %d" % reaper, flush=True)
                exit_process(0, term_grace_s=2.0)
            """,
        )
        assert rc == 0, f"harness failed ({rc}); stderr:\n{err}"
        reaper_pid = int(_read_marker_from_text(out, "REAPER "))
        assert _wait_for(
            lambda: not _pid_alive(reaper_pid), timeout_s=COLLAPSE_BUDGET_S
        ), f"cohort reaper {reaper_pid} outlived the run it was guarding"

    def test_the_reaper_does_not_pin_the_owner_stdout_pipe(self, tmp_path):
        """A reaper holding a duplicate of the run's stdout would hang every
        launcher that waits on pipe EOF -- the very bug ``run_child`` exists for."""
        script = tmp_path / "reaper_pipe.py"
        script.write_text(
            "import os, sys, time\n"
            "from mimarsinan.common.lifecycle.cohort_reaper import spawn_cohort_reaper\n"
            "from mimarsinan.common.lifecycle.owner import owner_token, stamp_cohort\n"
            "if __name__ == '__main__':\n"
            "    token = owner_token()\n"
            "    stamp_cohort(token)\n"
            "    print('REAPER %d' % spawn_cohort_reaper(token), flush=True)\n"
            "    os._exit(0)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, env=repo_env(),
        )
        out, _err = proc.communicate(timeout=30.0)
        reaper_pid = int(_read_marker_from_text(out, "REAPER "))
        try:
            assert proc.returncode == 0
        finally:
            try:
                os.kill(reaper_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class TestExitContractInstallsTheReaper:
    def test_installing_the_contract_starts_exactly_one_reaper(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "contract_starts_reaper.py", _PRELUDE, """
            from mimarsinan.common.lifecycle.process_tree import iter_descendants

            def _is_reaper(pid):
                with open("/proc/%d/cmdline" % pid, "rb") as f:
                    return b"cohort_reaper" in f.read()

            if __name__ == "__main__":
                install_exit_contract(term_grace_s=1.0)
                install_exit_contract(term_grace_s=1.0)
                time.sleep(0.5)
                reapers = [p for p in iter_descendants(os.getpid()) if _is_reaper(p)]
                print("REAPERS %d" % len(reapers), flush=True)
                exit_process(0, term_grace_s=2.0)
            """,
        )
        assert rc == 0, f"harness failed ({rc}); stderr:\n{err}"
        assert "REAPERS 1" in out, (
            "install_exit_contract must start one -- and only one -- cohort reaper "
            f"however often it is called; stdout:\n{out}\n{err}"
        )


def _read_marker_from_text(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"no {prefix!r} line in:\n{text}")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
