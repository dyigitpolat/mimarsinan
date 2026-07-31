"""The process exit contract: one reap-then-hard-exit epilogue on every path."""

import os
import signal
import time

import pytest

from mimarsinan.common.lifecycle.exit_contract import (
    TERMINATION_SIGNALS,
    signal_exit_code,
)

from .test_process_tree import run_script

_PRELUDE = """
    import os, signal, subprocess, sys, time

    from mimarsinan.common.lifecycle.exit_contract import (
        exit_process, install_exit_contract,
    )
    from mimarsinan.common.lifecycle.process_tree import iter_descendants
"""


def _await_pid_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class TestSignalExitCode:
    @pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
    def test_every_termination_signal_maps_to_the_shell_convention(self, signum):
        assert signal_exit_code(signum) == 128 + int(signum)

    def test_termination_signals_cover_the_catchable_kill_paths(self):
        assert set(TERMINATION_SIGNALS) == {
            signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT,
        }


class TestExitProcessOrdering:
    """RC-03 oracle: children are reaped BEFORE any telemetry teardown runs.

    While the teardown runs, every live worker still holds a duplicate of this
    process's stdout/stderr; an external SIGKILL during that window orphans the
    whole cohort onto a launcher's capture pipe.
    """

    def test_descendants_are_gone_before_teardown_starts(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "reap_before_teardown.py",
            _PRELUDE, """
            def _teardown():
                print("TEARDOWN_SEES %d" % len(iter_descendants(os.getpid())), flush=True)

            if __name__ == "__main__":
                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
                time.sleep(0.5)
                assert iter_descendants(os.getpid()), "no child to reap"
                exit_process(0, teardown=_teardown, term_grace_s=2.0)
            """,
        )
        assert rc == 0, f"child failed ({rc}); stderr:\n{err}"
        seen = [ln for ln in out.splitlines() if ln.startswith("TEARDOWN_SEES ")]
        assert seen, f"teardown never ran; stdout:\n{out}\n{err}"
        assert int(seen[0].split()[1]) == 0, (
            "the telemetry teardown ran while worker processes were still alive "
            "holding this process's stdout/stderr"
        )

    def test_a_raising_teardown_still_reaches_the_hard_exit(self, tmp_path):
        rc, _out, err = run_script(
            tmp_path, "teardown_raises.py",
            _PRELUDE, """
            def _teardown():
                raise RuntimeError("telemetry blew up")

            if __name__ == "__main__":
                exit_process(7, teardown=_teardown, term_grace_s=0.5)
            """,
        )
        assert rc == 7, f"exit code lost to a failing teardown; stderr:\n{err}"


class TestInstalledSignalContract:
    """RC-02 oracle: SIGHUP/SIGQUIT/SIGINT reap exactly like SIGTERM does."""

    @pytest.mark.parametrize("signame", ["SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"])
    def test_signal_reaps_children_and_exits_with_the_shell_code(
        self, tmp_path, signame
    ):
        rc, out, err = run_script(
            tmp_path, f"signal_{signame.lower()}.py",
            _PRELUDE, f"""
            if __name__ == "__main__":
                install_exit_contract(term_grace_s=1.0)
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(300)"]
                )
                time.sleep(0.5)
                print("CHILD %d" % child.pid, flush=True)
                os.kill(os.getpid(), signal.{signame})
                time.sleep(30)
            """,
        )
        assert rc == 128 + int(getattr(signal, signame)), (
            f"{signame} did not exit through the contract (rc={rc}); stderr:\n{err}"
        )
        child_lines = [ln for ln in out.splitlines() if ln.startswith("CHILD ")]
        assert child_lines, f"no child was spawned; stdout:\n{out}\n{err}"
        child_pid = int(child_lines[0].split()[1])
        assert _await_pid_gone(child_pid, timeout_s=10.0), (
            f"{signame} left child {child_pid} orphaned: the exit contract is "
            "not installed on this signal"
        )

    def test_install_is_idempotent_and_the_notice_is_refinable(self, tmp_path):
        rc, out, err = run_script(
            tmp_path, "idempotent_install.py",
            _PRELUDE, """
            def _notice(signum):
                print("NOTICE %d" % signum, flush=True)

            if __name__ == "__main__":
                install_exit_contract(term_grace_s=0.5)
                install_exit_contract(on_terminate=_notice, term_grace_s=0.5)
                os.kill(os.getpid(), signal.SIGHUP)
                time.sleep(30)
            """,
        )
        assert rc == 128 + int(signal.SIGHUP), f"stderr:\n{err}"
        assert f"NOTICE {int(signal.SIGHUP)}" in out, (
            f"the re-installed notice never fired; stdout:\n{out}\n{err}"
        )
