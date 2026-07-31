"""A SIGKILLed run must not leave its forkserver/dataloader cohort behind."""

import os
import signal
import subprocess
import sys
import textwrap
import time

from mimarsinan.common.lifecycle.process_tree import iter_descendants, process_identity

_CHILD_SCRIPT = textwrap.dedent("""
    import os, sys, time
    import torch

    from mimarsinan.common.lifecycle.process_tree import iter_descendants
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16
        def __getitem__(self, idx):
            return torch.zeros(4), 0

    class _Provider:
        def is_mp_safe(self):
            return True
        def enable_ffcv(self):
            return False
        def _get_training_dataset(self):
            return _Dataset()

    class _Factory:
        def create(self):
            return _Provider()

    class _Pipeline:
        config = {"num_workers": 2}
        data_provider_factory = _Factory()

    if __name__ == "__main__":
        pipeline = _Pipeline()
        factory = DataLoaderFactory.for_pipeline(pipeline)
        provider = factory.create_data_provider()
        loader = factory.create_training_loader(4, provider)
        next(iter(loader))
        print("COHORT " + " ".join(str(p) for p in iter_descendants(os.getpid())),
              flush=True)
        time.sleep(300)
""")


def _await(predicate, timeout_s, interval_s=0.1):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


class TestOrphanContainment:
    """RC-06 oracle: no member of the cohort may outlive an uncatchably-killed run.

    A DataLoader worker's real parent is the forkserver, whose pid never changes
    when the run dies, so torch's ``ManagerWatchdog`` can never fire; and the
    forkserver's own alive-pipe cannot reach EOF while any worker holds a
    duplicate of the write end. Without an owner-liveness contract the whole
    cohort -- forkserver, workers and resource tracker -- lives forever holding
    the dead run's stdout/stderr.
    """

    def test_sigkilled_run_leaves_no_forkserver_worker_or_tracker(self, tmp_path):
        env = dict(os.environ)
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(repo_root, "src"), env.get("PYTHONPATH", "")]
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        # forkserver workers unpickle __main__ classes, so the script must be a file.
        script_path = tmp_path / "orphan_containment_child.py"
        script_path.write_text(_CHILD_SCRIPT)

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=env, text=True,
        )
        try:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            assert line.startswith("COHORT "), (
                f"child never spun up workers: {line!r}\n"
                f"{proc.stderr.read() if proc.stderr else ''}"
            )
            cohort = [int(p) for p in line.split()[1:]]
            assert len(cohort) >= 3, (
                f"expected forkserver + tracker + workers, got {cohort}"
            )
            tokens = {pid: process_identity(pid) for pid in cohort}
            assert all(tokens.values()), f"cohort vanished before the kill: {tokens}"

            # Uncatchable: no handler, no atexit, no finally -- exactly what a
            # wall watchdog, the OOM killer or a launcher timeout does.
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10.0)

            def _cohort_gone():
                return not [
                    pid for pid, token in tokens.items()
                    if token is not None and process_identity(pid) == token
                ]

            survivors = []
            if not _await(_cohort_gone, timeout_s=30.0):
                survivors = [
                    pid for pid, token in tokens.items()
                    if token is not None and process_identity(pid) == token
                ]
            assert not survivors, (
                f"cohort members {survivors} outlived their SIGKILLed owner "
                f"(pid {proc.pid}); they still hold its stdout/stderr"
            )
        finally:
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass


class TestOwnerWatchIsWiredIntoWorkers:
    def test_multi_worker_loaders_carry_an_owner_bound_init(self):
        from mimarsinan.common.lifecycle.owner import install_owner_watch, owner_token
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

        factory = DataLoaderFactory(_provider_factory(), num_workers=2)
        init_fn = factory.worker_init_fn(workers=2)
        assert init_fn is not None
        assert init_fn.func is install_owner_watch
        assert init_fn.args == (owner_token(),)

    def test_single_process_loaders_need_no_watch(self):
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

        factory = DataLoaderFactory(_provider_factory(), num_workers=0)
        assert factory.worker_init_fn(workers=0) is None


def _provider_factory():
    class _Provider:
        def is_mp_safe(self):
            return True

        def enable_ffcv(self):
            return False

    class _Factory:
        def create(self):
            return _Provider()

    return _Factory()


def test_iter_descendants_is_importable_from_the_lifecycle_ssot():
    assert callable(iter_descendants)
