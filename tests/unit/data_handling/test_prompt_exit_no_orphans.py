"""Prompt-exit contract: persistent forkserver DataLoader workers never outlive the run."""

import os
import subprocess
import sys
import textwrap
import threading

from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    close_pipeline_loaders,
)

from conftest import MockPipeline

_CHILD_SCRIPT = textwrap.dedent("""
    import os, sys
    import torch

    from mimarsinan.data_handling.data_loader_factory import (
        DataLoaderFactory, close_pipeline_loaders,
    )
    from mimarsinan.common.process_tree import reap_descendants

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
        print("WORKERS_UP", flush=True)

        close_pipeline_loaders(pipeline)
        reap_descendants(term_grace_s=5.0)
        os._exit(0)
""")


class TestPromptExitReleasesPipes:
    def test_headless_exit_seam_reaches_eof_with_no_survivors(self, tmp_path):
        env = dict(os.environ)
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(repo_root, "src"), env.get("PYTHONPATH", "")]
        )
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        # forkserver workers unpickle __main__ classes, so the script must be a file.
        script_path = tmp_path / "prompt_exit_child.py"
        script_path.write_text(_CHILD_SCRIPT)
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=env, text=True,
        )
        try:
            eof = {}

            def _read_all():
                assert proc.stdout is not None
                eof["stdout"] = proc.stdout.read()
                eof["reached"] = True

            reader = threading.Thread(target=_read_all, daemon=True)
            reader.start()
            reader.join(timeout=60.0)
            stderr = ""
            if not eof.get("reached"):
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass
                reader.join(timeout=5.0)
            if proc.stderr is not None:
                try:
                    stderr = proc.stderr.read()
                except Exception:
                    pass
            assert eof.get("reached"), (
                "stdout EOF never arrived: forkserver/dataloader processes "
                f"outlived the run. stderr:\n{stderr}"
            )
            assert "WORKERS_UP" in eof.get("stdout", ""), (
                f"child never spun up workers; stderr:\n{stderr}"
            )
            assert proc.wait(timeout=10.0) == 0, f"child failed; stderr:\n{stderr}"

            import time
            deadline = time.monotonic() + 10.0
            group_gone = False
            while time.monotonic() < deadline:
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    group_gone = True
                    break
                time.sleep(0.1)
            assert group_gone, "processes survive in the child's process group"
        finally:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass


class TestClosePipelineLoaders:
    def test_closes_pooled_loaders_of_shared_factory(self):
        pipeline = MockPipeline(config={"num_workers": 0, "device": "cpu"})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        provider = factory.create_data_provider()
        factory.create_training_loader(4, provider)
        assert factory._loader_cache
        close_pipeline_loaders(pipeline)
        assert not factory._loader_cache
        assert not factory._eval_cache

    def test_noop_when_pipeline_has_no_factory(self):
        pipeline = MockPipeline(config={"num_workers": 0, "device": "cpu"})
        close_pipeline_loaders(pipeline)

    def test_noop_on_none(self):
        close_pipeline_loaders(None)
