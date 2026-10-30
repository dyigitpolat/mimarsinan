"""[ODIN multi-core, targeted pass] one testbench build per cache key, however many workers ask.

The RTL gates run under pytest-xdist, and every case of one geometry asks
``toolchain.build_testbench`` for the SAME cached build. Without a lock, the
workers that arrive before the first build finishes all start Verilator in the
same directory and corrupt each other's objects (an invalid precompiled header,
a missing ``__Syms.h``), so a gate run in parallel reports build errors instead
of verdicts. The build is now taken under an exclusive lock per cache entry and
the cache is re-checked inside it: the first worker builds, the rest reuse.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from mimarsinan.chip_simulation.odin_rtl import toolchain


class _Completed:
    returncode = 0
    stdout = ""
    stderr = ""


def test_concurrent_requests_for_one_build_run_the_compiler_once(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "BUILD_CACHE", tmp_path / "cache")
    monkeypatch.setattr(toolchain, "require_tool", lambda name: Path("/stub") / name)
    calls = []

    def compile_slowly(command, **_kwargs):
        calls.append(list(command))
        time.sleep(0.3)
        binary = Path(command[command.index("-o") + 1])
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("built\n")
        return _Completed()

    monkeypatch.setattr(toolchain.subprocess, "run", compile_slowly)
    builds = []

    def ask():
        builds.append(toolchain.build_testbench(
            n_cores=2, token_count=100, engine=toolchain.ENGINE_COMPILED,
            tb_name="tb_odin_fpga_kernel_axi", rtl_sources=[]))

    workers = [threading.Thread(target=ask) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(calls) == 1, f"{len(calls)} compiler runs for one cache key"
    assert len({build.binary for build in builds}) == 1
    assert sorted(build.cached for build in builds) == [False, True, True, True]
