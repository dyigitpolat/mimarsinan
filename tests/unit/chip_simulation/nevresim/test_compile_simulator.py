"""compile_simulator must define `success` before the time-trace branch reads it."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.nevresim import compile_nevresim
from mimarsinan.chip_simulation.nevresim.compile_nevresim import CompileResult, compile_simulator


class _FakeProc:
    def __init__(self, rc: int) -> None:
        self._rc = rc

    def wait(self) -> int:
        return self._rc


@pytest.mark.parametrize("rc", [0, 1])
def test_time_trace_clang_does_not_reference_success_before_assignment(
    tmp_path, monkeypatch, rc
) -> None:
    """time_trace + clang formerly hit `success` before assignment (UnboundLocalError)."""
    monkeypatch.setattr(compile_nevresim, "find_cpp20_compiler", lambda: ("clang++", "clang"))
    monkeypatch.setattr(compile_nevresim.subprocess, "Popen", lambda *a, **k: _FakeProc(rc))

    result = compile_simulator(
        generated_files_path=str(tmp_path),
        nevresim_path=str(tmp_path),
        output_path=str(tmp_path / "bin" / "simulator"),
        verbose=False,
        time_trace=True,
    )

    assert isinstance(result, CompileResult)
    assert result.success is (rc == 0)
    assert result.compiler_family == "clang"
