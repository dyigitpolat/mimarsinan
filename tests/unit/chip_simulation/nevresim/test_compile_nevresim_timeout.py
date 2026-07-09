"""compile_simulator wall cap: hung compilers are killed, retried once, then fail loud."""

from __future__ import annotations

import stat
import time
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.execution_bounds import SimulationTimeoutError
from mimarsinan.chip_simulation.nevresim import compile_nevresim
from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator


def _write_script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _patch_compiler(monkeypatch, script: Path) -> None:
    monkeypatch.setattr(
        compile_nevresim, "find_cpp20_compiler", lambda: (str(script), "gcc"),
    )


def test_hung_compiler_is_killed_retried_once_then_fails_loud(tmp_path, monkeypatch):
    attempts = tmp_path / "attempts"
    fake_cc = _write_script(
        tmp_path / "fake_cc",
        f"echo started >> {attempts}\nexec sleep 60\n",
    )
    _patch_compiler(monkeypatch, fake_cc)

    t0 = time.monotonic()
    with pytest.raises(SimulationTimeoutError, match="twice"):
        compile_simulator(
            str(tmp_path), str(tmp_path),
            output_path=str(tmp_path / "bin" / "simulator"),
            verbose=False, timeout_s=0.4,
        )
    assert time.monotonic() - t0 < 20.0
    assert attempts.read_text().count("started") == 2, "expected exactly one retry"


def test_normal_compile_returns_binary_path_unchanged(tmp_path, monkeypatch):
    fake_cc = _write_script(
        tmp_path / "fake_cc",
        'out=""\nwhile [ $# -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; fi\n  shift\ndone\n'
        ': > "$out"\n',
    )
    _patch_compiler(monkeypatch, fake_cc)

    out_path = tmp_path / "bin" / "simulator"
    binary = compile_simulator(
        str(tmp_path), str(tmp_path), output_path=str(out_path),
        verbose=False, timeout_s=30.0,
    )
    assert binary == str(out_path)
    assert out_path.exists()


def test_failed_compile_still_returns_none(tmp_path, monkeypatch):
    fake_cc = _write_script(tmp_path / "fake_cc", "exit 1\n")
    _patch_compiler(monkeypatch, fake_cc)

    binary = compile_simulator(
        str(tmp_path), str(tmp_path),
        output_path=str(tmp_path / "bin" / "simulator"),
        verbose=False, timeout_s=30.0,
    )
    assert binary is None
