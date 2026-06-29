"""C++ compiler discovery probes the actual standard-library mode."""

from __future__ import annotations

import subprocess

from mimarsinan.common import build_utils


def test_clang_rejected_when_libcxx_probe_fails(monkeypatch):
    calls = []

    def fake_run(cmd, *_, **__):
        calls.append(cmd)
        text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "clang++-20" in text or "clang++-19" in text or "clang++-18" in text:
            if "-stdlib=libc++" in text:
                raise subprocess.CalledProcessError(1, cmd, output="missing cstddef")
            return subprocess.CompletedProcess(cmd, 0, stdout="clang", stderr="")
        if "g++-13" in text:
            return subprocess.CompletedProcess(cmd, 0, stdout="gcc", stderr="")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(build_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(build_utils.shutil, "which", lambda cmd: cmd)

    assert build_utils.find_cpp20_compiler() == ("g++-13", "gcc")
    assert any("-stdlib=libc++" in " ".join(c) for c in calls if isinstance(c, list))


def test_modern_clang_accepted_when_libcxx_probe_passes(monkeypatch):
    def fake_run(cmd, *_, **__):
        text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "clang++-20" in text:
            return subprocess.CompletedProcess(cmd, 0, stdout="clang", stderr="")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(build_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(build_utils.shutil, "which", lambda cmd: cmd)

    assert build_utils.find_cpp20_compiler() == ("clang++-20", "clang")
