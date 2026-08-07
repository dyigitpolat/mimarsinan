"""Ratchet: exits and child launches go through the lifecycle SSOT, never ad hoc."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIFECYCLE = ROOT / "src" / "mimarsinan" / "common" / "lifecycle"
FRONTENDS = (ROOT / "run.py", ROOT / "src" / "main.py")


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _dotted(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _sources(paths) -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in paths if p.is_file()}


class TestFrontendsOwnNoExitMachinery:
    def test_no_frontend_calls_os_exit_directly(self):
        offenders = {
            path.name: _dotted(call.func)
            for path, src in _sources(FRONTENDS).items()
            for call in _calls(ast.parse(src))
            if _dotted(call.func) in ("os._exit", "_exit")
        }
        assert not offenders, (
            "frontends must exit through mimarsinan.common.lifecycle.exit_contract."
            f"exit_process so the reap can never be bypassed: {offenders}"
        )

    def test_no_frontend_installs_its_own_signal_handler(self):
        offenders = {
            path.name: _dotted(call.func)
            for path, src in _sources(FRONTENDS).items()
            for call in _calls(ast.parse(src))
            if _dotted(call.func) in ("signal.signal", "signal.sigaction")
        }
        assert not offenders, (
            "signal handling is the exit_contract's job; a per-frontend handler "
            f"covers one signal on one code path: {offenders}"
        )

    def test_run_py_installs_the_contract(self):
        src = (ROOT / "run.py").read_text(encoding="utf-8")
        assert "install_exit_contract" in src, (
            "run.py must install the exit contract before any child can spawn"
        )


class TestNoEofGatedChildWaits:
    """``subprocess.run(capture_output=..., timeout=...)`` completes on pipe EOF,
    not on the child's exit, so a leaked grandchild pins the caller for the full
    budget. Long-lived children must be launched through ``run_child``."""

    def _violations(self, paths) -> dict[str, int]:
        found: dict[str, int] = {}
        for path, src in _sources(paths).items():
            for call in _calls(ast.parse(src)):
                if _dotted(call.func) not in ("subprocess.run", "run"):
                    continue
                kwargs = {kw.arg for kw in call.keywords}
                piped = "capture_output" in kwargs or "stdout" in kwargs
                if piped and "timeout" in kwargs:
                    found[str(path.relative_to(ROOT))] = call.lineno
        return found

    def test_no_script_pipes_and_times_out_a_child_at_once(self):
        offenders = self._violations(sorted((ROOT / "scripts").glob("*.py")))
        assert not offenders, (
            "these launchers block until every process holding the child's "
            "stdout/stderr closes it, not until the child exits; use "
            f"mimarsinan.common.lifecycle.child_launcher.run_child: {offenders}"
        )

    def test_the_launcher_itself_gates_on_wait_not_communicate(self):
        src = (LIFECYCLE / "child_launcher.py").read_text(encoding="utf-8")
        assert ".communicate(" not in src, (
            "communicate() returns only at pipe EOF; the launcher must gate on "
            "the direct child's wait()"
        )
        assert "start_new_session=True" in src, (
            "without its own session the child's cohort cannot be killed as a unit"
        )
