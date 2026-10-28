"""The Lava executor is isolated from the process's multiprocessing state.

lava's message infrastructure calls ``multiprocessing.set_start_method('fork')`` when it is
first imported, which raises ``RuntimeError: context has already been set`` in a process whose
default context another caller already established (an xdist worker after any test that used
multiprocessing, a pipeline after a pool). The toolchain's start-method guard
(``_subtractive_lif_cls``) must therefore precede the first lava import on every path, the
executor's own graph build included. lava's runtime also runs only under ``fork`` actors, so a
process whose default start method is another one is refused by name before any graph is built,
and a runtime that raises before it starts must release its actors instead of hanging in
``stop()``.

Each behavioral case runs in a fresh interpreter that sets its own multiprocessing state, never
by a global ``set_start_method`` in the test process, and the child's whole process group is
killed if it outlives its wall cap.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
_CHILD_WALL_S = 180

# The child prints one JSON line: the executor's spikes for one input neuron at rate 1.0, T = 4, or the error.
_CHILD_PRELUDE = """
import json, multiprocessing as mp, traceback
import numpy as np
"""
_CHILD_RUN = """
out = {}
try:
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.lava_loihi.core_lava import LavaCoreMixin

    class Host(LavaCoreMixin):
        def __init__(self):
            self.T = 4
            self._behavior = NeuralBehaviorConfig.for_lava({"thresholding_mode": "<="})
            self.thresholding_mode = "<="

    spikes = Host()._run_core_lava(
        weights=np.array([[1.0]], dtype=np.float32), threshold=1.0, hardware_bias=None,
        input_spikes=np.ones((1, 4), dtype=np.float32))
    out.update(ok=True, spikes=np.asarray(spikes).reshape(-1).tolist())
except Exception as exc:
    out.update(ok=False, error=f"{type(exc).__name__}: {exc}")
out["start_method"] = mp.get_start_method(allow_none=True)
out["live_children"] = [p.pid for p in mp.active_children()]
print("CHILD_RESULT " + json.dumps(out), flush=True)
"""


def _run_child(setup: str) -> dict:
    """Run ``setup`` then the executor in a fresh interpreter; kill its process group past the wall cap."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC.parent), env.get("PYTHONPATH", "")])
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_PRELUDE + setup + _CHILD_RUN],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True,
    )
    try:
        output, _ = child.communicate(timeout=_CHILD_WALL_S)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        output, _ = child.communicate()
        pytest.fail(f"the executor did not return within {_CHILD_WALL_S} s (a hang); child output tail: {output[-2000:]}")
    lines = [line for line in output.splitlines() if line.startswith("CHILD_RESULT ")]
    assert lines, f"child exited {child.returncode} without a result: {output[-2000:]}"
    return json.loads(lines[-1][len("CHILD_RESULT "):])


@pytest.mark.slow
def test_the_executor_runs_where_the_start_method_is_already_established(lava_graph_slot):
    """An earlier ``get_start_method()`` (the state an xdist worker is left in) must not break the first lava import."""
    result = _run_child("mp.get_start_method()\n")
    assert result.get("ok"), result
    assert result["spikes"] == [1.0, 1.0, 1.0, 1.0], result
    assert result["start_method"] == "fork", result


@pytest.mark.slow
def test_a_process_whose_default_start_method_is_not_fork_is_refused_by_name(lava_graph_slot):
    """Under ``spawn`` actors lava's runtime cannot run; the executor refuses before any graph is built."""
    result = _run_child("mp.set_start_method('spawn')\n")
    assert not result.get("ok"), result
    error = result["error"]
    assert error.startswith("RuntimeError: "), error
    assert "start method" in error and "'spawn'" in error and "'fork'" in error, error
    assert result["live_children"] == [], result


@pytest.mark.slow
def test_a_runtime_that_raises_before_it_starts_releases_its_actors(lava_graph_slot):
    """A failed runtime start (here a thread that cannot start, as under a full task cap) surfaces as its own
    error with every actor released, instead of a hang in ``stop()`` joining actors that never got STOP."""
    setup = (
        "from lava.magma.runtime.runtime import Runtime\n"
        "def _fail(self):\n"
        "    raise RuntimeError(\"can't start new thread\")\n"
        "Runtime._start_ports = _fail\n"
    )
    result = _run_child("from mimarsinan.chip_simulation.lava_loihi.core_lava import _subtractive_lif_cls\n"
                        "_subtractive_lif_cls()\n" + setup)
    assert not result.get("ok"), result
    assert result["error"] == "RuntimeError: can't start new thread", result
    assert result["live_children"] == [], result


def _lava_imports(tree: ast.AST) -> list[ast.AST]:
    return [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "lava")
        or (isinstance(node, ast.Import) and any(a.name.split(".")[0] == "lava" for a in node.names))
    ]


def _enclosing_function(tree: ast.AST, target: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and target in ast.walk(node):
            return node  # type: ignore[return-value]
    return None


def test_every_lava_import_sits_behind_the_start_method_guard():
    """lava is imported in exactly two places: the process-model module (top level, so Lava's model scan finds it;
    loaded only by ``_subtractive_lif_cls``, which installs the guard first) and the executor's one accessor, which
    calls the guard before its first lava import."""
    process_model = SRC / "chip_simulation" / "subtractive_lif.py"
    core = SRC / "chip_simulation" / "lava_loihi" / "core_lava.py"
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _lava_imports(tree):
            if path == process_model:
                continue
            fn = _enclosing_function(tree, node)
            if path == core and fn is not None and fn.name == "_lava_executor_api":
                continue
            offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "mimarsinan.chip_simulation.subtractive_lif":
                fn = _enclosing_function(tree, node)
                if not (path == core and fn is not None and fn.name == "_subtractive_lif_cls"):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} (the process-model module)")
    assert not offenders, f"lava imported outside the guarded accessors: {offenders}"

    tree = ast.parse(core.read_text())
    accessor = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_lava_executor_api"
    )
    guard_line = min(
        n.lineno for n in ast.walk(accessor)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_subtractive_lif_cls"
    )
    assert all(guard_line < n.lineno for n in _lava_imports(accessor)), "the guard must precede every lava import"

    graph_builder = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_core_lava"
    )
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_lava_executor_api"
        for n in ast.walk(graph_builder)
    ), "the executor's graph build must take its lava classes from the guarded accessor"
