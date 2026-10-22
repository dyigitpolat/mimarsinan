"""The model-level golden and its packer import nothing from the toolchain (plan section 10.4, acceptance 4).

A fresh interpreter, with the toolchain importable, imports every golden module and runs one instance per
arm and every witness (so lazily imported code runs too); no module named ``mimarsinan*`` (nor torch,
which the toolchain pulls in) may appear in ``sys.modules``. An AST scan alone would miss a lazy or
indirect import, so it only backs the subprocess check.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "scripts" / "x3a_golden"

_PROBE = r"""
import json, pkgutil, sys
sys.path[:0] = [{scripts!r}, {src!r}]
import x3a_golden
mods = sorted(m.name for m in pkgutil.iter_modules(x3a_golden.__path__))
for name in mods:
    __import__("x3a_golden." + name)
from x3a_golden.screen import run_instance
from x3a_golden.hypotheses import LABELS
from x3a_golden.witnesses import all_witnesses
for arm in ("all_on",) + LABELS:
    run_instance(arm, 0)
all_witnesses()
import importlib.util
reachable = importlib.util.find_spec("mimarsinan") is not None
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("mimarsinan", "torch"))
print(json.dumps({{"modules": mods, "reachable": reachable, "loaded": loaded}}))
"""


def test_a_fresh_interpreter_loads_no_toolchain_module():
    code = _PROBE.format(scripts=str(ROOT / "scripts"), src=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["reachable"], "the toolchain must be importable, or the check proves nothing"
    assert len(report["modules"]) >= 10, report["modules"]
    assert report["loaded"] == [], report["loaded"]


def test_no_golden_source_names_the_toolchain():
    paths = sorted(GOLDEN.glob("*.py"))
    assert len(paths) >= 10, paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] in ("mimarsinan", "torch") for n in names), (path.name, names)
