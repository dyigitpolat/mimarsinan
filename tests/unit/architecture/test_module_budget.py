"""Module size budget script runs clean under global strict mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_module_budget_strict():
    script = ROOT / "scripts" / "check_module_budget.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
