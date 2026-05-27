"""Module size budget script runs clean under default allowlist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_module_budget_strict_allowlist():
    script = ROOT / "scripts" / "check_module_budget.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--strict",
            "--strict-prefix",
            "mapping/packing/",
            "--strict-prefix",
            "mapping/layout/",
            "--strict-prefix",
            "visualization/search_viz/",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
