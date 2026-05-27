"""Guardrail: legacy shim import paths must not reappear in src/ or tests/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_no_banned_legacy_import_paths():
    script = ROOT / "scripts" / "import_path_inventory.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check-legacy-path"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
