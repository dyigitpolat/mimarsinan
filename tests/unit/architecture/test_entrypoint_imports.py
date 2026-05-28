"""Guardrails for entrypoint and public-package import surfaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_gui_public_api_matches_exports():
    import mimarsinan.gui as gui_pkg
    import mimarsinan.gui.exports as exports

    for name in exports.__all__:
        assert hasattr(gui_pkg, name), f"mimarsinan.gui missing {name!r} (see gui.exports)"
        assert getattr(gui_pkg, name) is getattr(exports, name)


def test_run_headless_import_block():
    """Imports used by run.py::_run_headless must resolve without running a pipeline."""
    from mimarsinan.gui import GUIHandle, backfill_skipped_steps, to_json_safe
    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.runtime.composite_reporter import CompositeReporter
    from mimarsinan.gui.runtime.persistence import save_run_info, update_run_status
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

    assert callable(to_json_safe)
    assert GUIHandle is not None
    assert callable(backfill_skipped_steps)
    assert DataCollector is not None
    assert CompositeReporter is not None
    assert callable(save_run_info)
    assert callable(update_run_status)
    assert DeploymentPipeline is not None


def test_process_spawn_run_py_path():
    from mimarsinan.gui.runtime.process_spawn import _REPO_ROOT, _RUN_PY

    run_py = Path(_RUN_PY)
    assert run_py.is_file(), _RUN_PY
    assert run_py.name == "run.py"
    assert (_REPO_ROOT / "run.py").resolve() == run_py.resolve()
    assert (_REPO_ROOT / "src" / "mimarsinan").is_dir()


def test_import_path_inventory_resolved():
    script = ROOT / "scripts" / "import_path_inventory.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check-resolved"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def test_import_path_inventory_phase3_paths():
    script = ROOT / "scripts" / "import_path_inventory.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check-phase3-paths"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def test_import_symbols_script():
    script = ROOT / "scripts" / "check_import_symbols.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
