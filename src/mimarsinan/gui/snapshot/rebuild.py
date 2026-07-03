"""Rebuild GUI step snapshots from on-disk pipeline cache for known step types."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from mimarsinan.gui.snapshot.builders import snapshot_sanafe_simulation

logger = logging.getLogger("mimarsinan.gui.snapshot.rebuild")

_SANAFE_RESULTS_FILE = "SANA-FE Simulation.sanafe_simulation_results.pickle"


def rebuild_step_snapshot_from_disk(
    working_directory: str,
    step_name: str,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Return ``(snapshot, snapshot_key_kinds)`` rebuilt from cache, or ``None``."""
    run_dir = Path(working_directory)
    if step_name == "SANA-FE Simulation":
        return _rebuild_sanafe_snapshot(run_dir)
    return None


def _rebuild_sanafe_snapshot(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]] | None:
    pickle_path = run_dir / _SANAFE_RESULTS_FILE
    if not pickle_path.is_file():
        return None
    try:
        with open(pickle_path, "rb") as f:
            report = pickle.load(f)
        snap, _ = snapshot_sanafe_simulation(report)
        snapshot = {"step_name": "SANA-FE Simulation", "sanafe_simulation": snap}
        return snapshot, {"sanafe_simulation": "new"}
    except Exception:
        logger.debug("Failed to rebuild SANA-FE snapshot from %s", pickle_path, exc_info=True)
        return None
