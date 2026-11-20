"""Seed a new run working directory with pipeline cache from a previous run (edit & continue).

[U-LIF2 r4] A run directory holds one point: a seed that copies a run's entries (its cache, its steps file) writes that
run's own point into the new run with them (``_carry_run_point``), so the new run takes it.

[U-LIF2 r5] Each seed reads the source's point before its first copy (``run_directory_point``): a source whose point
is refused by name leaves the new run without entries."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from mimarsinan.chip_simulation.stored_point import DirectoryPoint, persist_run_point, run_directory_point
from mimarsinan.mapping.platform.target_expansion import door_read
from mimarsinan.gui.runtime.persistence.paths import (
    RESOURCE_SOURCES_DIRNAME,
    RESOURCES_DIRNAME,
)

logger = logging.getLogger("mimarsinan.gui")

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

_STRATEGY_TO_EXT = {
    "basic": ".json",
    "torch_model": ".pt",
    "pickle": ".pickle",
}


def copy_pipeline_cache_from_previous_run(
    generated_root: str,
    previous_run_id: str,
    dest_working_dir: str,
) -> None:
    """Copy ``metadata.json`` and cache blob files from a previous run into *dest_working_dir*.

    *previous_run_id* is the basename of the run directory under *generated_root*.
    """
    if not previous_run_id or not _SAFE_ID_RE.match(previous_run_id):
        logger.warning("Invalid previous_run_id for cache seed: %r", previous_run_id)
        return

    root = Path(generated_root).resolve()
    src = root / previous_run_id
    meta_path = src / "metadata.json"
    if not meta_path.is_file():
        logger.warning("No pipeline cache at %s (skip seeding)", meta_path)
        return
    held = run_directory_point(str(src), door=door_read)  # [U-LIF2 r5] before the first copy: a refusal leaves no entry

    dest = Path(dest_working_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", meta_path, e)
        return

    shutil.copy2(meta_path, dest / "metadata.json")

    for name, meta_val in metadata.items():
        if isinstance(meta_val, (list, tuple)) and len(meta_val) >= 1:
            strategy = meta_val[0]
        else:
            continue
        ext = _STRATEGY_TO_EXT.get(str(strategy))
        if not ext:
            logger.debug("Unknown cache strategy %r for key %r", strategy, name)
            continue
        src_file = src / f"{name}{ext}"
        if src_file.is_file():
            shutil.copy2(src_file, dest / f"{name}{ext}")
        else:
            logger.warning("Cache entry missing expected file %s", src_file)
    _carry_run_point(held, dest)


def copy_steps_json_from_previous_run(
    generated_root: str,
    previous_run_id: str,
    dest_working_dir: str,
) -> None:
    """Copy ``_GUI_STATE/steps.json`` from a previous run so backfill can load full snapshots."""
    if not previous_run_id or not _SAFE_ID_RE.match(previous_run_id):
        return
    src = Path(generated_root).resolve() / previous_run_id / "_GUI_STATE" / "steps.json"
    if not src.is_file():
        logger.debug("No steps.json at %s (skip seeding)", src)
        return
    held = run_directory_point(str(src.parent.parent), door=door_read)  # [U-LIF2 r5] before the first copy
    dest_dir = Path(dest_working_dir).resolve() / "_GUI_STATE"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "steps.json")
    _carry_run_point(held, dest_dir.parent)


def _carry_run_point(held: Optional[DirectoryPoint], dest: Path) -> None:
    """[U-LIF2 r4] Write the point the source run directory held (persisted, else stated by its saved document beside
    its entries; [U-LIF2 r5] read before the first copy) into the run directory its entries were copied to; nothing
    where the source held none."""
    if held is not None:
        persist_run_point(str(dest), held.point)


def copy_resources_from_previous_run(
    generated_root: str,
    previous_run_id: str,
    dest_working_dir: str,
) -> None:
    """Copy a previous run's resources AND resource sources into this run's state.

    Lets the live monitor serve backfilled steps' images without re-running them.
    Both trees are seeded because which one a run left behind depends on whether
    anything was watching it: a headless run persists sources and renders on
    demand, so seeding only the rendered tree would carry nothing forward.
    """
    if not previous_run_id or not _SAFE_ID_RE.match(previous_run_id):
        return
    previous_state = Path(generated_root).resolve() / previous_run_id / "_GUI_STATE"
    dest_state = Path(dest_working_dir).resolve() / "_GUI_STATE"
    for dirname in (RESOURCES_DIRNAME, RESOURCE_SOURCES_DIRNAME):
        src = previous_state / dirname
        if not src.is_dir():
            logger.debug("No %s dir at %s (skip seeding)", dirname, src)
            continue
        try:
            shutil.copytree(src, dest_state / dirname, dirs_exist_ok=True)
        except (OSError, shutil.Error) as e:
            logger.warning(
                "Failed to seed %s from %s → %s: %s", dirname, src, dest_state / dirname, e,
            )
