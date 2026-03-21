"""Seed a new run working directory with pipeline cache from a previous run (edit & continue).

The pipeline stores ``metadata.json`` and per-entry files (``.json`` / ``.pt`` / ``.pickle``)
in the run working directory. A new run starts with an empty directory; without copying
these files, ``run_from(start_step=...)`` fails with missing cache requirements.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

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
    If the source has no ``metadata.json`` or ``previous_run_id`` is unsafe, log and return.
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


def copy_steps_json_from_previous_run(
    generated_root: str,
    previous_run_id: str,
    dest_working_dir: str,
) -> None:
    """Copy ``_GUI_STATE/steps.json`` from a previous run so backfill can load full snapshots.

    Without this, ``_backfill_skipped_steps`` falls back to ``build_step_snapshot`` only,
    which may omit richer metrics/snapshots from the original run. The headless process
    then writes a merged ``steps.json`` after backfill (see ``_persist_skipped_steps_to_steps_json``).
    """
    if not previous_run_id or not _SAFE_ID_RE.match(previous_run_id):
        return
    src = Path(generated_root).resolve() / previous_run_id / "_GUI_STATE" / "steps.json"
    if not src.is_file():
        logger.debug("No steps.json at %s (skip seeding)", src)
        return
    dest_dir = Path(dest_working_dir).resolve() / "_GUI_STATE"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "steps.json")
