"""Management of saved deployment configuration templates."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def get_templates_dir() -> str:
    return os.environ.get("MIMARSINAN_TEMPLATES_DIR", "./templates")


def _validate_id(template_id: str) -> str:
    if not _SAFE_ID_RE.match(template_id):
        raise ValueError(f"Invalid template id: {template_id!r}")
    return template_id


def list_templates() -> list[dict[str, Any]]:
    """List all saved templates (name + basic metadata)."""
    tdir = Path(get_templates_dir())
    if not tdir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for child in sorted(tdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if child.suffix != ".json" or not child.is_file():
            continue
        try:
            with open(child, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        results.append({
            "id": child.stem,
            "name": config.get("experiment_name", child.stem),
            "pipeline_mode": config.get("pipeline_mode", "unknown"),
            "created_at": child.stat().st_mtime,
        })
    return results


def get_template(template_id: str) -> dict[str, Any] | None:
    """Load a template by ID (filename stem)."""
    _validate_id(template_id)
    path = Path(get_templates_dir()) / f"{template_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_template(name: str, config: dict[str, Any]) -> str:
    """Save a config as a named template. Returns the template ID."""
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip())
    if not safe_name:
        safe_name = "template"
    tdir = Path(get_templates_dir())
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{safe_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return safe_name


def delete_template(template_id: str) -> bool:
    """Delete a template by ID. Returns True if deleted."""
    _validate_id(template_id)
    path = Path(get_templates_dir()) / f"{template_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
