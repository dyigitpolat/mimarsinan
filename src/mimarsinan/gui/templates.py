"""Management of saved deployment configuration templates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mimarsinan.common.env import templates_dir
from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def name_and_deployment_from_post_body(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Split POST ``/api/templates`` JSON into (filename stem, deployment config).

    Accepts either ``{"name": str, "config": <deployment>}`` or a flat deployment object.
    """
    if not isinstance(body, dict):
        return "template", {}
    inner = body.get("config")
    if isinstance(inner, dict):
        name = (str(body.get("name") or "").strip() or str(inner.get("experiment_name") or "template"))
        return name, inner
    name = str(body.get("experiment_name") or "template")
    return name, body


def get_templates_dir() -> str:
    return templates_dir()


def _validate_id(template_id: str) -> str:
    if not _SAFE_ID_RE.match(template_id):
        raise ValueError(f"Invalid template id: {template_id!r}")
    return template_id


def _iter_template_files() -> list[tuple[Path, str]]:
    """(path, group) for every ``.json`` at the top level (group "") and exactly
    one level deep (group = subdirectory name). One level only: subdirectories
    are the deployment-mode example groups (tier_0..tier_3), not a tree."""
    tdir = Path(get_templates_dir())
    if not tdir.is_dir():
        return []
    def _configs(directory: Path, group: str) -> list[tuple[Path, str]]:
        # manifest.json is a generator index (tier groups), not a template.
        return [(p, group) for p in directory.iterdir()
                if p.is_file() and p.suffix == ".json" and p.name != "manifest.json"]

    found: list[tuple[Path, str]] = _configs(tdir, "")
    for child in tdir.iterdir():
        if child.is_dir():
            found += _configs(child, child.name)
    return found


def list_templates() -> list[dict[str, Any]]:
    """List all saved templates (name + metadata + group), newest first.

    Flat top-level templates carry group ""; templates one level deep carry
    their subdirectory name as ``group`` (the UI renders one section per group).
    """
    results: list[dict[str, Any]] = []
    for path, group in sorted(
        _iter_template_files(), key=lambda pg: pg[0].stat().st_mtime, reverse=True
    ):
        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(config, dict):
            continue
        results.append({
            "id": path.stem,
            "name": config.get("experiment_name", path.stem),
            "pipeline_mode": config.get("pipeline_mode", "unknown"),
            "group": group,
            "created_at": path.stat().st_mtime,
        })
    return results


def _resolve_template_path(template_id: str) -> Path | None:
    """The file backing ``template_id``: the flat file if present, else the
    first same-stem file in a subdirectory. Flat user templates shadow a
    same-stem example (searched first)."""
    _validate_id(template_id)
    tdir = Path(get_templates_dir())
    flat = tdir / f"{template_id}.json"
    if flat.exists():
        return flat
    for path, _group in _iter_template_files():
        if path.stem == template_id:
            return path
    return None


def get_template(template_id: str) -> dict[str, Any] | None:
    """Load a template by ID (filename stem), searching flat then subdirectories."""
    path = _resolve_template_path(template_id)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_template(name: str, config: dict[str, Any]) -> str:
    """Save a config as a named template. Returns the template ID."""
    display_name = name.strip()
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", display_name)
    if not safe_name:
        safe_name = "template"
    out = build_deployment_config_from_state(config)
    out["experiment_name"] = display_name or "template"
    tdir = Path(get_templates_dir())
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{safe_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return safe_name


def delete_template(template_id: str) -> bool:
    """Delete a template by ID (flat or in a subdirectory). Returns True if deleted."""
    path = _resolve_template_path(template_id)
    if path is not None:
        path.unlink()
        return True
    return False
