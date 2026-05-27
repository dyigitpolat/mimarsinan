"""Safe on-disk resource path layout under ``_GUI_STATE/resources/``."""

from __future__ import annotations

import re
from pathlib import Path

from mimarsinan.gui.runtime.persistence.paths import RESOURCES_DIRNAME, gui_state_dir

RESOURCE_EXT_FOR_MEDIA_TYPE = {
    "image/png": ".png",
    "application/json": ".json",
}

_SAFE_SEGMENT_CHAR_RE = re.compile(r"[^A-Za-z0-9 ._()+,@=-]")


def sanitize_path_segment(segment: str) -> str:
    if not segment or segment in (".", ".."):
        raise ValueError(f"Invalid path segment: {segment!r}")
    if "/" in segment or "\\" in segment or "\x00" in segment:
        raise ValueError(f"Invalid path segment: {segment!r}")
    sanitized = _SAFE_SEGMENT_CHAR_RE.sub("_", segment)
    if sanitized in (".", "..", ""):
        raise ValueError(f"Invalid path segment: {segment!r}")
    return sanitized


def resource_root(working_directory: str) -> Path:
    return gui_state_dir(working_directory) / RESOURCES_DIRNAME


def resource_disk_path(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    media_type: str,
) -> Path:
    ext = RESOURCE_EXT_FOR_MEDIA_TYPE.get(media_type)
    if ext is None:
        raise ValueError(f"Unsupported resource media_type: {media_type!r}")
    safe_step = sanitize_path_segment(step_name)
    safe_kind = sanitize_path_segment(kind)
    safe_rid_parts = [sanitize_path_segment(p) for p in rid.split("/") if p]
    if not safe_rid_parts:
        raise ValueError(f"Empty rid: {rid!r}")
    safe_rid_parts[-1] = safe_rid_parts[-1] + ext
    return resource_root(working_directory).joinpath(safe_step, safe_kind, *safe_rid_parts)
