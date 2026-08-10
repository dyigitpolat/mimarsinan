"""Read persisted resources from disk, rendering from saved sources on a miss."""

from __future__ import annotations

import logging
from pathlib import Path

from mimarsinan.gui.resources import encode_resource_payload
from mimarsinan.gui.runtime.persistence.resource_paths import resource_disk_path
from mimarsinan.gui.runtime.persistence.resource_sources import load_resource_source
from mimarsinan.gui.runtime.persistence.store import save_resource_to_disk

logger = logging.getLogger("mimarsinan.gui")


def load_resource_from_disk(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    *,
    media_type: str,
    variant: str = "ui",
) -> bytes | None:
    """Bytes for one persisted resource, rendering from its saved source on a miss.

    A run with no monitor attached renders nothing and persists source data
    instead, so the first fetch of a resource is also what produces it. The
    rendered bytes are written back under the normal resource path, which makes
    every later fetch -- and every later attach -- a plain file read.

    ``variant="full"`` serves the on-demand near-native render (cached as a
    sibling ``.full`` file); it needs the persisted SOURCE and returns ``None``
    without one, so callers fall back to the UI-resolution artifact.
    """
    try:
        path = resource_disk_path(working_directory, step_name, kind, rid, media_type)
    except ValueError:
        return None
    if variant == "full":
        return _load_full_variant(
            working_directory, step_name, kind, rid, path,
        )
    if path.is_file():
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None
    return _render_persisted_source(
        working_directory, step_name, kind, rid, media_type=media_type,
    )


def _load_full_variant(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    artifact_path: Path,
) -> bytes | None:
    full_path = artifact_path.with_name(artifact_path.stem + ".full" + artifact_path.suffix)
    if full_path.is_file():
        try:
            with open(full_path, "rb") as f:
                return f.read()
        except OSError:
            return None
    source = load_resource_source(working_directory, step_name, kind, rid)
    if source is None:
        return None
    payload = source.render_full()
    if not isinstance(payload, (bytes, bytearray)):
        return None
    _write_variant_file(full_path, bytes(payload))
    return bytes(payload)


def _write_variant_file(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(payload)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write resource variant %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _render_persisted_source(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    *,
    media_type: str,
) -> bytes | None:
    source = load_resource_source(working_directory, step_name, kind, rid)
    if source is None:
        return None
    payload = source.render()
    encoded = encode_resource_payload(payload, media_type)
    if encoded is None:
        logger.debug(
            "Resource source %s/%s/%s rendered a payload no %s encoder accepts",
            step_name, kind, rid, media_type,
        )
        return None
    save_resource_to_disk(
        working_directory, step_name, kind, rid, encoded, media_type=media_type,
    )
    return encoded


__all__ = ["load_resource_from_disk"]
