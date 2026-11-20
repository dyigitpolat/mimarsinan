"""[U-HOST3] A copy of the shipped host descriptions and their held store, so a test can correct a shipped description
as a release would (the corrected bytes added to the held store, the earlier bytes kept there) or place a held file
of its own, without touching the package. No description, mode, host or chip is named here: the first read
description's file pair is copied.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from mimarsinan.deployment_record.platform_physics.hosts import registry as REG
from mimarsinan.deployment_record.platform_physics.hosts.description import host_descriptions

NAME = host_descriptions()[0]


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hold(held: Path, data: bytes) -> str:
    """Place ``data`` in the held store under its own digest; return the digest."""
    name = hashlib.sha256(data).hexdigest()
    (Path(held) / f"{name}.json").write_bytes(data)
    return name


def first_row(data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """(the rows mapping holding it, the key): the first row the default mode resolves, its band to be corrected."""
    mode = data["modes"][data["default_mode"]].get("constants") or {}
    rows = {**data.get("constants", {}), **mode}
    key = sorted(k for k in rows if "nominal" in rows[k])[0]
    return (mode if key in mode else data["constants"]), key


def correct(shipped: Path, held: Path, factor: float = 1.01, keep_held: bool = True) -> str:
    """Correct the shipped description's first row (each corner times ``factor``), holding the corrected bytes unless
    ``keep_held`` is False; return the corrected row's key."""
    path = Path(shipped) / f"{NAME}.json"
    data = json.loads(path.read_text())
    rows, key = first_row(data)
    for end in ("low", "nominal", "high"):
        if end in rows[key]:
            rows[key][end] *= factor
    path.write_text(json.dumps(data, indent=1))
    if keep_held:
        shutil.copyfile(path, Path(held) / f"{digest(path)}.json")
    return key


@pytest.fixture
def shipped_copy(tmp_path, monkeypatch):
    """The shipped directory and its held store as copies (the first description's file pair, its bytes held);
    yields (shipped directory, held directory)."""
    shipped, held = tmp_path / "shipped", tmp_path / "shipped" / "held"
    held.mkdir(parents=True)
    source = REG.description_path(NAME)
    for suffix in (".json", ".md"):
        shutil.copyfile(source.with_suffix(suffix), shipped / f"{NAME}{suffix}")
    shutil.copyfile(source, held / f"{digest(source)}.json")
    monkeypatch.setattr(REG, "_SHIPPED", shipped)
    monkeypatch.setattr(REG, "_HELD", held)
    yield shipped, held
