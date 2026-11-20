"""[U-HOST2] Test-only host descriptions, registered through the registry's own contract.

A test that prices with host numbers other than a read description's declares them as a host description of its own:
the file pair (``<name>.json`` beside ``<name>.md``) written to a temporary directory and registered with
``hosts.admission.register_host_description``, which judges it whole as it judges a shipped one (a published,
datasheet, derived or estimated class, never measured; every energy row's plane stated). The rows a run declares are
then that description's own rows (``declared_rows``), admitted because they equal what the description resolves, and
the description is taken out again when the test ends. Nothing here relabels a row: a value enters only as a
description's row. No description, mode, host or chip is named here.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import pytest

from mimarsinan.deployment_record.platform_physics.hosts.admission import (
    register_host_description, unregister_host_description,
)
from mimarsinan.deployment_record.platform_physics.hosts.registry import description_path
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    ATTRIBUTABLE, ENERGY_PLANE_FIELD, HOST_IDENTITY_FIELD, SYSTEM, is_energy_row,
)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD

#: The one mode of a test description.
MODE = "test"


def identity(name: str) -> str:
    return f"{name} {MODE}"


def _planes(rows: Mapping[str, Any], system: tuple) -> Dict[str, str]:
    return {key: SYSTEM if key in system else ATTRIBUTABLE for key in rows if is_energy_row(key)}


def description_data(name: str, rows: Mapping[str, Mapping[str, Any]], system: tuple = (),
                     link_ends: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """The JSON body of a one-mode test description declaring exactly ``rows`` (``system``: its system-level energy
    rows, every other energy row attributable)."""
    return {"format_version": 1, "name": name, "display_name": f"{name} (a test-only host description)",
            "description_file": f"{name}.md", "default_mode": MODE, "departures": {},
            "energy_attribution": {"rows": _planes(rows, system), "basis": "a test's own statement (test)"},
            **({"link_ends": {"rows": dict(link_ends), "basis": "a test's own statement (test)"}} if link_ends else {}),
            "modes": {MODE: {"source": "a test (test)", "constants": {k: dict(v) for k, v in rows.items()}}}}


def declared_rows(name: str, rows: Mapping[str, Mapping[str, Any]], system: tuple = (),
                  link_ends: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """The override rows a run declares for the test description ``name``: each row as the description declares it,
    naming the host, with the plane and link ends the description states for it."""
    planes = _planes(rows, system)
    return {key: {**dict(row), HOST_IDENTITY_FIELD: identity(name),
                  **({ENERGY_PLANE_FIELD: planes[key]} if key in planes else {}),
                  **({LINK_ENDS_FIELD: dict(link_ends[key])} if link_ends and key in link_ends else {})}
            for key, row in rows.items()}


@contextmanager
def registered(directory: Path, name: str, rows: Mapping[str, Mapping[str, Any]], system: tuple = (),
               link_ends: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Iterator[str]:
    """Write and register the test description ``name`` for the block's duration; yield its identity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(f"# {name}\n\nA test-only host description (test numbers).\n")
    path = directory / f"{name}.json"
    path.write_text(json.dumps(description_data(name, rows, system, link_ends), indent=1))
    register_host_description(path)
    try:
        yield identity(name)
    finally:
        unregister_host_description(name)


@contextmanager
def registered_copy(directory: Path, name: str, edit: Callable[[Dict[str, Any]], Any],
                    as_name: Optional[str] = None) -> Iterator[str]:
    """A read description's file pair with ``edit`` applied to its JSON, registered under its own test name
    (``as_name``, default ``test_<name>``) for the block's duration; yield that name. The description's judge runs at
    registration, so an edit it refuses raises here, by name."""
    source = description_path(name)
    assert source is not None, f"no read host description {name!r}"
    new = as_name or f"test_{name}"
    data = json.loads(source.read_text())
    edit(data)
    data.update(name=new, description_file=f"{new}.md")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{new}.md").write_text((source.parent / f"{name}.md").read_text())
    path = directory / f"{new}.json"
    path.write_text(json.dumps(data, indent=1))
    register_host_description(path)
    try:
        yield new
    finally:
        unregister_host_description(new)


def registered_fixture(name: str, rows: Mapping[str, Mapping[str, Any]], system: tuple = (),
                       link_ends: Optional[Mapping[str, Mapping[str, Any]]] = None, autouse: bool = True):
    """A pytest fixture registering the test description ``name`` around each test of the module that uses it."""

    @pytest.fixture(autouse=autouse, name=f"registered_host_{name}")
    def _registered(tmp_path_factory):
        with registered(tmp_path_factory.mktemp(name), name, rows, system, link_ends) as host:
            yield host

    return _registered
