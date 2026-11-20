"""[U-LIF2 r3] The point a stored run resolved, read back as it was written.

A reader of a stored run, record, ledger row or cached entry takes the point that was resolved when it was written and
never resolves a saved document again under today's entry data (the declared set and default of ``window_firing_axis``):
once the default moves, a re-resolution would read an earlier run into a point that no run executed.

A run persists its resolved point beside its document (``_RUN_CONFIG/point.json``, written by the session once the run's
plan is resolved). The run directory is the place every reader of a stored run already opens; the point is written when
the run starts, so a run stopped before its record step holds it too; and it holds the whole point (mode, schedule,
variant, soma law, window firing, execution extensions), where the sealed deployment record holds only the window
firing and a cell key, and only once its record step ran. A stored run that persisted none (every run made before runs
persisted it) states the historical point: its document's own deployment parameters, read as the reader of record read
them, with the window firing it declared, else ``HISTORICAL_WINDOW_FIRING`` (every toolchain that persisted no point
resolved a silent document to it), a named holder of the ratchet t02c. A saved document a reader still derives (the
display of a run directory) is derived at its stored point: its activation axes the point's
(``document_at_stored_point``) and the window firing's entry data the run's value (``derived_at_stored_point``).

[U-LIF2 r4] A run directory holds one point, at the writer as at the readers. The point a directory holds
(``run_directory_point``) is the point it persisted, else, where it already holds a saved document and any step or cache
entry (``RUN_ENTRY_FILES``), the historical point that document states, else none. A run in a directory that holds a
point takes it before its pipeline is built (``run_at_directory_point``: a resume, a rerun in place from its first step
and a run continued from a seeded cache alike): a document silent on the window firing derives at the stored point, and
a document that declares another value, or a stored value today's declared set no longer offers, is refused by name
with the remedy. The seed that copies a run's entries into a new run writes that run's point with them.

[U-LIF2 r5] A run directory's states are a finite list with one outcome each, closed by construction: unused (no entry,
or a document alone), persisted, historical, or refused by name with the rule and the remedy (a point that cannot be
read: a point file outside its format, or a saved document that is no object of deployment parameters or whose point
does not resolve; entries beside no point and no saved document).

[U-LIF2 r6] A stored document is read as its run's door read it. The one reader of a run's document for its point
(``door_document``) reads it through the door every reader hands in (``DocumentDoor``: the session's own first step,
the target expanded, nothing written) before the core semantics are judged and the platform section is laid over the
sections: a run whose target carried its core semantics or its soma keys is stated at the point its door resolved. A
target that no longer expands is a document that cannot be read, refused by name (``DirectoryPointRefusal``) at every
reader of a stored run's point (``stored_run_point``, and ``run_directory_point`` through it)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from mimarsinan.chip_simulation.activation_axes import axes_from_legacy
from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
from mimarsinan.chip_simulation.spiking_point import (
    HISTORICAL_WINDOW_FIRING,
    SpikingPoint,
    point_from_record,
    point_record,
    stated_point,
    window_firing_apart,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.window_firing import (
    WINDOW_FIRING_KEY,
    declared_window_firings,
    stored_window_firing,
    window_firing_applies,
    window_firing_declaration,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.refusal.format_version import exact_version

RUN_CONFIG_DIRECTORY = "_RUN_CONFIG"
RUN_DOCUMENT_FILENAME = "config.json"
RUN_POINT_FILENAME = "point.json"
RUN_POINT_FORMAT_VERSION = 1
#: The derivation-owned twins of the activation axes a document written before the axes existed declares.
_LEGACY_TWINS = ("spiking_mode", "ttfs_cycle_schedule")
#: [U-LIF2 r4] The files a run directory's entries live in: the pipeline cache's index and the run's steps file.
RUN_ENTRY_FILES = ("metadata.json", os.path.join("_GUI_STATE", "steps.json"))
#: [U-LIF2 r6] The door a stored document is read through, handed in by every reader: the session's own first step
#: (``mapping.platform.target_expansion.door_read``: the target expanded, nothing written), which sits in a layer above
#: this one (it reads the registry, which imports this package), so the expansion is handed in, never imported here.
DocumentDoor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class DirectoryPointRefusal(ToolchainRefusal, ValueError):
    """A run that would put a second point into a run directory, refused by name with the remedy."""


@dataclass(frozen=True)
class DirectoryPoint:
    """The point a run directory holds (``point`` ``None``: a value-domain run) and whether the directory persisted it
    (else its saved document states it beside its stored entries)."""

    run_dir: str
    point: Optional[SpikingPoint]
    persisted: bool

    def source(self) -> str:
        return ("persisted in its _RUN_CONFIG/point.json" if self.persisted else
                "stated by its saved document beside its stored entries, which persisted no point")


def run_point_path(run_dir: str) -> str:
    return os.path.join(run_dir, RUN_CONFIG_DIRECTORY, RUN_POINT_FILENAME)


def persist_run_point(run_dir: str, point: Optional[SpikingPoint]) -> str:
    """Write the point a run resolved beside its document (``None``: a value-domain run, which has no spiking point);
    return the written path."""
    path = run_point_path(run_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stored = {"format_version": RUN_POINT_FORMAT_VERSION,
              "spiking_point": None if point is None else point_record(point)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_run_document(run_dir: str) -> Optional[Dict[str, Any]]:
    """The document a run saved (``_RUN_CONFIG/config.json``), ``None`` when the directory holds none: the one reader of
    a run directory's document."""
    path = os.path.join(run_dir, RUN_CONFIG_DIRECTORY, RUN_DOCUMENT_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def stored_run_point(run_dir: str, document: Optional[Mapping[str, Any]], *,
                     door: DocumentDoor) -> Optional[SpikingPoint]:
    """The point the run in ``run_dir`` resolved (``None`` for a value-domain run): the point it persisted, read back
    without any resolution, else the historical point its saved ``document`` states, read through ``door`` as its door
    read it (``door_document``). Never a resolution of the saved document under today's entry data. [U-LIF2 r6] A point
    or a document that cannot be read (a target that no longer expands included) is refused by name for every reader."""
    try:
        path = run_point_path(run_dir)
        if not os.path.isfile(path):
            return _historical_point(document, door)
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        if exact_version(stored.get("format_version"), path) != RUN_POINT_FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {stored.get('format_version')!r} != {RUN_POINT_FORMAT_VERSION}; "
                             "the format changed: migrate explicitly, never read silently")
        record = stored["spiking_point"]
        return None if record is None else point_from_record(record)
    except DirectoryPointRefusal:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as cause:
        raise _unreadable(run_dir, cause) from cause


def _unreadable(run_dir: str, cause: BaseException) -> DirectoryPointRefusal:
    """The named refusal of a run directory whose point or saved document cannot be read."""
    return DirectoryPointRefusal(
        f"the run directory {run_dir} holds a point that cannot be read ({type(cause).__name__}: {cause}): a run "
        "directory holds one point. Run in a new run directory that reuses none of its entries, or repair its "
        f"{RUN_CONFIG_DIRECTORY}/{RUN_POINT_FILENAME} or {RUN_DOCUMENT_FILENAME}.")


def door_document(document: Any, door: DocumentDoor) -> Mapping[str, Any]:
    """[U-LIF2 r6] A stored run's saved document as its run's door read it: in the form the door admits
    (``_statable``), read through ``door`` (the session's own first step: its platform target expanded into both
    sections, nothing written). A target that no longer expands raises its own refusal (a ``ValueError``): a document
    that cannot be read."""
    return door(_statable(document))


def _historical_point(document: Any, door: DocumentDoor) -> Optional[SpikingPoint]:
    """The point a stored run that persisted none states: its document read as its door read it (``door_document``),
    its deployment parameters with the window firing it declared (a declaration resolves to itself or the run is
    refused), else the historical one; ``None`` for the value domain. [U-LIF2 r4, m-1] Its soma law is read by the one
    constructor over both sections, the platform's declarations laid over the deployment parameters as the run lays
    them (a register's width and sign are platform keys)."""
    read = door_document(document, door)
    declared, params = window_firing_apart(read["deployment_parameters"])
    if is_mvm_core_semantics(resolve_core_semantics(params)):
        return None
    platform = read.get("platform_constraints")
    laid = {key: value for key, value in platform.items() if value is not None} if isinstance(platform, Mapping) else {}
    return stated_point(params, window_firing=HISTORICAL_WINDOW_FIRING if declared is None else declared,
                        soma_law=SomaLaw.resolve({**params, **laid}))


def run_directory_point(run_dir: str, *, door: DocumentDoor) -> Optional[DirectoryPoint]:
    """[U-LIF2 r4] The point the run directory ``run_dir`` already holds: the point it persisted; else, where it holds a
    saved document and any step or cache entry, the historical point that document states; else ``None`` (an unused
    directory, or a document alone: a run refused before any step). [U-LIF2 r5] Entries beside no point and no saved
    document, and a point that cannot be read, are refused by name (``DirectoryPointRefusal``). [U-LIF2 r6] A saved
    document is read through ``door``."""
    try:
        if os.path.isfile(run_point_path(run_dir)):
            return DirectoryPoint(run_dir, stored_run_point(run_dir, None, door=door), persisted=True)
        present = [entry for entry in RUN_ENTRY_FILES if os.path.isfile(os.path.join(run_dir, entry))]
        if not present:
            return None
        document = read_run_document(run_dir)
        if document is None:
            raise DirectoryPointRefusal(
                f"the run directory {run_dir} holds entries ({', '.join(present)}) but no point and no saved document, "
                "so the point they were made at is unknown: a run directory holds one point. Run in a new run "
                f"directory that reuses none of its entries, or restore its {RUN_CONFIG_DIRECTORY}.")
        return DirectoryPoint(run_dir, stored_run_point(run_dir, document, door=door), persisted=False)
    except DirectoryPointRefusal:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as cause:
        raise _unreadable(run_dir, cause) from cause


def _statable(document: Any) -> Mapping[str, Any]:
    """[U-LIF2 r5] A saved document in the form the historical statement reads: an object whose deployment parameters
    are an object and whose platform section, where it is not null, is an object (the session's door refuses every
    other form, so no run wrote one); any other form is a document that cannot be read (``TypeError``)."""
    platform = document.get("platform_constraints") if isinstance(document, Mapping) else None
    if (not isinstance(document, Mapping) or not isinstance(document.get("deployment_parameters"), Mapping)
            or not (platform is None or isinstance(platform, Mapping))):
        raise TypeError(f"{RUN_DOCUMENT_FILENAME} is no object with an object of deployment_parameters and a "
                        "platform_constraints object or null")
    return document


@contextmanager
def run_at_directory_point(deployment_parameters: Mapping[str, Any], held: Optional[DirectoryPoint]) -> Iterator[None]:
    """[U-LIF2 r4] Derive a run in a directory that holds a point (``held``; ``None``: an unused directory, today's
    data) at that point: where the run's document is at the window firing's point, a declaration of another value, or
    a stored value today's declared set does not offer, is refused by name with the remedy; a silent document derives
    at the stored value (``derived_at_stored_point``). Entered only by the session (the ratchet t02c)."""
    stored = None if held is None or held.point is None else held.point.window_firing
    if held is not None and stored is not None and window_firing_applies(deployment_parameters):
        declared = window_firing_declaration(deployment_parameters)
        where = f"the run directory {held.run_dir} holds a run at {WINDOW_FIRING_KEY}={stored!r} ({held.source()})"
        remedy = "Run in a new run directory that reuses none of its entries"
        if declared is not None and declared != stored:
            raise DirectoryPointRefusal(f"{WINDOW_FIRING_KEY}={declared!r} is declared, and {where}: a run directory "
                                        f"holds one point. {remedy}, or declare {WINDOW_FIRING_KEY}={stored!r}.")
        offered = declared_window_firings()
        if stored not in offered:
            raise DirectoryPointRefusal(
                f"{where}, which the declared set {{{', '.join(repr(value) for value in offered)}}} does not offer: a "
                f"run directory holds one point, so a run in it cannot take another. {remedy}.")
    with derived_at_stored_point(None if held is None else held.point):
        yield


def document_at_stored_point(document: Mapping[str, Any], point: Optional[SpikingPoint]) -> Dict[str, Any]:
    """A stored run's saved document with its activation axes the stored point's (family and variant): a document
    written before those axes existed names only the legacy twins (``spiking_mode``, ``ttfs_cycle_schedule``), keys the
    derivation owns and an emitted document drops, so a reader deriving it would otherwise take today's default variant
    for the variant the run resolved. Unchanged for a value-domain run (``None``)."""
    if point is None:
        return dict(document)
    params = {key: value for key, value in dict(document.get("deployment_parameters") or {}).items()
              if key not in _LEGACY_TWINS}
    family = axes_from_legacy(point.spiking_mode, point.ttfs_cycle_schedule)[0]
    return {**document, "deployment_parameters": {**params, "spiking_family": family,
                                                  "spiking_variant": point.spiking_variant}}


@contextmanager
def derived_at_stored_point(point: Optional[SpikingPoint]) -> Iterator[None]:
    """Derive a stored run's saved document at the point its run resolved: for the scope the window firing's entry data
    read as that run's value, so the derivation resolves it and nothing else (``None``: no window firing to hold)."""
    with stored_window_firing(None if point is None else point.window_firing):
        yield
