"""[U-J] A deployment host declared as data: its published power modes, the mode a declared rule selects, and the
host block a run merges as its ``host`` override block (``resolve.merge_override_blocks``).

A description is ``hosts/<name>.json`` beside ``<name>.md`` (the citations and every derivation in prose): per mode
its published facts (power budget, online CPUs, clocks) and its rows, and the rows every mode shares. Each row is a
HOST-group physics constant with its band and class (published, datasheet, derived or estimated; never measured: a
host description describes a machine the toolchain does not run on, and a measurement here is one of the machine
that ran the toolchain). A mode that lacks a row the host terms read refuses by name, with the description's reason.
[targeted M3] A row whose source departs from the source rule (published documentation or peer-reviewed measurements:
a preprint, another module's measurement) is named in the description's ``departures``, and its block row's note
states the departure.
Every row of a block names its host (``HOST_IDENTITY_FIELD``: the description and mode), so the resolver keeps one
host per deployment (``require_one_host``): an identified host replaces the profile's host rows whole. [U-HOST2] A
row naming a host is admitted only as its description's own row or a declared derivation of its rows
(``hosts.admission``, the one admission rule at the loader, judged on the value); a row naming no host, or another
value under a host's name, refuses by name. A description is read from this directory or registered
(``hosts.registry``).
[U-J2] The description states, per energy row (a power, an energy per byte), whether the row's energy is attributable
(the deployment's work) or system-level (the host's own draw, as its idle power): ``energy_attribution``. Every
energy row of its block carries that statement (``ENERGY_PLANE_FIELD``), so the pricer's main energy sums the
attributable terms and the system line carries the whole host (``cost.host_terms``). A system-level power is a floor
the host draws in every state: an attributable power lies at or above it at every corner, else the block refuses by
name, and the power above it takes the cost ends of the difference (``power_above``).
[U-J5 L-2] The description declares, per corner, which ends of the host-chip link its link row counts
(``link_ends``, ``hosts.link_ends``); the block's row carries it, so the link's energy names the chip's end unpriced
only where the row leaves it out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.refusal.format_version import is_version
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration, spec_for
from mimarsinan.deployment_record.platform_physics.hosts.derivation import DERIVATION_FIELD
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD, judged_link_ends
from mimarsinan.deployment_record.platform_physics.hosts.registry import description_paths, name_refusal, read_description
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.units import ENERGY, POWER

#: The field of an override row naming the host it describes (``"<description> <mode>"``).
HOST_IDENTITY_FIELD = "host"
FORMAT_VERSION = 1
#: The rows the host terms read: compute (rate, dispatch), power (active, idle), memory and the host-chip link.
REQUIRED_HOST_ROWS: Tuple[str, ...] = ("p_host", "p_host_idle", "host_macs_per_s", "t_host_op_overhead",
                                       "t_host_mem_per_byte", "e_host_mem_per_byte", "e_link_per_byte",
                                       "t_link_per_byte", "t_link_transfer")
#: The mode rule that takes the description's documented default mode.
DEFAULT_RULE = "default"
#: [U-HOST] The evidence classes a host row is admitted with (``hosts.admission``): a description of a machine the
#: toolchain does not run on, never measured.
ADMITTED_CLASSES = ("published", "datasheet", "derived", "estimated")
#: [U-J2] The field of a host energy row stating whose energy it is: the deployment's work, or the host's own draw.
ENERGY_PLANE_FIELD = "energy_plane"
ATTRIBUTABLE, SYSTEM = "attributable", "system"
ENERGY_PLANES = (ATTRIBUTABLE, SYSTEM)
_CORNERS = ("low", "nominal", "high")


class HostDescriptionError(ToolchainRefusal, KeyError):
    """A host description that is not shipped, or one that does not read."""


class HostModeError(ToolchainRefusal, KeyError):
    """A mode rule that names no mode of the description."""


class HostModeIncompleteError(ToolchainRefusal, ValueError):
    """A mode that lacks a row the host terms read."""


class HostRowAdmissionError(ToolchainRefusal, ValueError):
    """[U-HOST] A host row that no identified host description declares (no host named, a host no description
    ships, a measured row, or rows of two hosts in one physics), refused by name (``hosts.admission``)."""


#: Host rows of two hosts in one deployment (an edge host's beside a measured server row, or two descriptions): the
#: one admission rule's refusal since [U-HOST].
HostBlockMixError = HostRowAdmissionError


@dataclass(frozen=True)
class HostDescription:
    """One shipped host: its modes (facts and rows), the rows every mode shares, and its documented default mode."""

    name: str
    display_name: str
    description_file: str
    default_mode: str
    modes: Mapping[str, Mapping[str, Any]]
    constants: Mapping[str, Mapping[str, Any]]
    #: [targeted M3] Per row whose source departs from the source rule, the departure in words.
    departures: Mapping[str, str]
    #: [U-J2] Per energy row (power or energy per byte) it declares, ``attributable`` or ``system``, and the basis.
    energy_attribution: Mapping[str, str] = field(default_factory=dict)
    energy_attribution_basis: str = ""
    #: [U-J5 L-2] Per link row it declares, the ends of the link the row counts at each corner.
    link_ends: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def host_descriptions() -> Tuple[str, ...]:
    """The read host descriptions, by name: the shipped ones and [U-HOST2] any registered (``hosts.registry``)."""
    return tuple(sorted(description_paths()))


def _judged(name: str, where: str, rows: Mapping[str, Mapping[str, Any]]) -> None:
    for key, row in rows.items():
        if spec_for(key).group != HOST:
            raise HostDescriptionError(f"host description {name!r} ({where}): {key!r} is not a host row")
        if row.get("evidence_kind") not in ADMITTED_CLASSES:
            raise HostDescriptionError(f"host description {name!r} ({where}): {key!r} has class "
                                       f"{row.get('evidence_kind')!r}, not one of {list(ADMITTED_CLASSES)}")
        constant_from_declaration(key, row)


def host_description(name: str) -> HostDescription:
    """The read description ``name`` (``hosts.registry.read_description``), every row judged; unknown, it refuses."""
    found = read_description(name)
    if found is None:
        raise HostDescriptionError(f"no host description {name!r}; read: {list(host_descriptions())}")
    return judged_description(name, found[1], found[0].name, found[0].parent)


def judged_description(name: str, raw: bytes, where: str, beside: Optional[Path]) -> HostDescription:
    """One description's JSON bytes judged whole (its prose file checked ``beside`` it where it is read as a pair;
    [U-HOST3] a held description, read by its digest, is its JSON alone)."""
    data = json.loads(raw)
    if why := name_refusal(name):
        raise HostDescriptionError(f"{where}: {why}")
    if not is_version(data.get("format_version"), FORMAT_VERSION) or data.get("name") != name:
        raise HostDescriptionError(f"{where}: format_version {data.get('format_version')!r} (expected "
                                   f"{FORMAT_VERSION}) or name {data.get('name')!r} does not match its file")
    if beside is not None and not (beside / str(data.get("description_file", ""))).is_file():
        raise HostDescriptionError(f"host description {name!r}: its description file "
                                   f"{data.get('description_file')!r} is missing beside it")
    modes = dict(data["modes"])
    if data.get("default_mode") not in modes:
        raise HostDescriptionError(f"host description {name!r}: default mode {data.get('default_mode')!r} is not one "
                                   f"of its modes {list(modes)}")
    _judged(name, "every mode", data.get("constants") or {})
    for mode, facts in modes.items():
        _judged(name, f"mode {mode}", facts.get("constants") or {})
    departures = {str(key): str(text) for key, text in (data.get("departures") or {}).items()}
    rows = set(data.get("constants") or {}).union(*(facts.get("constants") or {} for facts in modes.values()))
    if set(departures) - rows:
        raise HostDescriptionError(f"host description {name!r}: departures {sorted(set(departures) - rows)} name no "
                                   "row it declares")
    attribution = data.get("energy_attribution") or {}
    planes = {str(key): str(plane) for key, plane in (attribution.get("rows") or {}).items()}
    _judged_attribution(name, planes, rows)
    ends = {str(key): value for key, value in ((data.get(LINK_ENDS_FIELD) or {}).get("rows") or {}).items()}
    if stray := sorted(set(ends) - rows):
        raise HostDescriptionError(f"host description {name!r} ({LINK_ENDS_FIELD}): {stray} name no row it declares")
    for key, value in ends.items():
        try:
            judged_link_ends(key, value, f"host description {name!r}")
        except ValueError as why:
            raise HostDescriptionError(str(why)) from None
    return HostDescription(name=name, display_name=str(data["display_name"]),
                           description_file=str(data["description_file"]), default_mode=str(data["default_mode"]),
                           modes=modes, constants=dict(data.get("constants") or {}), departures=departures,
                           energy_attribution=planes, energy_attribution_basis=str(attribution.get("basis", "")),
                           link_ends=ends)


def is_energy_row(key: str) -> bool:
    """[U-J2] A host row whose value prices energy: a power (over a time) or an energy per byte."""
    spec = spec_for(key)
    return spec.group == HOST and spec.dimension in (POWER, ENERGY)


def _judged_attribution(name: str, planes: Mapping[str, str], rows: set) -> None:
    """[U-J2] Every energy row the description declares is stated once, as attributable or system-level; a statement
    of a row it does not declare, of a row that prices no energy, or of another plane refuses by name."""
    where = f"host description {name!r} (energy_attribution)"
    if bad := sorted(key for key, plane in planes.items() if plane not in ENERGY_PLANES):
        raise HostDescriptionError(f"{where}: {bad} state {sorted({planes[k] for k in bad})}, not one of "
                                   f"{list(ENERGY_PLANES)}")
    if stray := sorted(set(planes) - rows):
        raise HostDescriptionError(f"{where}: {stray} name no row it declares")
    if not_energy := sorted(key for key in planes if not is_energy_row(key)):
        raise HostDescriptionError(f"{where}: {not_energy} price no energy (a host power or energy per byte)")
    if unstated := sorted(key for key in rows if is_energy_row(key) and key not in planes):
        raise HostDescriptionError(f"{where}: the energy rows {unstated} state neither {ATTRIBUTABLE!r} nor "
                                   f"{SYSTEM!r}: whose energy each is must be declared, never assumed")


def energy_plane_refusal(key: str, plane: Any) -> Optional[str]:
    """[U-J2] Why a declaration's ``energy_plane`` does not stand, or None: it is stated only on a host energy row,
    as one of ``ENERGY_PLANES``."""
    if not is_energy_row(key):
        return f"{key}: {ENERGY_PLANE_FIELD} is stated only on a host row that prices energy (a power or an energy " \
               "per byte)"
    if plane not in ENERGY_PLANES:
        return f"{key}: {ENERGY_PLANE_FIELD} {plane!r} is not one of {list(ENERGY_PLANES)}"
    return None


def power_above(power: Band, floor: Band, basis: str) -> Tuple[Optional[Band], Optional[str]]:
    """[U-J2] ``power`` above a system-level ``floor`` (both module powers, the floor inside every reading of the
    power), or why not: below the floor at a corner (one setup: the two rows' same end read together).
    [targeted M1] The difference of two bands takes its cost ends, never its corners: the high end is the power's
    high less the floor's low, the low end the power's low less the floor's high, never below zero (the host draws at
    least its floor while it works), the nominal one setup's. Each end bounds the difference across the two rows'
    sources, so the band is ordered by construction."""
    corners = [getattr(power, end) - getattr(floor, end) for end in _CORNERS]
    tol = 1e-12 * max(abs(getattr(power, end)) for end in _CORNERS)
    worst = min(range(len(corners)), key=corners.__getitem__)
    if corners[worst] < -tol:
        return None, f"lies below the floor at the {_CORNERS[worst]} corner ({corners[worst]:.6g} W)"
    return Band(max(0.0, power.low - floor.high), max(0.0, power.nominal - floor.nominal),
                max(0.0, power.high - floor.low), basis), None


def _floor_refusal(name: str, mode: str, rows: Mapping[str, Mapping[str, Any]], planes: Mapping[str, str]) -> None:
    """[U-J2] Each attributable power of the mode lies at or above its system-level floor at every corner."""
    powers = [key for key in rows if spec_for(key).dimension == POWER and key in planes]
    floors = [key for key in powers if planes[key] == SYSTEM]
    for key in (k for k in powers if planes[k] == ATTRIBUTABLE):
        for floor in floors:
            _band, why = power_above(constant_from_declaration(key, rows[key]).band,
                                     constant_from_declaration(floor, rows[floor]).band, f"{key} - {floor}")
            if why:
                raise HostDescriptionError(f"host {name!r} mode {mode}: {key} above its system-level floor {floor} "
                                           f"{why}; an attributable power is charged above the floor")


def select_mode(description: HostDescription, rule: str = DEFAULT_RULE) -> str:
    """The mode ``rule`` selects: ``default`` takes the documented default mode, a mode's name takes that mode."""
    if rule == DEFAULT_RULE:
        return description.default_mode
    if rule in description.modes:
        return rule
    raise HostModeError(f"host {description.name!r}: mode rule {rule!r} names no mode; the modes are "
                        f"{sorted(description.modes, key=str)} (or {DEFAULT_RULE!r}: {description.default_mode})")


def host_description_block(name: str, mode_rule: str = DEFAULT_RULE) -> Dict[str, Dict[str, Any]]:
    """The host block of ``name`` in the mode ``mode_rule`` selects: every row the host terms read, each naming its
    host; a mode lacking one refuses by name with the description's reason."""
    description = host_description(name)
    mode = select_mode(description, mode_rule)
    facts = description.modes[mode]
    rows = {**description.constants, **(facts.get("constants") or {})}
    missing = [key for key in REQUIRED_HOST_ROWS if key not in rows]
    if missing:
        reasons = facts.get("refused") or {}
        raise HostModeIncompleteError(
            f"host {name!r} mode {mode} declares no {', '.join(missing)}: "
            + "; ".join(f"{key}: {reasons.get(key, 'not declared')}" for key in missing))
    identity = f"{name} {mode}"
    planes = description.energy_attribution
    _floor_refusal(name, mode, {key: rows[key] for key in REQUIRED_HOST_ROWS}, planes)
    return {key: {**dict(rows[key]), HOST_IDENTITY_FIELD: identity, **_departure(rows[key], description, key),
                  **({ENERGY_PLANE_FIELD: planes[key]} if key in planes else {}),
                  **({LINK_ENDS_FIELD: description.link_ends[key]} if key in description.link_ends else {})}
            for key in REQUIRED_HOST_ROWS}


def _departure(row: Mapping[str, Any], description: HostDescription, key: str) -> Dict[str, str]:
    """[targeted M3] The row's note with its named departure from the source rule, where the description names one."""
    text = description.departures.get(key)
    return {} if text is None else {"note": " ".join(part for part in (str(row.get("note") or ""),
                                                                      f"Departure from the source rule: {text}") if part)}


def describes_host(physics: Any) -> bool:
    """[targeted M1] Whether ``physics`` declares a host's whole model, every row the host terms read
    (``REQUIRED_HOST_ROWS``), as a host description's block does: a host the toolchain does not run on, which prices
    its own time; a partial host block (a server's measured calibration) does not."""
    return physics is not None and physics.declares_all(REQUIRED_HOST_ROWS)


def require_one_host(overrides: Mapping[str, Mapping[str, Any]],
                     profile_constants: Mapping[str, Any]) -> Tuple[str, ...]:
    """One host per deployment, judged on a run's overrides: the profile's host rows an identified host replaces
    whole (empty without one). [U-HOST] Which rows are admitted (an identified description's, never measured, one
    host) is the one admission rule's (``hosts.admission``), judged on the result."""
    if not any(row.get(HOST_IDENTITY_FIELD) for key, row in overrides.items() if spec_for(key).group == HOST):
        return ()
    return tuple(key for key in profile_constants if spec_for(key).group == HOST and key not in overrides)


def host_attach_block(name: str, mode_rule: str = DEFAULT_RULE) -> Dict[str, Dict[str, Any]]:
    """The programming channel's host stage: the chip's programming port is fed over the host-chip link, so a
    payload byte moves no faster than the slower of the chip's ``t_program_per_byte`` and the link's
    ``t_link_per_byte`` (``combine: slower``, the backing store's rule); merged beside the host block."""
    link = host_description_block(name, mode_rule)["t_link_per_byte"]
    fields = ("low", "nominal", "high", "unit", "citation", HOST_IDENTITY_FIELD)  # class, words and note: below
    row = {key: link[key] for key in fields if key in link}
    row["note"] = ("the host-chip link as the programming channel's host stage (its t_link_per_byte, derived as: "
                   + str(link.get("derivation") or "") + "); " + str(link.get("note", "")))
    # [U-HOST2] the stage is declared as what it is, a derivation of the host's link row (``hosts.derivation``);
    # [U-HOST3] its derivation text states that declaration, the link row's own arithmetic kept in the note
    row.update(evidence_kind="derived", derivation="the slower stage of 1.0 x t_link_per_byte, the host's link row")
    return {"t_program_per_byte": {**row, "combine": "slower", DERIVATION_FIELD: {"of": {"t_link_per_byte": 1.0}}}}
