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
host per deployment (``require_one_host``): an identified host replaces the profile's host rows whole, and a host
row that names no host (a measured server row) beside it refuses by name."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration, spec_for
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST

#: The field of an override row naming the host it describes (``"<description> <mode>"``).
HOST_IDENTITY_FIELD = "host"
FORMAT_VERSION = 1
#: The rows the host terms read: compute (rate, dispatch), power (active, idle), memory and the host-chip link.
REQUIRED_HOST_ROWS: Tuple[str, ...] = ("p_host", "p_host_idle", "host_macs_per_s", "t_host_op_overhead",
                                       "t_host_mem_per_byte", "e_host_mem_per_byte", "e_link_per_byte",
                                       "t_link_per_byte", "t_link_transfer")
#: The mode rule that takes the description's documented default mode.
DEFAULT_RULE = "default"
_DIR = Path(__file__).resolve().parent
_CLASSES = ("published", "datasheet", "derived", "estimated")


class HostDescriptionError(ToolchainRefusal, KeyError):
    """A host description that is not shipped, or one that does not read."""


class HostModeError(ToolchainRefusal, KeyError):
    """A mode rule that names no mode of the description."""


class HostModeIncompleteError(ToolchainRefusal, ValueError):
    """A mode that lacks a row the host terms read."""


class HostBlockMixError(ToolchainRefusal, ValueError):
    """Host rows of two hosts in one deployment (an edge host's beside a measured server row, or two descriptions)."""


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


def host_descriptions() -> Tuple[str, ...]:
    """The shipped host descriptions, by name."""
    return tuple(sorted(path.stem for path in _DIR.glob("*.json")))


def _judged(name: str, where: str, rows: Mapping[str, Mapping[str, Any]]) -> None:
    for key, row in rows.items():
        if spec_for(key).group != HOST:
            raise HostDescriptionError(f"host description {name!r} ({where}): {key!r} is not a host row")
        if row.get("evidence_kind") not in _CLASSES:
            raise HostDescriptionError(f"host description {name!r} ({where}): {key!r} has class "
                                       f"{row.get('evidence_kind')!r}, not one of {list(_CLASSES)}")
        constant_from_declaration(key, row)


def host_description(name: str) -> HostDescription:
    """The shipped description ``name``, every row judged; an unknown name refuses, listing the shipped ones."""
    path = _DIR / f"{name}.json"
    if not path.is_file():
        raise HostDescriptionError(f"no host description {name!r}; shipped: {list(host_descriptions())}")
    data = json.loads(path.read_text())
    if data.get("format_version") != FORMAT_VERSION or data.get("name") != name:
        raise HostDescriptionError(f"{path.name}: format_version {data.get('format_version')!r} (expected "
                                   f"{FORMAT_VERSION}) or name {data.get('name')!r} does not match its file")
    if not (_DIR / str(data.get("description_file", ""))).is_file():
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
    return HostDescription(name=name, display_name=str(data["display_name"]),
                           description_file=str(data["description_file"]), default_mode=str(data["default_mode"]),
                           modes=modes, constants=dict(data.get("constants") or {}), departures=departures)


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
    return {key: {**dict(rows[key]), HOST_IDENTITY_FIELD: identity, **_departure(rows[key], description, key)}
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
    (empty without one, so a server block keeps the profile's), or a refusal by name of host rows of two hosts."""
    host_rows = {key: row for key, row in overrides.items() if spec_for(key).group == HOST}
    named = {key: row.get(HOST_IDENTITY_FIELD) for key, row in host_rows.items()}
    hosts = sorted({host for host in named.values() if host})
    if not hosts:
        return ()
    if len(hosts) > 1:
        raise HostBlockMixError(f"one deployment has one host: its host rows name {hosts}")
    unnamed = sorted(key for key, host in named.items() if not host)
    if unnamed:
        raise HostBlockMixError(
            f"host rows {unnamed} name no host beside the rows of host {hosts[0]!r}: one deployment has one host, "
            "and a row measured on another machine (a server's calibration) is not this host's")
    measured = sorted(key for key, row in host_rows.items() if row.get("evidence_kind") == "measured")
    if measured:
        raise HostBlockMixError(f"host rows {measured} of host {hosts[0]!r} are measured: a host description's rows "
                                "are published, derived or estimated, never a measurement of the machine running "
                                "the toolchain")
    return tuple(key for key in profile_constants if spec_for(key).group == HOST and key not in overrides)


def host_attach_block(name: str, mode_rule: str = DEFAULT_RULE) -> Dict[str, Dict[str, Any]]:
    """The programming channel's host stage: the chip's programming port is fed over the host-chip link, so a
    payload byte moves no faster than the slower of the chip's ``t_program_per_byte`` and the link's
    ``t_link_per_byte`` (``combine: slower``, the backing store's rule); merged beside the host block."""
    link = host_description_block(name, mode_rule)["t_link_per_byte"]
    fields = ("low", "nominal", "high", "unit", "evidence_kind", "citation", "derivation", "note", HOST_IDENTITY_FIELD)
    row = {key: link[key] for key in fields if key in link}
    row["note"] = ("the host-chip link as the programming channel's host stage (its t_link_per_byte); "
                   + str(link.get("note", "")))
    return {"t_program_per_byte": {**row, "combine": "slower"}}
