"""Declared quantizer state (WI-5a), its schema and loader: the sha-pinned map keyed by FX node name, every section
judged at load ([U-BRANGE2 (d)]: the operand banks', the consumer groups' and the trained-grid inventory's sections, a
trained grid without a site refused by name). ``quantizer_state`` installs the declarations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.activations.value_quantizer import ENTRY_ROUNDINGS

QUANTIZER_STATE_FORMAT = "mimarsinan.quantizer_state.v1"
#: The attribute a readout quantizer carries when its range came from a declared map: the map's sha256.
DECLARED_STATE_ATTRIBUTE = "declared_state_sha256"
#: The attribute a lane quantizer carries when its grid's offset came from a declared map (WI-5e).
DECLARED_OFFSET_ATTRIBUTE = "declared_offset_sha256"
_PERCEPTRON_FIELDS = frozenset({"readout_range", "readout_offset", "entry", "groups"})
_ENTRY_FIELDS = frozenset({"scale", "offset", "rounding", "codes"})
#: A lane end that is no perceptron (a pool lane, WI-5e), keyed by its op's FX node name.
_LANE_FIELDS = frozenset({"readout_range", "readout_offset"})
#: [U-BRANGE2 (d)] An operand bank perceptron (N2), keyed by its own name: its host-fed entry grid and its readout range.
_BANK_FIELDS = frozenset({"entry", "readout_range"})
#: [U-BRANGE2 (d)] One consumer group's declared range, keyed by its readers (``consumer_group_readers``' names).
_GROUP_FIELDS = frozenset({"readers", "range"})


class QuantizerStateError(ToolchainRefusal, ValueError):
    """A declared quantizer state that is unreadable, malformed, foreign to the network or stale."""


@dataclass(frozen=True)
class EntryGridDeclaration:
    """One segment entry's declared grid: its scale (``step = scale / levels``; on a lane chip an unsigned code range
    is the unsigned lane grid, ``step = scale / (2^b - 1)``), offset, rounding and codes."""

    scale: float
    offset: float = 0.0
    rounding: str = "nearest"
    codes: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class QuantizerState:
    """The map's content, keyed by FX node name, and the sha256 of its bytes. Readout offsets (an offset lane grid,
    WI-5e) and the ranges of lane ends that are no perceptron (``lanes``) are optional."""

    sha256: str
    readout_ranges: Mapping[str, float]
    entry_grids: Mapping[str, EntryGridDeclaration]
    readout_offsets: Mapping[str, float] = MappingProxyType({})
    lane_ranges: Mapping[str, float] = MappingProxyType({})
    lane_offsets: Mapping[str, float] = MappingProxyType({})
    #: [U-BRANGE2 (d)] {perceptron FX name: {readers: range}}, {bank name: grid}, {bank name: range} and the
    #: checkpoint's trained-grid inventory {grid name: ("<kind>:<site key>", ...)}.
    group_ranges: Mapping[str, Mapping[Tuple[Tuple[str, str], ...], float]] = MappingProxyType({})
    bank_entry_grids: Mapping[str, EntryGridDeclaration] = MappingProxyType({})
    bank_readout_ranges: Mapping[str, float] = MappingProxyType({})
    trained: Mapping[str, Tuple[str, ...]] = MappingProxyType({})


@dataclass(frozen=True)
class SiteKind:
    """[U-BRANGE2 stage 1e] One kind of declared site: the ``QuantizerState`` field its map section loads into
    (keyed by the site's key, the field its installer reads), the owner class the key names (``perceptron`` by FX
    node name, ``bank`` an operand bank by its name, ``lane`` a pool lane by its FX node name) and the tag its
    installed quantizer carries (``grid``: the declared entry grid's ``state_sha256``; ``range``, ``offset``: the
    readout or lane quantizer's ``DECLARED_STATE_ATTRIBUTE``, ``DECLARED_OFFSET_ATTRIBUTE``)."""

    field: str
    owner: str
    tag: str


#: [U-BRANGE2 stage 1e] The one inventory of declared site kinds: the map's sections load into these fields, the
#: hooks' installers read them, the trained-grid inventory names its sites by these kinds, and Soft Core Mapping's
#: stale-state check (``verify_declared_state``) reads every kind from here. No reader lists a kind by hand.
DECLARED_SITES: Mapping[str, SiteKind] = MappingProxyType({
    "entry": SiteKind("entry_grids", "perceptron", "grid"),
    "readout": SiteKind("readout_ranges", "perceptron", "range"),
    "readout_offset": SiteKind("readout_offsets", "perceptron", "offset"),
    "group": SiteKind("group_ranges", "perceptron", "range"),
    "bank_entry": SiteKind("bank_entry_grids", "bank", "grid"),
    "bank_readout": SiteKind("bank_readout_ranges", "bank", "range"),
    "lane": SiteKind("lane_ranges", "lane", "range"),
    "lane_offset": SiteKind("lane_offsets", "lane", "offset"),
})
#: [U-BRANGE2 (d)] The kinds a trained grid of the checkpoint can be declared at: every kind of the inventory.
SITE_KINDS = tuple(DECLARED_SITES)


def declared_sites(state: QuantizerState) -> Dict[str, frozenset]:
    """[U-BRANGE2 stage 1e] {kind: the site keys the map declares of it}, for every kind of the inventory."""
    return {kind: frozenset(getattr(state, spec.field)) for kind, spec in DECLARED_SITES.items()}


def quantizer_state_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise QuantizerStateError(f"quantizer_state {where} must be a finite number above 0, got {value!r}")
    return float(value)


def _entry(name: str, body: Any) -> EntryGridDeclaration:
    if not isinstance(body, Mapping) or set(body) - _ENTRY_FIELDS or "scale" not in body:
        raise QuantizerStateError(f"quantizer_state {name!r}.entry must hold scale and only "
                                  f"{sorted(_ENTRY_FIELDS)}, got {body!r}")
    offset = body.get("offset", 0.0)
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise QuantizerStateError(f"quantizer_state {name!r}.entry.offset must be a finite number, got {offset!r}")
    rounding = body.get("rounding", "nearest")
    if rounding not in ENTRY_ROUNDINGS:
        raise QuantizerStateError(f"quantizer_state {name!r}.entry.rounding {rounding!r} is not one of "
                                  f"{list(ENTRY_ROUNDINGS)}")
    codes = body.get("codes")
    if codes is not None:
        if (not isinstance(codes, (list, tuple)) or len(codes) != 2
                or not all(isinstance(c, int) and not isinstance(c, bool) for c in codes) or codes[0] >= codes[1]):
            raise QuantizerStateError(f"quantizer_state {name!r}.entry.codes must be [low, high] integers with "
                                      f"low below high, got {codes!r}")
        codes = (int(codes[0]), int(codes[1]))
    return EntryGridDeclaration(scale=_positive(body["scale"], f"{name!r}.entry.scale"), offset=float(offset),
                                rounding=str(rounding), codes=codes)


def _offset(name: str, body: Mapping[str, Any], offsets: Dict[str, float]) -> None:
    if "readout_offset" not in body:
        return
    value = body["readout_offset"]
    if "readout_range" not in body:
        raise QuantizerStateError(f"quantizer_state {name!r} declares a readout_offset without its readout_range")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QuantizerStateError(f"quantizer_state {name!r}.readout_offset must be a finite number, got {value!r}")
    offsets[name] = float(value)


def _groups(name: str, body: Any) -> Dict[Tuple[Tuple[str, str], ...], float]:
    """[U-BRANGE2 (d)] A perceptron's declared consumer-group ranges, each keyed by its readers' names."""
    if not isinstance(body, list) or not body:
        raise QuantizerStateError(f"quantizer_state {name!r}.groups must be a non-empty list, got {body!r}")
    out: Dict[Tuple[Tuple[str, str], ...], float] = {}
    for group in body:
        readers = group.get("readers") if isinstance(group, Mapping) else None
        if (not isinstance(group, Mapping) or set(group) != _GROUP_FIELDS or not isinstance(readers, list)
                or not readers or not all(isinstance(r, list) and len(r) == 2 for r in readers)):
            raise QuantizerStateError(f"quantizer_state {name!r}.groups entries hold readers ([[bank, role], ...]) and "
                                      f"range only, got {group!r}")
        key = tuple(sorted((str(r[0]), str(r[1])) for r in readers))
        if key in out:
            raise QuantizerStateError(f"quantizer_state {name!r} declares the group read by {list(key)} twice")
        out[key] = _positive(group["range"], f"{name!r}.groups[{list(key)}].range")
    return out


def _site_sections(document: Mapping[str, Any]):
    """[U-BRANGE2 (d)] The banks section and the trained-grid inventory. A trained grid the map places at no site is
    refused by name: a grid a checkpoint was trained under is declared, never recalibrated."""
    banks = document.get("banks", {})
    trained = document.get("trained", {})
    if not isinstance(banks, Mapping) or not isinstance(trained, Mapping):
        raise QuantizerStateError("quantizer_state banks and trained are mappings")
    bank_entries: Dict[str, EntryGridDeclaration] = {}
    bank_ranges: Dict[str, float] = {}
    for name, body in banks.items():
        if not isinstance(body, Mapping) or not body or set(body) - _BANK_FIELDS:
            raise QuantizerStateError(f"quantizer_state bank {name!r} must declare entry, readout_range or both, got "
                                      f"{body!r}")
        if "entry" in body:
            bank_entries[str(name)] = _entry(str(name), body["entry"])
        if "readout_range" in body:
            bank_ranges[str(name)] = _positive(body["readout_range"], f"bank {name!r}.readout_range")
    placed: Dict[str, Tuple[str, ...]] = {}
    unplaced = sorted(str(g) for g, site in trained.items() if site is None or site == [])
    if unplaced:
        raise QuantizerStateError(f"quantizer_state's trained-grid inventory places {unplaced} at no site: a grid the "
                                  f"checkpoint was trained under is declared at its site, never recalibrated")
    for grid, sites in trained.items():
        sites = [sites] if isinstance(sites, str) else sites
        for site in sites if isinstance(sites, list) else [sites]:
            if not isinstance(site, str) or ":" not in site or site.split(":", 1)[0] not in SITE_KINDS:
                raise QuantizerStateError(f"quantizer_state trained grid {grid!r} names site {site!r}, not "
                                          f"'<kind>:<key>' with kind one of {list(SITE_KINDS)}")
        placed[str(grid)] = tuple(str(s) for s in sites)
    return bank_entries, bank_ranges, placed


def _check_inventory(state: QuantizerState) -> None:
    """[U-BRANGE2 (d)] Every site the inventory places a trained grid at is declared in the map: the grid is read at
    its site, not left to calibration ([U-BRANGE2 stage 1e] the sites by kind from ``declared_sites``)."""
    declared = declared_sites(state)
    missing = sorted(f"{grid} -> {site}" for grid, sites in state.trained.items() for site in sites
                     if site.split(":", 1)[1] not in declared[site.split(":", 1)[0]])
    if missing:
        raise QuantizerStateError(f"quantizer_state's trained-grid inventory places grids at sites the map does not "
                                  f"declare (they would be recalibrated): {missing}")


def _state(document: Any, sha256: str) -> QuantizerState:
    if not isinstance(document, Mapping) or document.get("format") != QUANTIZER_STATE_FORMAT:
        raise QuantizerStateError(f"quantizer_state document is not of format {QUANTIZER_STATE_FORMAT!r}")
    perceptrons = document.get("perceptrons")
    lanes = document.get("lanes", {})
    if (not isinstance(perceptrons, Mapping) or not isinstance(lanes, Mapping)
            or set(document) - {"format", "perceptrons", "lanes", "source", "banks", "trained"}):
        raise QuantizerStateError("quantizer_state document holds format, perceptrons, optional lanes, banks and "
                                  "trained, and an optional source only")
    ranges: Dict[str, float] = {}
    entries: Dict[str, EntryGridDeclaration] = {}
    offsets: Dict[str, float] = {}
    groups: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}
    for name, body in perceptrons.items():
        if not isinstance(body, Mapping) or not body or set(body) - _PERCEPTRON_FIELDS:
            raise QuantizerStateError(f"quantizer_state {name!r} must declare readout_range, entry, groups or a "
                                      f"combination, got {body!r}")
        if "readout_range" in body and "groups" in body:  # [U-BRANGE2 stage 1e] one site, one declared source
            raise QuantizerStateError(f"quantizer_state {name!r} declares a readout_range beside its groups: a "
                                      f"grouped readout's range is the largest of its groups, so the one site "
                                      f"would have two declared sources (declare the groups alone)")
        if "readout_range" in body:
            ranges[str(name)] = _positive(body["readout_range"], f"{name!r}.readout_range")
        _offset(str(name), body, offsets)
        if "entry" in body:
            entries[str(name)] = _entry(str(name), body["entry"])
        if "groups" in body:
            groups[str(name)] = _groups(str(name), body["groups"])
    bank_entries, bank_ranges, trained = _site_sections(document)
    lane_ranges: Dict[str, float] = {}
    lane_offsets: Dict[str, float] = {}
    for name, body in lanes.items():
        if not isinstance(body, Mapping) or "readout_range" not in body or set(body) - _LANE_FIELDS:
            raise QuantizerStateError(f"quantizer_state lane {name!r} must declare readout_range (and an optional "
                                      f"readout_offset), got {body!r}")
        lane_ranges[str(name)] = _positive(body["readout_range"], f"lane {name!r}.readout_range")
        _offset(str(name), body, lane_offsets)
    state = QuantizerState(sha256=sha256, readout_ranges=ranges, entry_grids=entries, readout_offsets=offsets,
                           lane_ranges=lane_ranges, lane_offsets=lane_offsets, group_ranges=groups,
                           bank_entry_grids=bank_entries, bank_readout_ranges=bank_ranges, trained=trained)
    _check_inventory(state)
    return state


def load_quantizer_state(declaration: Optional[Mapping[str, Any]]) -> Optional[QuantizerState]:
    """The declared map, after its bytes are checked against the declared sha256; ``None`` when undeclared."""
    if declaration is None:
        return None
    path = os.path.expanduser(str(declaration["file"]))
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as error:
        raise QuantizerStateError(f"quantizer_state path {path!r} cannot be read: {error}") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != declaration["sha256"]:
        raise QuantizerStateError(f"quantizer_state {path!r} has sha256 {digest}, not the declared "
                                  f"{declaration['sha256']}")
    try:
        document = json.loads(data.decode("utf-8"))
    except ValueError as error:
        raise QuantizerStateError(f"quantizer_state {path!r} is not JSON: {error}") from error
    return _state(document, digest)

