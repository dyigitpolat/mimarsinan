"""Declared quantizer state (WI-5a): boundary grids and readout ranges from a sha-pinned map keyed by FX node name."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.activations.lane_quantization import LanePoolModule
from mimarsinan.models.nn.activations.value_quantizer import (
    ENTRY_ROUNDINGS,
    DeclaredBoundaryGrid,
    ValueGridQuantizer,
)

QUANTIZER_STATE_FORMAT = "mimarsinan.quantizer_state.v1"
#: The attribute a readout quantizer carries when its range came from a declared map: the map's sha256.
DECLARED_STATE_ATTRIBUTE = "declared_state_sha256"
#: The attribute a lane quantizer carries when its grid's offset came from a declared map (WI-5e).
DECLARED_OFFSET_ATTRIBUTE = "declared_offset_sha256"
_PERCEPTRON_FIELDS = frozenset({"readout_range", "readout_offset", "entry"})
_ENTRY_FIELDS = frozenset({"scale", "offset", "rounding", "codes"})
#: A lane end that is no perceptron (a pool lane, WI-5e), keyed by its op's FX node name.
_LANE_FIELDS = frozenset({"readout_range", "readout_offset"})


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


def _state(document: Any, sha256: str) -> QuantizerState:
    if not isinstance(document, Mapping) or document.get("format") != QUANTIZER_STATE_FORMAT:
        raise QuantizerStateError(f"quantizer_state document is not of format {QUANTIZER_STATE_FORMAT!r}")
    perceptrons = document.get("perceptrons")
    lanes = document.get("lanes", {})
    if (not isinstance(perceptrons, Mapping) or not isinstance(lanes, Mapping)
            or set(document) - {"format", "perceptrons", "lanes", "source"}):
        raise QuantizerStateError("quantizer_state document holds format, perceptrons, optional lanes and an optional "
                                  "source only")
    ranges: Dict[str, float] = {}
    entries: Dict[str, EntryGridDeclaration] = {}
    offsets: Dict[str, float] = {}
    for name, body in perceptrons.items():
        if not isinstance(body, Mapping) or not body or set(body) - _PERCEPTRON_FIELDS:
            raise QuantizerStateError(f"quantizer_state {name!r} must declare readout_range, entry or both, got "
                                      f"{body!r}")
        if "readout_range" in body:
            ranges[str(name)] = _positive(body["readout_range"], f"{name!r}.readout_range")
        _offset(str(name), body, offsets)
        if "entry" in body:
            entries[str(name)] = _entry(str(name), body["entry"])
    lane_ranges: Dict[str, float] = {}
    lane_offsets: Dict[str, float] = {}
    for name, body in lanes.items():
        if not isinstance(body, Mapping) or "readout_range" not in body or set(body) - _LANE_FIELDS:
            raise QuantizerStateError(f"quantizer_state lane {name!r} must declare readout_range (and an optional "
                                      f"readout_offset), got {body!r}")
        lane_ranges[str(name)] = _positive(body["readout_range"], f"lane {name!r}.readout_range")
        _offset(str(name), body, lane_offsets)
    return QuantizerState(sha256=sha256, readout_ranges=ranges, entry_grids=entries, readout_offsets=offsets,
                          lane_ranges=lane_ranges, lane_offsets=lane_offsets)


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


def _named_perceptrons(model_repr):
    """(FX node name, perceptron) per perceptron owner: a conv's mapper carries the name, a linear's perceptron."""
    order, _, _ = model_repr._ensure_exec_graph()
    for node in order:
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None:
            own = getattr(node, "name", None)
            yield (own if isinstance(own, str) else str(perceptron.name)), perceptron


def perceptron_fx_names(model_repr) -> Dict[str, Any]:
    """{FX node name: perceptron}, refused by name when two perceptrons carry one name."""
    names: Dict[str, Any] = {}
    for name, perceptron in _named_perceptrons(model_repr):
        if names.get(name, perceptron) is not perceptron:
            raise QuantizerStateError(f"two perceptrons carry the FX node name {name!r}")
        names[name] = perceptron
    return names


def _resolve(model, state: QuantizerState, names: Iterable[str]) -> Dict[str, Any]:
    """{name: perceptron} of ``names``, after every name the map holds is checked against the network."""
    known = perceptron_fx_names(model.get_mapper_repr())
    unknown = sorted((set(state.readout_ranges) | set(state.entry_grids)) - set(known))
    if unknown:
        raise QuantizerStateError(f"quantizer_state names {unknown}, which are no perceptron's FX node name")
    return {name: known[name] for name in names}


def declare_boundary_grids(model, entries: list, quantizers: list, state: QuantizerState, *,
                           unsigned_dac: bool = False) -> int:
    """Install each declared entry grid on its entry's boundary quantizer; the others stay calibrated (a lane
    chip's DAC, ``unsigned_dac``, also reads unsigned 2^b codes)."""
    targets = _resolve(model, state, state.entry_grids)
    by_perceptron = {id(p): q for p, q in zip(entries, quantizers)}
    for name, perceptron in targets.items():
        quantizer = by_perceptron.get(id(perceptron))
        if not isinstance(quantizer, ValueGridQuantizer):
            raise QuantizerStateError(f"quantizer_state declares an entry grid for {name!r}, which reads no host "
                                      f"boundary (not a segment entry)")
        entry = state.entry_grids[name]
        low, high = entry.codes if entry.codes is not None else (None, None)
        quantizer.declare(DeclaredBoundaryGrid(
            scale=entry.scale, bits=quantizer.activation_bits, offset=entry.offset, rounding=entry.rounding,
            code_min=low, code_max=high, state_sha256=state.sha256, unsigned_dac=unsigned_dac,
        ))
    return len(targets)


def declare_readout_ranges(model, perceptrons: list, quantizers: list, state: QuantizerState) -> int:
    """Set each declared readout range on its perceptron's readout quantizer, tagged with the map's digest."""
    targets = _resolve(model, state, state.readout_ranges)
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    for name, perceptron in targets.items():
        quantizer = by_perceptron.get(id(perceptron))
        if quantizer is None:
            raise QuantizerStateError(f"quantizer_state declares a readout range for {name!r}, which carries no "
                                      f"readout quantizer")
        quantizer.calibrate(state.readout_ranges[name])
        setattr(quantizer, DECLARED_STATE_ATTRIBUTE, state.sha256)
    return len(targets)


def _tag(carried: Dict[str, Dict[str, str]], kind: str, name: str, module, attribute: str) -> None:
    tag = getattr(module, attribute, None)
    if tag is not None:
        carried[kind][name] = str(tag)


def _carried(model) -> Dict[str, Dict[str, str]]:
    """{"entry" | "readout_range" | "readout_offset" | "lane": {FX name: the digest its grid, range or offset was
    declared under}} (a lane: a pool lane's quantizer, WI-5e)."""
    carried: Dict[str, Dict[str, str]] = {"entry": {}, "readout_range": {}, "readout_offset": {}, "lane": {}}
    for name, perceptron in _named_perceptrons(model.get_mapper_repr()):
        wire = getattr(perceptron, "input_activation", None)
        for module in wire.modules() if wire is not None else ():
            grid = module.grid if isinstance(module, ValueGridQuantizer) else None
            if isinstance(grid, DeclaredBoundaryGrid):
                carried["entry"][name] = grid.state_sha256
        _tag(carried, "readout_range", name, getattr(perceptron, "scaler", None), DECLARED_STATE_ATTRIBUTE)
        _tag(carried, "readout_offset", name, getattr(perceptron, "scaler", None), DECLARED_OFFSET_ATTRIBUTE)
    order, _, _ = model.get_mapper_repr()._ensure_exec_graph()
    for node in order:
        if isinstance(getattr(node, "module", None), LanePoolModule):
            _tag(carried, "lane", str(node.name), node.module.quantizer, DECLARED_STATE_ATTRIBUTE)
    return carried


def verify_declared_state(model, declaration: Optional[Mapping[str, Any]]) -> None:
    """Refuse a program whose grids and ranges are not exactly the declared map's (a stale or unread map)."""
    carried = _carried(model)
    state = load_quantizer_state(declaration)
    if state is None:
        stale = sorted(set().union(*carried.values()))
        if stale:
            raise QuantizerStateError(f"perceptrons {stale} carry a declared quantizer state but no quantizer_state "
                                      f"is declared: rerun Boundary Quantization and Readout Law")
        return
    wanted = {"entry": set(state.entry_grids), "readout_range": set(state.readout_ranges),
              "readout_offset": set(state.readout_offsets), "lane": set(state.lane_ranges)}
    for kind, names in wanted.items():
        stale = sorted(n for n, digest in carried[kind].items() if digest != state.sha256)
        if stale:
            raise QuantizerStateError(f"perceptrons {stale} carry a {kind} from a stale quantizer state, not "
                                      f"{state.sha256}: rerun Boundary Quantization and Readout Law")
        unread = sorted(names - set(carried[kind]))
        if unread:
            raise QuantizerStateError(f"quantizer_state declares a {kind} for {unread}, which no quantizer reads "
                                      f"(Boundary Quantization needs activation_bits; Readout Law readout_law; an "
                                      f"offset or a lane needs onchip_arithmetic)")
