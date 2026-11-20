"""Declared quantizer state (WI-5a): boundary grids and readout ranges from a sha-pinned map keyed by FX node name,
installed after each step's calibration (``quantizer_state_schema`` reads and judges the map; ``quantizer_state_banks``
installs the operand banks' and consumer groups' sites)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.mapping.mappers.operand.operand_mapper import operand_perceptrons
from mimarsinan.mapping.support.quantizer_state_schema import (
    DECLARED_OFFSET_ATTRIBUTE,
    DECLARED_SITES,
    DECLARED_STATE_ATTRIBUTE,
    QUANTIZER_STATE_FORMAT,
    SITE_KINDS,
    EntryGridDeclaration,
    QuantizerState,
    QuantizerStateError,
    SiteKind,
    declared_sites,
    load_quantizer_state,
    quantizer_state_sha256,
)
from mimarsinan.models.nn.activations.lane_quantization import LanePoolModule
from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, ValueGridQuantizer

__all__ = [
    "DECLARED_OFFSET_ATTRIBUTE", "DECLARED_SITES", "DECLARED_STATE_ATTRIBUTE", "QUANTIZER_STATE_FORMAT",
    "SITE_KINDS", "SITE_TAGS", "EntryGridDeclaration", "QuantizerState", "QuantizerStateError", "SiteKind",
    "declare_boundary_grids", "declare_readout_ranges", "declared_sites", "load_quantizer_state",
    "perceptron_fx_names", "quantizer_state_sha256", "site_owners", "verify_declared_state",
]


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


def site_owners(model_repr) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    """[U-BRANGE2 stage 1e] Every holder of a declared site, by the inventory's owner classes: {"perceptron": {FX
    node name: (its input wire, its readout quantizer)}, "bank": {operand bank name: (its input wire, its readout
    quantizer)}, "lane": {pool lane's FX node name: (None, its lane quantizer)}} (the banks the installers and
    Soft Core Mapping's stamp read, ``operand_perceptrons``)."""
    order, _, _ = model_repr._ensure_exec_graph()

    def held(owner) -> Tuple[Any, Any]:
        return getattr(owner, "input_activation", None), getattr(owner, "scaler", None)

    return {"perceptron": {name: held(p) for name, p in _named_perceptrons(model_repr)},
            "bank": {str(p.name): held(p) for p in operand_perceptrons(model_repr)},
            "lane": {str(node.name): (None, node.module.quantizer) for node in order
                     if isinstance(getattr(node, "module", None), LanePoolModule)}}


def _grid_tag(wire, _quantizer) -> Optional[str]:
    """The digest of the declared entry grid on ``wire`` (the last of its modules' grids), or None."""
    tag = None
    for module in wire.modules() if wire is not None else ():
        grid = module.grid if isinstance(module, ValueGridQuantizer) else None
        if isinstance(grid, DeclaredBoundaryGrid):
            tag = grid.state_sha256
    return tag


#: [U-BRANGE2 stage 1e] How each tag of the inventory is read from a site's holder (its input wire, its quantizer).
SITE_TAGS: Mapping[str, Callable[[Any, Any], Optional[str]]] = MappingProxyType({
    "grid": _grid_tag,
    "range": lambda _wire, quantizer: getattr(quantizer, DECLARED_STATE_ATTRIBUTE, None),
    "offset": lambda _wire, quantizer: getattr(quantizer, DECLARED_OFFSET_ATTRIBUTE, None),
})


def _carried(model) -> Dict[str, Dict[str, str]]:
    """[U-BRANGE2 stage 1e] {kind: {site key: the digest its grid, range or offset was declared under}}, for every
    kind of the inventory (``DECLARED_SITES``), each read from its owners by its tag (``SITE_TAGS``)."""
    owners = site_owners(model.get_mapper_repr())
    carried: Dict[str, Dict[str, str]] = {}
    for kind, spec in DECLARED_SITES.items():
        read = SITE_TAGS[spec.tag]
        tags = ((key, read(wire, quantizer)) for key, (wire, quantizer) in owners[spec.owner].items())
        carried[kind] = {key: str(tag) for key, tag in tags if tag is not None}
    return carried


def verify_declared_state(model, declaration: Optional[Mapping[str, Any]]) -> None:
    """Refuse a program whose grids and ranges are not exactly the declared map's (a stale or unread map), at every
    site kind the map declares ([U-BRANGE2 stage 1e] read from the one inventory, ``DECLARED_SITES``): a site
    carrying another map's digest is stale, a declared site no quantizer carries is unread, and with no map every
    carried digest is stale."""
    carried = _carried(model)
    state = load_quantizer_state(declaration)
    if state is None:
        stale = sorted(f"{kind}:{name}" for kind, names in carried.items() for name in names)
        if stale:
            raise QuantizerStateError(f"sites {stale} carry a declared quantizer state but no quantizer_state "
                                      f"is declared: rerun Boundary Quantization and Readout Law")
        return
    for kind, names in declared_sites(state).items():
        field = DECLARED_SITES[kind].field
        stale = sorted(n for n, digest in carried[kind].items() if digest != state.sha256)
        if stale:
            raise QuantizerStateError(f"sites {stale} carry a {kind} ({field}) from a stale quantizer state, not "
                                      f"{state.sha256}: rerun Boundary Quantization and Readout Law")
        unread = sorted(names - set(carried[kind]))
        if unread:
            raise QuantizerStateError(f"quantizer_state declares a {kind} ({field}) for {unread}, which no quantizer "
                                      f"reads (Boundary Quantization needs activation_bits; Readout Law readout_law; "
                                      f"an offset or a lane needs onchip_arithmetic)")
