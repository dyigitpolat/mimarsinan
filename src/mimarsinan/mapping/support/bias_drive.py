"""The always-on bias row's drive on the integer value path: the largest code value of the grid its reader reads.

A parameter-encoded bias (cores without a bias register) rides an always-on axon row of the shared weight grid. On the
integer value path (a declared ``readout_law`` on value-domain cores) the network keeps its own units, so the row's
input value was the raw 1.0, a code of the reader's input grid only by accident; where a folded bias is large in those
units it set the shared grid and rounded most weights to 0 (W38: ResNet-50 at 8 bits, max|b| up to 21.9 max|w|). The
row now carries the largest code value of the grid its reader's input lies on (``reader_input_full_scales``):

* a reader of host-sourced axons reads its armed entry grid, whose top value is ``code_max step - offset``;
* a reader of core outputs reads the grid the law solver assigns it: the producers read together by any one core share
  the widest range of their union (``law_grids.assign_core_grids``, ``lane_grids.assign_lane_grids``), so the reader's
  grid is that union's grid and its top value ``levels (range / levels) - offset``. The producers are traced through
  layout nodes and the declared lanes (an on-chip add, reduction or pool writes on its lane quantizer; an absorbed clamp
  passes its producer's lane grid), the same union rule over the same quantizers' ranges;
* a reader whose grid is not determinable here (a consumer-grouped producer, a reader mixing host and core inputs, an
  unarmed grid) keeps the row at 1.0, and the step prints it by name.

The value is stamped on the perceptron at Weight Quantization entry (``stamp_bias_row_drives``), before any projection,
so every projection of the run (rungs, probes, recovery) codes the row on one lattice; the IR's cores carry it from their
perceptron (``install_bias_row_drives``); the readout law's bias term multiplies the row codes by it
(``law_rules.core_bias_and_data_weights``), and each solver refuses by name a core whose drive is not the top code value
of the input grid it assigned that core (``refuse_drives_off_their_grid``). The deployed program then computes
``(Delta_in acc + c x_fs) / theta`` with the projected model's bias exactly; what moves is the rounding: the bias is coded
on the drive's lattice (step ``x_fs / s``), and the shared grid compares ``max|w|`` with ``max|b| / x_fs``, which is
finer than the raw gauge's where ``x_fs > 1`` and can be coarser where ``x_fs < 1``.

Where the bias is not on a row of the weight grid (an on-chip register, a law-resolution bias, the register route), no
integer law reads the row, or the cores are not value-domain cores (the spiking wires, whose normalized full scale is 1),
no drive is stamped and every program is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_grids import GridUnion
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.lane_quantization import (
    LaneAddModule,
    LanePoolModule,
    LaneReduceModule,
    unsigned_levels,
)
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of
from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive, set_bias_row_drive

__all__ = [
    "BiasRowDriveRefusal", "BiasRowDriveStamps", "bias_rows_take_a_drive", "clear_bias_row_drives",
    "entry_grid_top", "install_bias_row_drives", "model_bias_row_drives", "perceptron_bias_row_drives",
    "reader_input_full_scales", "refuse_bias_row_drives", "refuse_drives_off_their_grid", "stamp_bias_row_drives",
]


class BiasRowDriveRefusal(ToolchainRefusal, ValueError):
    """A pass that reads an always-on axon at 1.0 met a row driven at another value, or a core's row drive is not the
    top code value of the input grid its law solver assigned it."""


def bias_rows_take_a_drive(config, *, on_chip_bias: bool, law_bias: bool, register_bias: bool,
                           value_domain: bool) -> bool:
    """Whether the run's always-on bias rows carry a drive: a declared integer readout law on value-domain cores, and
    the bias on an always-on row of the weight grid (no on-chip register, no law-resolution bias, not the register
    route)."""
    return (config.get("readout_law") is not None and bool(value_domain)
            and not (on_chip_bias or law_bias or register_bias))


def entry_grid_top(grid) -> Optional[float]:
    """The largest code value of an armed entry grid (``code_max step - offset``), None without a positive one."""
    if grid is None or not grid.armed:
        return None
    top = float(grid.value_range[1])
    return top if top > 0.0 else None


def _perceptron_nodes(model_repr):
    order, _, _ = model_repr._ensure_exec_graph()
    return [n for n in order if getattr(n, "perceptron", None) is not None]


def _writer(node):
    """The quantizer whose grid a reader of ``node``'s output reads: a core-mapped perceptron's readout quantizer, or
    the quantizer a declared lane op (an on-chip add, reduction or pool) rounds its output on; None otherwise."""
    perceptron = getattr(node, "perceptron", None)
    if perceptron is not None:
        scaler = getattr(perceptron, "scaler", None)
        if getattr(perceptron, "is_encoding_layer", False) or not isinstance(scaler, ReadoutQuantizer):
            return None
        return scaler
    module = getattr(node, "module", None)
    if isinstance(module, (LaneAddModule, LaneReduceModule, LanePoolModule)):
        return module.quantizer
    return None


def _traced_writers(model_repr, node, absorbed: FrozenSet[int]) -> Tuple[List, bool]:
    """(the writers of ``node``'s data inputs, whether any input is host-side), traced back through layout nodes and
    the lane-absorbed nodes that pass their producer's lane grid (an absorbed clamp)."""
    deps = model_repr._deps
    stack, seen, writers, host = list(deps.get(node, [])), set(), [], False
    while stack:
        source = stack.pop()
        if id(source) in seen:
            continue
        seen.add(id(source))
        writer = _writer(source)
        if writer is not None:
            writers.append(writer)
        elif getattr(source, "moves_elements_only", False) or id(source) in absorbed:
            stack.extend(deps.get(source, []))
        else:
            host = True
    return writers, host


def _lane_top(quantizer, widest: float) -> Optional[float]:
    """The largest code value of ``quantizer``'s grid kind at range ``widest``: codes ``[0, 2^b - 1]`` when unsigned,
    else ``[-L, L]``, values ``k (range / k_max) - offset`` (``lane_grids.LaneGrid``; a plain readout grid is the
    symmetric kind without offset, ``law_grids.assign_core_grids``)."""
    bits = int(quantizer.activation_bits)
    high = unsigned_levels(bits) if bool(getattr(quantizer, "lane_unsigned", False)) else grid_levels(bits)
    top = high * (float(widest) / high) - float(getattr(quantizer, "lane_offset", 0.0))
    return top if high > 0 and top > 0.0 else None


def reader_input_full_scales(model_repr, *, absorbed: FrozenSet[int] = frozenset()) -> Dict[int, Optional[float]]:
    """``{id(perceptron node): the top code value of the grid its input lies on}`` for every core-mapped perceptron
    (encoding layers left out), None where it is not determinable. ``absorbed`` holds the ``id`` of every host-op
    mapper the declared lanes absorb (``mapping.lanes.model_lanes.lane_absorbed``). The union is the solvers' own
    (``law_grids.GridUnion``)."""
    nodes =[n for n in _perceptron_nodes(model_repr) if not getattr(n.perceptron, "is_encoding_layer", False)]
    union = GridUnion()
    quantizers: Dict[int, Any] = {}
    reads: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    tops: Dict[int, Optional[float]] = {}
    for node in nodes:
        writers, host = _traced_writers(model_repr, node, absorbed)
        if not writers:
            tops[id(node)] = entry_grid_top(boundary_grid_of(getattr(node.perceptron, "input_activation", None)))
            continue
        grouped = any(getattr(w, "channel_groups", None) is not None for w in writers)
        if host or grouped:
            tops[id(node)] = None
            continue
        keys = tuple((id(w), 0) for w in writers)
        for w in writers:
            quantizers[id(w)] = w
        union.join_keys(keys)
        reads[id(node)] = keys
    widest: Dict[Tuple[int, int], Tuple[float, Any]] = {}
    for key in list(union.keys()):
        root = union.find_key(key)
        quantizer = quantizers[key[0]]
        value = float(quantizer.range)
        if root not in widest or value > widest[root][0]:
            widest[root] = (value, quantizer)
    for node_id, keys in reads.items():
        value, quantizer = widest[union.find_key(keys[0])]
        tops[node_id] = _lane_top(quantizer, value) if value > 0.0 else None
    return tops


@dataclass
class BiasRowDriveStamps:
    """What :func:`stamp_bias_row_drives` stamped: ``{perceptron name: drive}`` of the drives other than 1.0, and the
    perceptrons whose reader grid is not determinable (their rows stay at 1.0)."""

    drives: Dict[str, float] = field(default_factory=dict)
    undetermined: Tuple[str, ...] = ()


def _name(perceptron) -> str:
    return str(getattr(perceptron, "name", id(perceptron)))


def stamp_bias_row_drives(model, *, absorbed: FrozenSet[int] = frozenset()) -> BiasRowDriveStamps:
    """Stamp every core-mapped perceptron's row drive (the top code value of its reader's input grid,
    :func:`reader_input_full_scales`); encoding layers (run on the host) and undeterminable readers keep 1.0."""
    model_repr = model.get_mapper_repr()
    tops = reader_input_full_scales(model_repr, absorbed=absorbed)
    stamps = BiasRowDriveStamps()
    undetermined = []
    for node in _perceptron_nodes(model_repr):
        perceptron = node.perceptron
        if getattr(perceptron, "is_encoding_layer", False):
            set_bias_row_drive(perceptron, 1.0)
            continue
        drive = tops.get(id(node))
        set_bias_row_drive(perceptron, 1.0 if drive is None else drive)
        if drive is None:
            undetermined.append(_name(perceptron))
        elif drive != 1.0:
            stamps.drives[_name(perceptron)] = drive
    stamps.undetermined = tuple(undetermined)
    return stamps


def clear_bias_row_drives(model) -> None:
    """Remove every perceptron's row drive (the route takes none)."""
    for perceptron in model.get_perceptrons():
        set_bias_row_drive(perceptron, 1.0)


def perceptron_bias_row_drives(model) -> Dict[str, float]:
    """``{perceptron name: drive}`` of the model's stamped perceptrons (drives other than 1.0)."""
    return {_name(p): bias_row_drive(p) for p in model.get_perceptrons() if bias_row_drive(p) != 1.0}


def model_bias_row_drives(model) -> Dict[int, float]:
    """``{perceptron index: drive}`` of the model's stamped perceptrons (drives other than 1.0)."""
    out: Dict[int, float] = {}
    for node in _perceptron_nodes(model.get_mapper_repr()):
        index = getattr(node, "perceptron_index", None)
        drive = bias_row_drive(node.perceptron)
        if index is not None and drive != 1.0:
            out[int(index)] = drive
    return out


def install_bias_row_drives(ir_graph, model) -> int:
    """Stamp each IR core with its perceptron's row drive (idempotent); return how many carry one other than 1.0."""
    drives = model_bias_row_drives(model)
    stamped = 0
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        drive = drives.get(node.perceptron_index, 1.0) if node.perceptron_index is not None else 1.0
        set_bias_row_drive(node, drive)
        stamped += int(drive != 1.0)
    return stamped


def _has_always_on_rows(core: NeuralCore) -> bool:
    return any(source.is_always_on() for source in core.input_sources.flatten())


def reads_a_core(sources: Iterable, nodes_by_id: Mapping[int, object]) -> bool:
    """Whether any of a core's data-axon ``sources`` is a core output (else its data axons are host-sourced)."""
    return any(source.node_id >= 0 and isinstance(nodes_by_id.get(int(source.node_id)), NeuralCore)
               for source in sources)


def refuse_drives_off_their_grid(ir_graph, core_fed_top: Callable[[NeuralCore], Optional[float]]) -> int:
    """Refuse by name a core whose always-on rows carry a drive other than the top code value of the input grid its
    law solver assigned it: ``core_fed_top(core)`` for a reader of core outputs (None for a reader of host-sourced axons,
    which reads its entry grid). Returns how many cores were checked."""
    checked = 0
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or bias_row_drive(node) == 1.0 or not _has_always_on_rows(node):
            continue
        top = core_fed_top(node)
        if top is None:
            top = entry_grid_top(getattr(node, "boundary_grid", None))
        drive = bias_row_drive(node)
        if top != drive:
            raise BiasRowDriveRefusal(
                f"core {node.name!r}: its always-on rows are driven at {drive!r}, but the input grid its readout law "
                f"reads tops out at {top!r}: the drive stamped at Weight Quantization is not a code of the grid the "
                "deployed core reads")
        checked += 1
    return checked


def refuse_bias_row_drives(ir_graph, what: str) -> None:
    """Refuse by name a pass that reads always-on axons at 1.0 on an IR whose cores carry another row drive
    (``install_bias_row_drives`` stamps them right after mapping)."""
    drives = [bias_row_drive(n) for n in ir_graph.nodes if isinstance(n, NeuralCore) and bias_row_drive(n) != 1.0]
    if drives:
        raise BiasRowDriveRefusal(
            f"{what} reads an always-on axon at 1.0, but {len(drives)} core(s) carry their bias rows at their "
            f"reader's input full scale (drives {min(drives):.6g} to {max(drives):.6g}): this combination is not "
            "built on this tree")
