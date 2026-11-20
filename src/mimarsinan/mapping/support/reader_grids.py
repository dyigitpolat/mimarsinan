"""The grid each core-mapped perceptron's input lies on, traced through the model's mapper graph (the readers' side of
the integer value path): an armed entry grid for a reader of host values, else the union of the producer grids it reads
together (the law solvers' own union, ``law_grids.GridUnion``, over their keys). The always-on row's drive is a code
of that grid (``bias_drive``), and a union of producer grids read together is one range site of the range rule
(``read_together``).
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from mimarsinan.mapping.export.law_grids import GridUnion, readout_grid_keys, readout_key_grid
from mimarsinan.mapping.support.drive_rule import DriveGrid
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.lane_quantization import (
    LaneAddModule,
    LanePoolModule,
    LaneReduceModule,
    unsigned_levels,
)
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of

__all__ = ["entry_drive_grid", "entry_grid_top", "read_together", "reader_input_full_scales", "reader_input_grids"]


def entry_grid_top(grid) -> Optional[float]:
    """The largest code value of an armed entry grid (``code_max step - offset``), None without a positive one."""
    if grid is None or not grid.armed:
        return None
    top = float(grid.value_range[1])
    return top if top > 0.0 else None


def entry_drive_grid(grid) -> Optional[DriveGrid]:
    """[U-BRANGE2] An armed entry grid's code values (``k step - offset``), None without a positive top."""
    if entry_grid_top(grid) is None:
        return None
    return DriveGrid(step=float(grid.step), offset=float(grid.entry_offset()), code_max=int(grid.code_range[1]))


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


def _bank_writers(node) -> List:
    """[U-BRANGE2] The readout quantizers of an operand product's banks (N2: each bank a core whose output grid its
    reader reads, joined by the law solvers as any producer), empty for any other node or a bank without one."""
    banks = getattr(node, "banks", None) or ()
    scalers = [getattr(op, "scaler", None) for op in banks]
    return scalers if scalers and all(isinstance(s, ReadoutQuantizer) for s in scalers) else []


def _traced_writers(model_repr, node, absorbed: FrozenSet[int]) -> Tuple[List, bool]:
    """(the writers of ``node``'s data inputs, whether any input is host-side), traced back through layout nodes and
    the lane-absorbed nodes that pass their producer's lane grid (an absorbed clamp); an operand product's writers are
    its banks' readout quantizers."""
    deps = model_repr._deps
    stack, seen, writers, host = list(deps.get(node, [])), set(), [], False
    while stack:
        source = stack.pop()
        if id(source) in seen:
            continue
        seen.add(id(source))
        writer = _writer(source)
        banks = [] if writer is not None else _bank_writers(source)
        if writer is not None:
            writers.append(writer)
        elif banks:
            writers.extend(banks)
        elif getattr(source, "moves_elements_only", False) or id(source) in absorbed:
            stack.extend(deps.get(source, []))
        else:
            host = True
    return writers, host


def _lane_grid(quantizer, widest: float) -> Optional[DriveGrid]:
    """The code values of ``quantizer``'s grid kind at range ``widest``: codes ``[0, 2^b - 1]`` when unsigned,
    else ``[-L, L]``, values ``k (range / k_max) - offset`` (``lane_grids.LaneGrid``; a plain readout grid is the
    symmetric kind without offset, ``law_grids.assign_core_grids``); None without a positive top."""
    bits = int(quantizer.activation_bits)
    high = unsigned_levels(bits) if bool(getattr(quantizer, "lane_unsigned", False)) else grid_levels(bits)
    if high <= 0:
        return None
    grid = DriveGrid(step=float(widest) / high, offset=float(getattr(quantizer, "lane_offset", 0.0)), code_max=high)
    return grid if grid.top > 0.0 else None


def _lane_top(quantizer, widest: float) -> Optional[float]:
    """The largest code value of ``quantizer``'s grid kind at range ``widest`` (:func:`_lane_grid`'s top)."""
    grid = _lane_grid(quantizer, widest)
    return None if grid is None else grid.top


def reader_input_full_scales(model_repr, *, absorbed: FrozenSet[int] = frozenset(),
                             why: Optional[Dict[int, str]] = None) -> Dict[int, Optional[float]]:
    """``{id(perceptron node): the top code value of the grid its input lies on}`` (:func:`reader_input_grids`)."""
    return {k: (None if g is None else g.top) for k, g in reader_input_grids(model_repr, absorbed=absorbed,
                                                                             why=why).items()}


def reader_input_grids(model_repr, *, absorbed: FrozenSet[int] = frozenset(),
                       why: Optional[Dict[int, str]] = None) -> Dict[int, Optional[DriveGrid]]:
    """``{id(perceptron node): the grid its input lies on}`` for every core-mapped perceptron (encoding layers left
    out), None where it is not determinable ([U-LMP] its reason in ``why``). ``absorbed`` holds the ``id`` of every
    host-op mapper the declared lanes absorb (``mapping.lanes.model_lanes.lane_absorbed``). The union is the solvers'
    own (``law_grids.GridUnion``) over their keys (``readout_grid_keys``: one per consumer group)."""
    why = {} if why is None else why
    nodes = [n for n in _perceptron_nodes(model_repr) if not getattr(n.perceptron, "is_encoding_layer", False)]
    union = GridUnion()
    quantizers: Dict[int, Any] = {}
    reads: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    tops: Dict[int, Optional[DriveGrid]] = {}
    for node in nodes:
        writers, host = _traced_writers(model_repr, node, absorbed)
        if not writers:
            tops[id(node)] = entry_drive_grid(boundary_grid_of(getattr(node.perceptron, "input_activation", None)))
            if tops[id(node)] is None:
                why[id(node)] = "it reads host values without an armed entry grid (the law solver refuses it)"
        elif host:
            tops[id(node)], why[id(node)] = None, "it reads host values and core outputs (no one DAC grid)"
        else:
            quantizers.update((id(w), w) for w in writers)
            reads[id(node)] = tuple(key for w in writers for key in readout_grid_keys(w))
            union.join_keys(reads[id(node)])
    widest: Dict[Tuple[int, int], Tuple[float, Any, int, bool]] = {}
    kinds: Dict[Tuple[int, int], set] = {}
    for key in list(union.keys()):
        root, quantizer = union.find_key(key), quantizers[key[0]]
        value, count, grouped = readout_key_grid(quantizer, key[1])
        kinds.setdefault(root, set()).add((count, grouped))
        if root not in widest or value > widest[root][0]:
            widest[root] = (value, quantizer, count, grouped)
    for node_id, keys in reads.items():
        value, quantizer, count, grouped = widest[union.find_key(keys[0])]
        joined = kinds[union.find_key(keys[0])]
        if any(g for _c, g in joined) and len({c for c, _g in joined}) > 1:
            tops[node_id] = None
            why[node_id] = f"it joins producer grids of {sorted({c for c, _g in joined})} levels (no one DAC grid)"
        elif value <= 0.0:
            tops[node_id] = None
        else:
            tops[node_id] = (DriveGrid(step=value / count, offset=0.0, code_max=int(count)) if grouped
                             else _lane_grid(quantizer, value))
    return tops


def read_together(model_repr, *, absorbed: FrozenSet[int] = frozenset()) -> List[Tuple[Tuple[Any, int], ...]]:
    """[U-BRANGE2, the review's m-3] The producer grids one reader reads together: every union of two or more keys of
    the law solvers' own union (``readout_grid_keys``: a plain quantizer's one key, a grouped quantizer's key per
    group, a lane's), as ``(quantizer, group)`` pairs. One core reads such a union on one DAC grid (the solvers give it
    the widest of its members' ranges), so the union is one range site of the rule: its values pooled over every
    member (``range_rule.least_squares_pooled``)."""
    nodes = [n for n in _perceptron_nodes(model_repr) if not getattr(n.perceptron, "is_encoding_layer", False)]
    union = GridUnion()
    quantizers: Dict[int, Any] = {}
    for node in nodes:
        writers, host = _traced_writers(model_repr, node, absorbed)
        if writers and not host:
            quantizers.update((id(w), w) for w in writers)
            union.join_keys(tuple(key for w in writers for key in readout_grid_keys(w)))
    members: Dict[Any, List[Tuple[Any, int]]] = {}
    for key in sorted(union.keys(), key=lambda k: (k[1], list(quantizers).index(k[0]))):
        members.setdefault(union.find_key(key), []).append((quantizers[key[0]], int(key[1])))
    return [tuple(group) for group in members.values() if len(group) > 1]
