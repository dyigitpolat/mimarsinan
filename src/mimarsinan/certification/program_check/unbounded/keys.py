"""[E1 F1] An IR's neurons named by their identity at unbounded geometry, so the IR a deployment maps at the declared
geometry compares neuron by neuron with the IR the same network maps at unbounded geometry.

A perceptron's neuron is ``(perceptron, output column, channel)``: its node's ``perceptron_index``, its
``perceptron_output_column`` (a per-column node's column, -1 for none) and its channel, the node's
``perceptron_output_slice`` start plus its column before IR pruning (``pre_pruning_col_mask``). Core geometry tiles
a perceptron's output into nodes and the IR's node ids are cost-only, so the identity never depends on either. A host
op's output is ``(name, index)``, the raw input's ``index``, the always-on axon one source of its own. Each identity
is one int64 code in a ``KeySpace`` both IRs share (owner index above bit 32, channel below), so a node's fan-in is its
rows summed per source code (``view.canonical``), exactly as the static check sums a placement's.

[E1 fix round] Every node is read through the compaction the identity builder applies to it (``_ir_keep_indices``,
``compact_soft_core_mapping``'s own rule): only its kept rows carry fan-in, only its kept columns are neurons, and a
row reading a column its producer drops reads an off axon, as the built program does."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.certification.program_check.view import OFF, Canonical, canonical
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore.compaction import _ir_keep_indices

INPUT_OWNER, ON_OWNER = ("input",), ("always-on",)
#: Coalescing and partial-sum roles the predicate reads: a node holding a neuron's whole fan-in (the IR's coalescing
#: master, or no group). A partial or accumulator node would hold a share of a neuron's fan-in, which no builder of this
#: tree emits, so the predicate refuses to reassemble one by name (``partial_sum_group``).
WHOLE_FAN_IN_ROLES = (None, "master")


class KeySpace:
    """One int64 code per neuron identity, shared by the two IRs compared: an owner (a perceptron's output column, a
    host op, the raw input, the always-on axon, or one IR's unkeyed node) gets an index on first sight."""

    def __init__(self) -> None:
        self._owners: Dict[Tuple[Any, ...], int] = {}
        self.names: List[Tuple[Any, ...]] = []

    def owner(self, owner: Tuple[Any, ...]) -> np.int64:
        if owner not in self._owners:
            self._owners[owner] = len(self.names)
            self.names.append(owner)
        return np.int64(self._owners[owner]) << np.int64(32)

    def codes(self, owner: Tuple[Any, ...], channels: np.ndarray) -> np.ndarray:
        return self.owner(owner) | np.asarray(channels, dtype=np.int64)

    def text(self, code: int) -> str:
        owner, channel = self.names[int(code) >> 32], int(code) & 0xFFFFFFFF
        if owner[0] == "perceptron":
            column = "" if owner[2] < 0 else f" column {owner[2]}"
            return f"perceptron {owner[1]}{column} channel {channel}"
        if owner[0] == "host":
            return f"host op {owner[1]}[{channel}]"
        if owner == ON_OWNER:
            return "always-on axon"
        if owner == INPUT_OWNER:
            return f"input[{channel}]"
        return f"unkeyed node {owner[2]}[{channel}] of the {owner[1]} IR"


def node_width(node: Any, graph: IRGraph) -> int:
    """The node's output count before soft-core compaction (a bank-backed node's without materializing its matrix)."""
    if isinstance(node, NeuralCore):
        return int(node.get_output_count())
    shape = getattr(node, "output_shape", None)
    return int(np.prod(shape)) if shape is not None else 0


def perceptron_channels(node: NeuralCore, width: int) -> Optional[np.ndarray]:
    """The perceptron output channel of each of ``node``'s columns, None when the node has no perceptron identity or
    its pre-pruning mask's survivors are not its columns."""
    if node.perceptron_index is None or node.perceptron_output_slice is None:
        return None
    columns = np.arange(width, dtype=np.int64)
    mask = getattr(node, "pre_pruning_col_mask", None)
    if mask is not None:
        columns = np.flatnonzero(~np.asarray(mask, dtype=bool)).astype(np.int64)
        if len(columns) != width:
            return None
    return int(node.perceptron_output_slice[0]) + columns


def kept(node: NeuralCore) -> Tuple[np.ndarray, np.ndarray]:
    """The rows and columns of ``node`` the identity builder's soft-core compaction keeps."""
    rows, cols = _ir_keep_indices(node)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def output_channels(node: Any, graph: IRGraph) -> Optional[np.ndarray]:
    """The channel at unbounded geometry of each program output index of ``node`` (its kept columns in order; a host
    op's and the raw input's output index), None for a neural node with no perceptron identity: the channel identity
    a dithered encode must rotate each encoded neuron by, derived independently of the builders' channel tables."""
    if not isinstance(node, NeuralCore):
        return np.arange(node_width(node, graph), dtype=np.int64)
    channels = perceptron_channels(node, node_width(node, graph))
    return None if channels is None else channels[kept(node)[1]]


def source_pairs(sources) -> Tuple[np.ndarray, np.ndarray]:
    """Per source: the producer node id (-1 off or no ``IRSource``, -2 raw input, -3 always-on) and its index."""
    flat = list(sources)
    nodes = np.fromiter((int(s.node_id) if isinstance(s, IRSource) else -1 for s in flat), np.int64, len(flat))
    index = np.fromiter((int(s.index) if isinstance(s, IRSource) else 0 for s in flat), np.int64, len(flat))
    return nodes, index


@dataclass
class KeyedIR:
    """An IR's neurons in a shared key space: per node its codes and channels per column before compaction (a column
    compaction drops codes ``OFF``), its kept rows and columns, the neuron of every code, the nodes it cannot key, the
    duplicated codes, and per host-op name its node."""

    label: str
    graph: IRGraph
    space: KeySpace
    codes: Dict[int, np.ndarray] = field(default_factory=dict)
    channels: Dict[int, np.ndarray] = field(default_factory=dict)
    rows: Dict[int, np.ndarray] = field(default_factory=dict)
    cols: Dict[int, np.ndarray] = field(default_factory=dict)
    neurons: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    unkeyed: List[int] = field(default_factory=list)
    duplicated: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    hosts: Dict[str, Any] = field(default_factory=dict)
    by_id: Dict[int, Any] = field(default_factory=dict)

    def source_codes(self, sources) -> np.ndarray:
        """One code per IR source: the producer neuron's identity, the raw input's, the always-on axon's, ``OFF`` for
        an off axon or a column its producer's compaction drops, and a code only this IR produces for an unkeyed
        producer."""
        nodes, index = source_pairs(sources)
        out = np.full(len(nodes), OFF, dtype=np.int64)
        out[nodes == -3] = self.space.owner(ON_OWNER)
        raw = nodes == -2
        out[raw] = self.space.owner(INPUT_OWNER) | index[raw]
        live = np.flatnonzero(nodes >= 0)
        order = live[np.argsort(nodes[live], kind="stable")]
        starts = np.flatnonzero(np.r_[True, nodes[order][1:] != nodes[order][:-1]]) if len(order) else order
        for start, end in zip(starts.tolist(), np.r_[starts[1:], len(order)].tolist()):
            sel = order[start:end]
            producer, idx = int(nodes[sel[0]]), index[sel]
            table = self.codes.get(producer)
            unkeyed = self.space.owner(("unkeyed", self.label, producer)) | idx
            if table is None:
                out[sel] = unkeyed
                continue
            inside = idx < len(table)
            out[sel] = np.where(inside, table[np.where(inside, idx, 0)], unkeyed)
        return out

    def node_sources(self, node: Any) -> list:
        return np.asarray(node.input_sources, dtype=object).flatten().tolist()

    def block(self, node: NeuralCore) -> Canonical:
        """The node's fan-in per source code over its kept rows (rows summed per code, off rows dropped), every column
        before compaction (a dropped column's weights are never read)."""
        rows = self.rows[int(node.id)]
        sources = self.node_sources(node)
        matrix = np.asarray(node.get_core_matrix(self.graph))
        return canonical(self.source_codes([sources[r] for r in rows.tolist()]), matrix[rows], None)


def keyed(graph: IRGraph, space: KeySpace, label: str) -> KeyedIR:
    out = KeyedIR(label, graph, space)
    for node in graph.nodes:
        nid = int(node.id)
        out.by_id[nid] = node
        if isinstance(node, ComputeOp):
            name = str(node.name)
            if name in out.hosts:
                out.duplicated.append((-1, (int(out.hosts[name].id), 0), (nid, 0)))
            out.hosts[name] = node
            channels = np.arange(node_width(node, graph), dtype=np.int64)
            owner: Tuple[Any, ...] = ("host", name)
        elif isinstance(node, NeuralCore):
            found, perceptron = perceptron_channels(node, node_width(node, graph)), node.perceptron_index
            if found is None or perceptron is None:
                out.unkeyed.append(nid)
                continue
            channels = found
            column = -1 if node.perceptron_output_column is None else int(node.perceptron_output_column)
            owner = ("perceptron", int(perceptron), column)
            out.rows[nid], out.cols[nid] = kept(node)
        else:
            continue
        codes = space.codes(owner, channels)
        if isinstance(node, NeuralCore):
            dropped = np.ones(len(codes), dtype=bool)
            dropped[out.cols[nid]] = False
            codes[dropped] = OFF
            for column in out.cols[nid].tolist():
                code = int(codes[column])
                held = out.neurons.setdefault(code, (nid, column))
                if held != (nid, column):
                    out.duplicated.append((code, held, (nid, column)))
        out.codes[nid], out.channels[nid] = codes, channels
    return out
