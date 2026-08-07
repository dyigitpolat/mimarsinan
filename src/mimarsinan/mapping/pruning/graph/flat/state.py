"""One-time lowering of the pruning context into flat arrays.

Everything here is STATIC for the whole analysis -- structure, exemptions,
protections, edges. The only mutable state the wave loop owns is the pair of
boolean vectors ``row_dead`` / ``col_dead`` (plus bank masks), which map back to
the context's per-node sets only at the boundary. No graph object is touched
inside a wave.

Ports are addressed flat: core k's neuron j is ``col_base[k] + j``; its axon i
is ``row_base[k] + i``. Axons come in three static families, mirroring the
reference kernel ``_cross_core_dead_axons`` case for case:

  always_dead   OFF-wired axons -- dead from wave 0, forever
  direct        axon reads ONE producer port -> dead iff that port is dead
  grouped       axon reads through a ComputeOp relay -> dead iff EVERY
                transfer-mapped producer port is dead (the ALL-reduce)

Orphaning mirrors ``_orphans_from_plan``: a port with no consumer at all is
orphaned outright (``orphan_now``); otherwise it dies when every consuming axon
row is dead. Output and transfer-protected ports never orphan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from mimarsinan.mapping.ir import IRSource

__all__ = ["FlatState", "build_flat_state"]


@dataclass
class FlatState:
    """Flat addressing plus the static edge families of one analysis."""

    node_ids: List[int]                 # ctx.neural_cores order
    index_of: Dict[int, int]            # node id -> dense index
    row_base: np.ndarray                # (n_cores+1,) axon offsets
    col_base: np.ndarray                # (n_cores+1,) neuron offsets

    always_dead_rows: np.ndarray        # flat axon indices, OFF-wired
    direct_axon: np.ndarray             # flat axon indices with ONE producer port
    direct_producer: np.ndarray         # matching flat producer-port indices
    group_axon: np.ndarray              # flat axon indices read through a relay
    group_offsets: np.ndarray           # CSR offsets into group_producers
    group_producers: np.ndarray         # flat producer-port indices, concatenated

    port_readers: Dict[tuple, List[int]]   # raw (node_id, index) -> reader positions
    core_readers: Dict[int, List[int]]     # producer node id -> reader positions

    orphan_now: np.ndarray              # flat ports with zero consumers
    consumer_port: np.ndarray           # CSR: port whose consumers follow
    consumer_offsets: np.ndarray
    consumer_rows: np.ndarray           # flat consuming axon-row indices
    never_orphan: np.ndarray            # bool over ports: output or protected

    n_rows: int
    n_cols: int

    def rows_of(self, k: int) -> slice:
        return slice(int(self.row_base[k]), int(self.row_base[k + 1]))

    def cols_of(self, k: int) -> slice:
        return slice(int(self.col_base[k]), int(self.col_base[k + 1]))


def build_flat_state(ctx) -> FlatState:
    """Lower ``ctx``'s static structure; mirrors the reference kernels exactly."""
    cores = list(ctx.neural_cores)
    node_ids = [n.id for n in cores]
    index_of = {nid: k for k, nid in enumerate(node_ids)}

    n_ax = [int(len(n.input_sources.flatten())) for n in cores]
    n_ne = []
    for n in cores:
        mat = ctx.base_node_matrix(n)
        n_ne.append(int(mat.shape[1]) if mat is not None else 0)
    row_base = np.zeros(len(cores) + 1, dtype=np.int64)
    col_base = np.zeros(len(cores) + 1, dtype=np.int64)
    np.cumsum(n_ax, out=row_base[1:])
    np.cumsum(n_ne, out=col_base[1:])

    fwd = ctx.computeop_transfers.forward_producers

    port_readers: Dict[tuple, List[int]] = {}
    core_readers: Dict[int, List[int]] = {}

    always_dead: List[int] = []
    d_axon: List[int] = []
    d_prod: List[int] = []
    g_axon: List[int] = []
    g_offsets: List[int] = [0]
    g_prods: List[int] = []

    def _flat_port(nid: int, col: int) -> int | None:
        k = index_of.get(nid)
        if k is None:
            return None
        return int(col_base[k]) + int(col)

    for k, node in enumerate(cores):
        base = int(row_base[k])
        for i, src in enumerate(node.input_sources.flatten()):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                always_dead.append(base + i)
                continue
            port_readers.setdefault((src.node_id, src.index), []).append(k)
            core_readers.setdefault(src.node_id, []).append(k)
            producers = fwd.get((src.node_id, src.index))
            if producers is not None:
                flat = [p for p in (_flat_port(nid, col) for nid, col in producers)
                        if p is not None]
                # the reference treats a relay with any unmapped producer as
                # unresolvable through this rule; only fully-mapped groups relay
                if flat and len(flat) == len(producers):
                    g_axon.append(base + i)
                    g_prods.extend(flat)
                    g_offsets.append(len(g_prods))
            elif src.node_id >= 0:
                p = _flat_port(src.node_id, src.index)
                if p is not None:
                    d_axon.append(base + i)
                    d_prod.append(p)

    protected = ctx.computeop_transfers.protected_ports
    effective = ctx.computeop_transfers.effective_consumers
    outputs = ctx.model_output_neurons

    orphan_now: List[int] = []
    c_port: List[int] = []
    c_offsets: List[int] = [0]
    c_rows: List[int] = []
    never = np.zeros(int(col_base[-1]), dtype=bool)

    for k, node in enumerate(cores):
        nid = node.id
        cbase = int(col_base[k])
        for j in range(n_ne[k]):
            key = (nid, j)
            if key in outputs or key in protected:
                never[cbase + j] = True
                continue
            consumers = list(ctx.consumer_axons.get(key, ()))
            consumers.extend(effective.get(key, ()))
            if not consumers:
                orphan_now.append(cbase + j)
                continue
            flat_rows = [
                int(row_base[index_of[cid]]) + int(ax)
                for cid, ax in consumers if cid in index_of
            ]
            if len(flat_rows) != len(consumers):
                # a consumer outside the lowered core set cannot be proven dead,
                # so the reference would keep the port alive forever
                never[cbase + j] = True
                continue
            c_port.append(cbase + j)
            c_rows.extend(flat_rows)
            c_offsets.append(len(c_rows))

    return FlatState(
        node_ids=node_ids, index_of=index_of,
        port_readers=port_readers, core_readers=core_readers,
        row_base=row_base, col_base=col_base,
        always_dead_rows=np.asarray(always_dead, dtype=np.int64),
        direct_axon=np.asarray(d_axon, dtype=np.int64),
        direct_producer=np.asarray(d_prod, dtype=np.int64),
        group_axon=np.asarray(g_axon, dtype=np.int64),
        group_offsets=np.asarray(g_offsets, dtype=np.int64),
        group_producers=np.asarray(g_prods, dtype=np.int64),
        orphan_now=np.asarray(orphan_now, dtype=np.int64),
        consumer_port=np.asarray(c_port, dtype=np.int64),
        consumer_offsets=np.asarray(c_offsets, dtype=np.int64),
        consumer_rows=np.asarray(c_rows, dtype=np.int64),
        never_orphan=never,
        n_rows=int(row_base[-1]), n_cols=int(col_base[-1]),
    )
