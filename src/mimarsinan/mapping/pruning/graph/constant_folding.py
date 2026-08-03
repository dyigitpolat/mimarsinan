"""Graph-level constant propagation: the lattice sweep and the physical fold plan.

``ConstantFoldState`` is the graph-level half of the constant lattice. It owns
(a) the lattice itself, (b) each core's constant CARRIER, and (c) the carrier
DELTAS that the folds accumulate, and it serves the EFFECTIVE (post-fold)
matrix / bias to every propagation kernel so that starvation, orphaning and
the column-value rule all see the same structure the deployed program will
see. Nothing is mutated on the IR until :func:`apply_constant_folds` runs, so
the analysis stays replayable and the default (nothing folded) path hands back
the original arrays unchanged — byte-identity by construction.

``refresh_constant_folds`` is ONE monotone sweep, and it obeys the same HOP
discipline as the rest of the framework:

- ComputeOp descents are applied inside the sweep, because a transferable op
  is transparent WIRING, not a hop (the W4b-1 ``forward_producers`` map
  composes op chains the same way);
- NeuralCore row folds, row kills and column descents are GATHERED from the
  pre-sweep state and committed together, so a core-to-core constant costs
  exactly one hop — which is what keeps the depth replay's wave semantics and
  the ``masked <= closure <= cascade`` ordering honest.

The masked arm never sweeps. Closure sweeps ONCE with ``cross_core=False``:
a CONST line still kills the rows that read it (one-hop coupling), but a
column is never *discovered* to be constant, because that is emergent
deadness and closure must not find it. The cascade fixpoint and the ledger's
depth replay sweep to quiescence with ``cross_core=True``.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.pruning.liveness_transfer.constant_core import (
    CARRIER_BIAS,
    CARRIER_ROW,
    ConstantCarrier,
    derive_core_constants,
    resolve_constant_carrier,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_lattice import (
    build_source_plan,
    ConstantLattice,
    source_constant,
)
from mimarsinan.mapping.pruning.liveness_transfer.probe_memo import (
    derive_constant_outputs_memoized,
)

__all__ = [
    "ConstantFoldApplicationError",
    "ConstantFoldState",
    "ConstantSweep",
    "apply_constant_folds",
    "refresh_constant_folds",
]

class ConstantFoldApplicationError(RuntimeError):
    """A recorded fold has nowhere to land — the analysis and the IR disagree."""

@dataclass
class ConstantSweep:
    """The deferred half of one sweep: core kills, folds, column descents."""

    dead_rows: Dict[int, Set[int]] = field(default_factory=dict)
    folds: List[Tuple[int, int, float, np.ndarray]] = field(default_factory=list)
    descents: List[Tuple[Tuple[int, int], float]] = field(default_factory=list)
    op_changed: bool = False

    def commit(self, ctx) -> bool:
        """Apply the gathered facts; True iff anything changed."""
        state: ConstantFoldState = ctx.constants
        changed = self.op_changed
        for nid, rows in self.dead_rows.items():
            new = rows - ctx.pruned_rows[nid]
            if new:
                ctx.pruned_rows[nid] |= new
                changed = True
        for nid, row, constant, weights in self.folds:
            if row in ctx.pruned_rows[nid]:
                continue
            state.record_fold(nid, row, constant, weights)
            ctx.pruned_rows[nid].add(row)
            changed = True
        for port, value in self.descents:
            if state.lattice.descend(port, value):
                changed = True
        return changed

@dataclass
class ConstantFoldState:
    """Lattice + carrier deltas; the SSOT for effective (post-fold) structure."""

    enabled: bool = False
    lattice: ConstantLattice = field(default_factory=ConstantLattice)
    carriers: Dict[int, ConstantCarrier | None] = field(default_factory=dict)
    deltas: Dict[int, np.ndarray] = field(default_factory=dict)
    folded_rows: Dict[int, Dict[int, float]] = field(default_factory=dict)
    _matrix_cache: Dict[int, tuple] = field(default_factory=dict, repr=False)
    # (op_id, bitwise key) -> resolved outputs; injectable so one run's arms
    # and replay share every battery (see liveness_transfer.probe_memo).
    probe_memo: Dict[tuple, Dict[int, float]] = field(default_factory=dict, repr=False)

    def carrier(
        self, node: NeuralCore, *, pruned_rows, exempt_rows
    ) -> ConstantCarrier | None:
        """The core's carrier, pinned on first use so folds stay consistent."""
        nid = node.id
        if nid in self.carriers:
            return self.carriers[nid]
        resolved = resolve_constant_carrier(
            node, pruned_rows=pruned_rows, exempt_rows=exempt_rows
        )
        self.carriers[nid] = resolved
        return resolved

    def record_fold(self, node_id: int, row: int, constant: float, row_weights) -> None:
        """Move ``constant * row_weights`` onto the carrier delta of one core."""
        contribution = float(constant) * np.asarray(row_weights, dtype=np.float64)
        delta = self.deltas.get(node_id)
        self.deltas[node_id] = (
            contribution if delta is None else delta + contribution
        )
        self.folded_rows.setdefault(node_id, {})[int(row)] = float(constant)
        self._matrix_cache.pop(node_id, None)

    def effective_matrix(self, node: NeuralCore, base: np.ndarray) -> np.ndarray:
        """``base`` with any row-carrier delta applied (identity when nothing folded)."""
        delta = self.deltas.get(node.id)
        carrier = self.carriers.get(node.id)
        if delta is None or carrier is None or carrier.kind != CARRIER_ROW:
            return base
        cached = self._matrix_cache.get(node.id)
        if cached is not None and cached[0] is base:
            return cached[1]
        folded = np.array(base, dtype=np.float64, copy=True)
        folded[carrier.row, :] += delta
        self._matrix_cache[node.id] = (base, folded)
        return folded

    def effective_bias(self, node: NeuralCore) -> "np.ndarray | None":
        """``hardware_bias`` with any bias-carrier delta applied."""
        base = getattr(node, "hardware_bias", None)
        delta = self.deltas.get(node.id)
        carrier = self.carriers.get(node.id)
        if delta is None or carrier is None or carrier.kind != CARRIER_BIAS:
            return base
        if base is None:
            return delta
        return np.asarray(base, dtype=np.float64) + delta

    def total_folded_rows(self) -> int:
        return sum(len(rows) for rows in self.folded_rows.values())

def refresh_constant_folds(ctx, *, cross_core: bool = True, only_ids=None) -> ConstantSweep:
    """One monotone sweep of the constant lattice; the caller commits it.

    Deferred commit lets the replay run a whole wave against pre-wave state.
    """
    state: ConstantFoldState = ctx.constants
    sweep = ConstantSweep()
    if not state.enabled:
        return sweep
    _t0 = time.perf_counter()
    sweep.op_changed = _sweep_compute_ops(ctx, state)
    _t1 = time.perf_counter()
    _gather_neural_cores(ctx, state, sweep, cross_core=cross_core, only_ids=only_ids)
    if time.perf_counter() - _t0 > 5.0:   # slow sweeps only; tests stay silent
        print(f"[ConstantSweep] op_probe={_t1 - _t0:.1f}s core_gather="
              f"{time.perf_counter() - _t1:.1f}s only={only_ids is not None}", flush=True)
    return sweep

def _sweep_compute_ops(ctx, state: ConstantFoldState) -> bool:
    """Op descents apply immediately: a host op is wiring, not a hop."""
    changed = False
    transfers = ctx.computeop_transfers.per_op
    for node in ctx.graph.nodes:
        if not isinstance(node, ComputeOp):
            continue
        transfer = transfers.get(node.id)
        if transfer is None:
            continue
        in_values = [
            source_constant(
                src, lattice=state.lattice, pruned_cols=ctx.pruned_cols
            )
            for src in node.input_sources.flatten()
        ]
        if all(v is None for v in in_values):
            continue
        resolved = derive_constant_outputs_memoized(
            node, transfer, in_values, state.probe_memo
        )
        for out_idx, value in resolved.items():
            if state.lattice.descend((node.id, out_idx), value):
                changed = True
    return changed

def _gather_neural_cores(
    ctx, state: ConstantFoldState, sweep: ConstantSweep, *, cross_core: bool,
    only_ids=None,
) -> None:
    # only_ids: pure restriction (change tracking); the engine's final full
    # gather asserts a skipped core had nothing new (FlatEngineQuiescenceError).
    plans = getattr(ctx, "_source_plans", None)
    if plans is None:
        plans = {}
        setattr(ctx, "_source_plans", plans)

    for node in ctx.neural_cores:
        if only_ids is not None and node.id not in only_ids:
            continue
        base = ctx.base_node_matrix(node)
        if base is None:
            continue
        nid = node.id
        exempt = ctx.exempt_rows.get(nid, frozenset())
        carrier = state.carrier(
            node, pruned_rows=ctx.pruned_rows[nid], exempt_rows=exempt
        )
        matrix = state.effective_matrix(node, base)
        facts = derive_core_constants(
            matrix=matrix,
            bias=state.effective_bias(node),
            threshold=float(getattr(node, "threshold", 1.0)),
            row_values=(
                plans.get(nid)
                or plans.setdefault(nid, build_source_plan(node.input_sources))
            ).row_values(lattice=state.lattice, pruned_cols=ctx.pruned_cols),
            pruned_rows=ctx.pruned_rows[nid],
            pruned_cols=ctx.pruned_cols[nid],
            exempt_rows=exempt,
            carrier=carrier,
            admits_nonzero=state.lattice.admits_nonzero,
        )
        new_dead = facts.dead_rows - ctx.pruned_rows[nid]
        if new_dead:
            sweep.dead_rows.setdefault(nid, set()).update(new_dead)
        for row, constant in sorted(facts.fold_rows.items()):
            sweep.folds.append((nid, row, constant, np.array(matrix[row, :])))
        if not cross_core:
            continue
        for col, value in facts.column_values.items():
            sweep.descents.append(((nid, col), value))

def apply_constant_folds(graph, state: ConstantFoldState) -> int:
    """Materialize the carrier deltas on the IR; returns the cores touched.

    Arrays are COPIED before mutation: cores and bank slices share deduped
    ndarrays ([F2] upload memo, bank bias views), so an in-place ``+=`` would
    leak one instance's fold into another's weights.
    """
    if not state.deltas:
        return 0
    by_id = {n.id: n for n in graph.nodes if isinstance(n, NeuralCore)}
    touched = 0
    for nid, delta in state.deltas.items():
        node = by_id.get(nid)
        carrier = state.carriers.get(nid)
        if node is None or carrier is None or not carrier.writable:
            raise ConstantFoldApplicationError(
                f"constant fold recorded for NeuralCore id={nid} has no "
                f"writable carrier to land on (node={node is not None}, "
                f"carrier={carrier!r}); the rows were already counted as "
                "eliminated, so silently dropping the delta would change the "
                "program."
            )
        if carrier.kind == CARRIER_ROW and node.core_matrix is not None:
            matrix = np.array(node.core_matrix, copy=True)
            matrix[carrier.row, :] += delta.astype(matrix.dtype, copy=False)
            node.core_matrix = matrix
            touched += 1
        elif carrier.kind == CARRIER_BIAS:
            base = getattr(node, "hardware_bias", None)
            bias = (
                np.array(delta, copy=True) if base is None
                else np.array(base, copy=True)
                + delta.astype(np.asarray(base).dtype, copy=False)
            )
            node.hardware_bias = bias
            touched += 1
        else:
            raise ConstantFoldApplicationError(
                f"constant fold for NeuralCore id={nid} names carrier "
                f"{carrier!r} that the IR no longer provides."
            )
    return touched

