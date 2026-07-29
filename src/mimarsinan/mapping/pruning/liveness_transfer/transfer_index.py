"""Graph-level composition of per-op transfers: the maps the kernels consume.

``build_computeop_transfer_index`` folds the per-op ``LivenessTransfer``
relations through arbitrary op CHAINS (e.g. NC -> ReLU -> AvgPool -> NC)
into three graph-level structures; closure, the cascade fixpoint, and the
elimination ledger's depth replay all consume the SAME index, so the
replay-reconciliation guard holds by construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Set, Tuple

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import (
    build_computeop_producer_map,
    build_computeop_referenced_neurons,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_policy import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    require_computeop_liveness_transfers,
)
from mimarsinan.mapping.pruning.liveness_transfer.transfer_registry import (
    LivenessTransfer,
    derive_liveness_transfer,
)

__all__ = ["ComputeOpTransferIndex", "build_computeop_transfer_index"]

Port = Tuple[int, int]


@dataclass(frozen=True)
class ComputeOpTransferIndex:
    """Transfers composed through op chains into graph-level liveness maps.

    - ``forward_producers[(op_id, out)]`` — the NeuralCore ports that must ALL
      be dead for that op output to be constant zero (an empty set means the
      whole region is off-wired: constant zero already). Ports without an
      entry never relay forward.
    - ``effective_consumers[(nc_id, col)]`` — NeuralCore ``(consumer, axon)``
      readers reached THROUGH transferable ops; dead only when all are dead.
    - ``protected_ports`` — producer ports that feed an opaque op, reach a
      model output through ops, or sit on an underivable path: the starvation
      guard, never orphan-killed.
    """

    forward_producers: Mapping[Port, FrozenSet[Port]]
    effective_consumers: Mapping[Port, FrozenSet[Port]]
    protected_ports: FrozenSet[Port]


def _identity_only_index(ir_graph: IRGraph) -> ComputeOpTransferIndex:
    """Bit-equal to the pre-W4b maps: identity relays + blanket guard."""
    producer_map = build_computeop_producer_map(ir_graph)
    return ComputeOpTransferIndex(
        forward_producers={
            port: frozenset({src}) for port, src in producer_map.items()
        },
        effective_consumers={},
        protected_ports=build_computeop_referenced_neurons(ir_graph),
    )


def build_computeop_transfer_index(
    ir_graph: IRGraph,
    *,
    policy: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
) -> ComputeOpTransferIndex:
    """Compose per-op transfers over the graph (chains included, both ways)."""
    policy = require_computeop_liveness_transfers(policy)
    if policy == COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY:
        return _identity_only_index(ir_graph)

    ops: Dict[int, ComputeOp] = {
        n.id: n for n in ir_graph.nodes if isinstance(n, ComputeOp)
    }
    if not ops:
        return ComputeOpTransferIndex(
            forward_producers={}, effective_consumers={},
            protected_ports=frozenset(),
        )
    transfers: Dict[int, LivenessTransfer] = {
        op_id: derive_liveness_transfer(op, policy=policy)
        for op_id, op in ops.items()
    }
    forward = _compose_forward(ops, transfers)
    effective, protected = _compose_backward(ir_graph, ops, transfers)
    return ComputeOpTransferIndex(
        forward_producers=forward,
        effective_consumers=effective,
        protected_ports=protected,
    )


def _compose_forward(
    ops: Mapping[int, ComputeOp],
    transfers: Mapping[int, LivenessTransfer],
) -> Dict[Port, FrozenSet[Port]]:
    """Fixpoint: op output -> ultimate NeuralCore producers (ALL-dead => dead).

    Off-wired inputs contribute nothing (they are constant zero); model-input
    and always-on sources can never die, so regions containing them get no
    entry; opaque upstream ops likewise block the entry (conservative).
    """
    forward: Dict[Port, FrozenSet[Port]] = {}
    while True:
        changed = False
        for op_id, op in ops.items():
            transfer = transfers[op_id]
            if transfer.is_opaque:
                continue
            flat = op.input_sources.flatten()
            for out_idx, region in transfer.out_to_ins.items():
                key = (op_id, out_idx)
                if key in forward:
                    continue
                producers: Set[Port] = set()
                derivable = True
                for i in region:
                    if i >= len(flat):
                        derivable = False
                        break
                    src = flat[i]
                    if not isinstance(src, IRSource):
                        derivable = False
                        break
                    if src.is_off():
                        continue
                    if src.node_id < 0:
                        derivable = False
                        break
                    if src.node_id in ops:
                        upstream = forward.get((src.node_id, src.index))
                        if upstream is None:
                            derivable = False
                            break
                        producers |= upstream
                    else:
                        producers.add((src.node_id, src.index))
                if derivable:
                    forward[key] = frozenset(producers)
                    changed = True
        if not changed:
            return forward


def _compose_backward(
    ir_graph: IRGraph,
    ops: Mapping[int, ComputeOp],
    transfers: Mapping[int, LivenessTransfer],
) -> Tuple[Dict[Port, FrozenSet[Port]], FrozenSet[Port]]:
    """NeuralCore producer port -> through-op readers + starvation guard."""
    direct_readers: Dict[Port, Set[Port]] = defaultdict(set)
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
            continue
        for axon_idx, src in enumerate(node.input_sources.flatten()):
            if isinstance(src, IRSource) and src.node_id >= 0:
                direct_readers[(src.node_id, src.index)].add((node.id, axon_idx))

    op_readers: Dict[Port, List[Tuple[int, int]]] = defaultdict(list)
    for op_id, op in ops.items():
        for in_idx, src in enumerate(op.input_sources.flatten()):
            if isinstance(src, IRSource) and src.node_id >= 0:
                op_readers[(src.node_id, src.index)].append((op_id, in_idx))

    model_output_ports: Set[Port] = set()
    if ir_graph.output_sources.size:
        for src in ir_graph.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                model_output_ports.add((src.node_id, src.index))

    memo: Dict[Port, Tuple[FrozenSet[Port], bool]] = {}

    def resolve_port(port: Port, visiting: Set[Port]) -> Tuple[FrozenSet[Port], bool]:
        """(NeuralCore readers, terminal?) of one op-OUTPUT port."""
        cached = memo.get(port)
        if cached is not None:
            return cached
        if port in visiting:
            return frozenset(), True  # cycle: conservative terminal
        visiting.add(port)
        readers: Set[Port] = set(direct_readers.get(port, ()))
        terminal = port in model_output_ports
        for reader_op_id, in_idx in op_readers.get(port, ()):
            reader_transfer = transfers[reader_op_id]
            if reader_transfer.is_opaque:
                terminal = True
                continue
            for out_idx in reader_transfer.in_to_outs.get(in_idx, frozenset()):
                sub_readers, sub_terminal = resolve_port(
                    (reader_op_id, out_idx), visiting
                )
                readers |= sub_readers
                terminal = terminal or sub_terminal
        visiting.discard(port)
        result = (frozenset(readers), terminal)
        memo[port] = result
        return result

    op_ids = set(ops)
    effective: Dict[Port, Set[Port]] = defaultdict(set)
    protected: Set[Port] = set()
    for op_id, op in ops.items():
        transfer = transfers[op_id]
        for in_idx, src in enumerate(op.input_sources.flatten()):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            if src.node_id in op_ids:
                continue  # op-to-op edges are folded in by resolve_port
            producer_port = (src.node_id, src.index)
            if transfer.is_opaque:
                protected.add(producer_port)
                continue
            readers: Set[Port] = set()
            terminal = False
            for out_idx in transfer.in_to_outs.get(in_idx, frozenset()):
                sub_readers, sub_terminal = resolve_port((op_id, out_idx), set())
                readers |= sub_readers
                terminal = terminal or sub_terminal
            if terminal:
                protected.add(producer_port)
            else:
                effective[producer_port] |= readers
    return (
        {port: frozenset(v) for port, v in effective.items()},
        frozenset(protected),
    )
