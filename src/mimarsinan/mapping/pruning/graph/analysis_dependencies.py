"""SSOT for the elimination analysis's dependency relation [O4].

The fixpoint may be driven synchronously (visit everything each sweep) or by a
worklist (visit only what a change can affect). Both are correct ONLY if the
"who reads what" relation is exact, and a missing edge fails SILENTLY: it yields
fewer kills, never wrong values, so it slips past the lattice's disagreement
guard, past ``masked <= closure <= cascade``, and past the certificate.

The defence is that no edge is ever hand-written. Every relation below is a
projection of the SAME structure the operators read, so an operator cannot start
reading something without its edge appearing:

    node.input_sources               -> which cores read a producer port
    ctx.consumer_axons               -> which cores consume a core's neurons
    computeop_transfers.*            -> relayed (non-direct) producer/consumer edges
    node.weight_bank_id              -> which instances share a bank

Bank rules are "all"-conditions (a bank row dies only when dead in EVERY sharing
instance), so a bank depends on every one of its instances -- declared once here
rather than re-derived at each call site.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from mimarsinan.mapping.ir import IRSource

__all__ = ["AnalysisDependencies", "PortPlan", "build_analysis_dependencies", "BANK", "CORE"]

CORE = "core"
BANK = "bank"

Item = Tuple[str, int]          # ("core", node_id) | ("bank", bank_id)


@dataclass
class AnalysisDependencies:
    """Reverse-reachability: what must be re-examined when an item changes."""

    # (producer_node_id, column) -> cores whose axons read that port
    port_readers: Dict[Tuple[int, int], Set[int]] = field(default_factory=lambda: defaultdict(set))
    # producer core -> cores that read ANY of its columns (the union of the above)
    core_consumers: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    # consumer core -> cores that produce into it (orphaning runs backwards)
    core_producers: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    # bank -> every sharing instance, and instance -> its bank
    bank_instances: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    core_bank: Dict[int, int] = field(default_factory=dict)

    def dependents(self, item: Item) -> Iterable[Item]:
        """Every item whose result may change because ``item`` changed.

        Rows dying in a core can orphan its PRODUCERS; columns dying can starve
        its CONSUMERS; either can move the bank's intersection. The union is
        returned rather than a direction-split set because a single refresh
        updates both row and column sets, so the caller cannot know which moved.
        """
        kind, ident = item
        if kind == CORE:
            for cid in self.core_consumers.get(ident, ()):
                yield (CORE, cid)
            for pid in self.core_producers.get(ident, ()):
                yield (CORE, pid)
            bank = self.core_bank.get(ident)
            if bank is not None:
                yield (BANK, bank)
        else:
            # a bank mask projects onto every instance that shares it
            for cid in self.bank_instances.get(ident, ()):
                yield (CORE, cid)

    def all_items(self, ctx) -> List[Item]:
        """Every item the synchronous sweep would visit, in its order."""
        items: List[Item] = [(CORE, n.id) for n in ctx.neural_cores]
        items.extend((BANK, b) for b in getattr(ctx, "banks", {}) or {})
        return items


def build_analysis_dependencies(ctx) -> AnalysisDependencies:
    """Derive the relation from the graph the operators actually read."""
    deps = AnalysisDependencies()

    for node in ctx.neural_cores:
        nid = node.id
        bank = getattr(node, "weight_bank_id", None)
        if bank is not None:
            deps.bank_instances[int(bank)].add(nid)
            deps.core_bank[nid] = int(bank)

        # direct reads: this core's axons name their producer ports
        for src in node.input_sources.flatten():
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            deps.port_readers[(src.node_id, src.index)].add(nid)
            deps.core_consumers[src.node_id].add(nid)
            deps.core_producers[nid].add(src.node_id)

    # relayed reads: a ComputeOp between two cores is wiring, not a boundary, so
    # its transfer edges are dependencies exactly like a direct read.
    transfers = getattr(ctx, "computeop_transfers", None)
    if transfers is not None:
        for port, producers in (getattr(transfers, "forward_producers", None) or {}).items():
            for reader in deps.port_readers.get(port, ()):  # who reads the op's output
                for pnid, pcol in producers:
                    deps.port_readers[(pnid, pcol)].add(reader)
                    deps.core_consumers[pnid].add(reader)
                    deps.core_producers[reader].add(pnid)
        for port, consumers in (getattr(transfers, "effective_consumers", None) or {}).items():
            pnid = port[0]
            for cid, _axon in consumers:
                deps.core_consumers[pnid].add(cid)
                deps.core_producers[cid].add(pnid)

    return deps


class PortPlan:
    """[O1] The STATIC per-core port structure both liveness kernels walk.

    ``input_sources``, ``forward_producers``, ``consumer_axons``,
    ``effective_consumers``, ``protected_ports`` and ``model_output_neurons``
    are all fixed for the whole analysis; only ``pruned_rows``/``pruned_cols``
    move, and they only ever grow. Resolving the structure once removes the
    per-port ``isinstance``, method dispatch, tuple build and static dict
    lookups from every sweep, leaving just the monotone membership tests.
    """

    __slots__ = ("always_dead_axons", "axon_producers", "orphan_now", "neuron_consumers")

    def __init__(self, node, n_neurons, computeop_transfers, consumer_axons,
                 model_output_neurons):
        nid = node.id
        fwd = computeop_transfers.forward_producers
        self.always_dead_axons: Set[int] = set()
        self.axon_producers: list = []          # (axon_index, ((nid, col), ...))
        for i, src in enumerate(node.input_sources.flatten()):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                self.always_dead_axons.add(i)
                continue
            producers = fwd.get((src.node_id, src.index))
            if producers is not None:
                self.axon_producers.append((i, tuple(producers)))
            elif src.node_id >= 0:
                self.axon_producers.append((i, ((src.node_id, src.index),)))

        protected = computeop_transfers.protected_ports
        effective = computeop_transfers.effective_consumers
        self.orphan_now: Set[int] = set()       # no consumer at all -> always orphan
        self.neuron_consumers: list = []        # (neuron, ((consumer_id, axon), ...))
        for j in range(n_neurons):
            key = (nid, j)
            if key in model_output_neurons or key in protected:
                continue
            consumers = list(consumer_axons.get(key, ()))
            consumers.extend(effective.get(key, ()))
            if not consumers:
                self.orphan_now.add(j)
            else:
                self.neuron_consumers.append((j, tuple(consumers)))


def _dead_axons_from_plan(plan: "PortPlan", pruned_cols) -> Set[int]:
    """Same answer as :func:`_cross_core_dead_axons`, over the resolved plan."""
    dead = set(plan.always_dead_axons)
    get = pruned_cols.get
    empty = frozenset()
    # C-level ``all`` over the resolved producers: the explicit Python loop it
    # replaced cost more per element than the static lookups it saved.
    dead.update(
        i for i, producers in plan.axon_producers
        if all(col in get(nid, empty) for nid, col in producers)
    )
    return dead


def _orphans_from_plan(plan: "PortPlan", pruned_rows) -> Set[int]:
    """Same answer as :func:`_orphan_neurons`, over the resolved plan."""
    dead = set(plan.orphan_now)
    get = pruned_rows.get
    empty = frozenset()
    dead.update(
        j for j, consumers in plan.neuron_consumers
        if not any(axon_i not in get(cid, empty) for cid, axon_i in consumers)
    )
    return dead
