"""Static per-core port structure for the liveness kernels [O1].

``input_sources``, ``consumer_axons``, the ComputeOp transfer index and the
protection sets are all fixed for the whole analysis; only the pruned sets
move, and they only ever grow. Resolving the static dispatch once per core
removes the per-port ``isinstance``, method dispatch, tuple builds and static
dict lookups from every sweep, leaving just the monotone membership tests.

(The worklist-driver dependency SSOT that briefly lived here was retired:
superseded by the flat wave engine, whose per-wave cost makes revisit
sparseness irrelevant and whose correctness is oracle-checked rather than
edge-list-trusted. See docs/elimination_flat_engine_plan.md.)
"""

from __future__ import annotations

from typing import Set


from mimarsinan.mapping.ir import IRSource

__all__ = ["PortPlan"]


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
