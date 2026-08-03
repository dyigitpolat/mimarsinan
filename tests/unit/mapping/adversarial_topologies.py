"""Graph shapes that break a worklist in ways the ViT vehicles cannot.

A worklist fixpoint fails SILENTLY when a dependency edge is missing: it yields
fewer kills, never wrong values, so it survives the lattice's disagreement
guard, the ``masked <= closure <= cascade`` ordering and the certificate. The
only defences are the quiescence lock and topologies that actually exercise
every edge class.

Each generator below targets one edge class:

  chain        long propagation distance -- a worklist that under-enqueues
               converges early and silently stops short
  fan_out      one producer column read by MANY cores -- the port -> readers
               edge must reach all of them, not just the first
  fan_in       one core reading MANY producers -- orphaning runs backwards, so
               the consumer -> producers edge must exist too
  bank_shared  many instances over one bank -- the bank rule is an ALL
               condition (dead in EVERY view), so any instance changing must
               re-arm the bank
  tail_seeded  seeds at the END of a chain -- exercises backward-only
               propagation, where a forward-only edge set looks complete
"""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank

__all__ = ["ADVERSARIAL_TOPOLOGIES", "build_topology"]


def _srcs(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _dense(rng, a, n):
    m = np.round(rng.normal(0, 1, (a, n)) * 8) / 8.0
    m[np.abs(m) < 0.15] = 0.0
    return m


def _chain(rng, *, depth=12, width=8):
    nodes = []
    for d in range(depth):
        srcs = [(-2, i) for i in range(width)] if d == 0 else [(d - 1, i) for i in range(width)]
        nodes.append(NeuralCore(
            id=d, name=f"c{d}", input_sources=_srcs(srcs),
            core_matrix=_dense(rng, width, width), threshold=1.0, latency=d,
        ))
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(depth - 1, 0)]))
    return graph, {0: (set(range(width // 2)), set())}


def _fan_out(rng, *, readers=24, width=8):
    nodes = [NeuralCore(id=0, name="src", input_sources=_srcs([(-2, i) for i in range(width)]),
                        core_matrix=_dense(rng, width, width), threshold=1.0, latency=0)]
    for k in range(readers):
        nodes.append(NeuralCore(
            id=k + 1, name=f"r{k}", input_sources=_srcs([(0, i) for i in range(width)]),
            core_matrix=_dense(rng, width, width), threshold=1.0, latency=1,
        ))
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(readers, 0)]))
    return graph, {0: (set(range(width // 2)), set())}


def _fan_in(rng, *, producers=24, width=8):
    nodes = [NeuralCore(id=k, name=f"p{k}", input_sources=_srcs([(-2, i) for i in range(width)]),
                        core_matrix=_dense(rng, width, width), threshold=1.0, latency=0)
             for k in range(producers)]
    srcs = [(k, i % width) for k in range(producers) for i in range(2)]
    nodes.append(NeuralCore(
        id=producers, name="sink", input_sources=_srcs(srcs),
        core_matrix=_dense(rng, len(srcs), width), threshold=1.0, latency=1,
    ))
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(producers, 0)]))
    return graph, {0: (set(range(width // 2)), set())}


def _bank_shared(rng, *, instances=32, width=8):
    bank = WeightBank(id=0, core_matrix=_dense(rng, width, width))
    nodes = [NeuralCore(
        id=k, name=f"i{k}", input_sources=_srcs([(-2, i) for i in range(width)]),
        core_matrix=None, weight_bank_id=0, weight_row_slice=(0, width),
        threshold=1.0, latency=0,
    ) for k in range(instances)]
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(instances - 1, 0)]),
                    weight_banks={0: bank})
    # seed ONE instance: the bank's ALL-condition must not fire, but every
    # instance must still be re-examined
    return graph, {0: (set(range(width // 2)), set())}


def _tail_seeded(rng, *, depth=12, width=8):
    graph, _ = _chain(rng, depth=depth, width=width)
    return graph, {depth - 1: (set(), set(range(width // 2)))}


def _reverse_chain(rng, *, depth=12, width=8):
    """A chain whose node IDS run OPPOSITE to the data flow.

    The worklist seeds its queue in node order, so a forward-numbered chain
    settles in one pass and never re-arms anything -- the forward
    (producer -> consumer) edge is never exercised. Numbering the consumer
    BEFORE its producer forces every forward step to depend on the edge.
    """
    nodes = []
    for d in range(depth):
        producer = d + 1                       # higher id feeds lower id
        srcs = ([(-2, i) for i in range(width)] if d == depth - 1
                else [(producer, i) for i in range(width)])
        nodes.append(NeuralCore(
            id=d, name=f"rc{d}", input_sources=_srcs(srcs),
            core_matrix=_dense(rng, width, width), threshold=1.0,
            latency=depth - d,
        ))
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(0, 0)]))
    # EVERY row of the deepest producer: half-seeding starves no column, so
    # nothing propagates and the forward edge is never exercised.
    return graph, {depth - 1: (set(range(width)), set())}


def _bank_sequential(rng, *, instances=16, width=8):
    """Bank instances whose deaths must re-arm the bank's ALL-condition.

    The bank rule fires only when a row is dead in EVERY sharing view, so it
    can only conclude after the LAST instance dies. Chaining the instances
    means they die one after another, and each death has to re-arm the bank.
    """
    bank = WeightBank(id=0, core_matrix=_dense(rng, width, width))
    nodes = []
    for k in range(instances):
        srcs = ([(-2, i) for i in range(width)] if k == 0
                else [(k - 1, i) for i in range(width)])
        nodes.append(NeuralCore(
            id=k, name=f"b{k}", input_sources=_srcs(srcs), core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, width),
            threshold=1.0, latency=k,
        ))
    graph = IRGraph(nodes=nodes, output_sources=_srcs([(instances - 1, 0)]),
                    weight_banks={0: bank})
    return graph, {0: (set(range(width)), set())}


ADVERSARIAL_TOPOLOGIES = {
    "reverse_chain": _reverse_chain,
    "bank_sequential": _bank_sequential,
    "chain": _chain,
    "fan_out": _fan_out,
    "fan_in": _fan_in,
    "bank_shared": _bank_shared,
    "tail_seeded": _tail_seeded,
}


def build_topology(name: str, seed: int = 0):
    """(graph, seeds_per_node) for one adversarial shape."""
    rng = np.random.default_rng(seed)
    graph, seeds = ADVERSARIAL_TOPOLOGIES[name](rng)
    return graph, seeds
