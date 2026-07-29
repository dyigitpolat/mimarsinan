"""[W4b-2] hand-built dyadic vehicles for constant-lattice propagation.

Every core carries an always-on axon row (``IRSource(-3)``) as its constant
carrier — the crossbar bias encoding the converter emits — and every weight
sits on the 2^-4 grid, so the cascade certificate can compare bit-exactly.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore

__all__ = [
    "bias_only_collapse_graph",
    "dyadic",
    "residual_join_graph",
    "sigmoid_chain_graph",
    "srcs",
]


def srcs(specs):
    return np.array(
        [IRSource(node_id=n, index=i) for n, i in specs], dtype=object
    )


def dyadic(rng, shape, fraction_bits=4, span=8):
    ints = rng.integers(-span, span + 1, size=shape).astype(np.float64)
    return np.ldexp(ints, -fraction_bits)


def _nonzero_bias_row(matrix: np.ndarray, fraction_bits=4) -> np.ndarray:
    """Force the always-on (last) row to be non-zero everywhere: a real bias."""
    matrix[-1, :] = np.ldexp(
        np.arange(1, matrix.shape[1] + 1, dtype=np.float64), -fraction_bits
    )
    return matrix


def sigmoid_chain_graph(seed=31):
    """NC0 -> Sigmoid -> NC1 -> 2 logits.

    Sigmoid is the classic hard barrier: it is NOT zero-preserving, so W4b-1
    leaves it OPAQUE and a dead NC0 column relays nothing. Under the lattice
    the sigmoid emits CONST(0.5) — a dyadic value — which folds onto NC1's
    carrier exactly.
    """
    rng = np.random.default_rng(seed)
    core0 = NeuralCore(
        id=0, name="c0", input_sources=srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (3, 4))),
        threshold=1.0, latency=0,
    )
    op = ComputeOp(
        id=1, name="sig", input_sources=srcs([(0, j) for j in range(4)]),
        op_type="Sigmoid",
        params={"module": nn.Sigmoid().eval(), "input_shape": (4,)},
        input_shape=(4,), output_shape=(4,),
    )
    core1 = NeuralCore(
        id=2, name="c1",
        input_sources=srcs([(1, j) for j in range(4)] + [(-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (5, 2))),
        threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[core0, op, core1], output_sources=srcs([(2, 0), (2, 1)])
    )


def residual_join_graph(seed=37):
    """stem -> {skip, branch} -> add -> head (a cifar_resnet20-style block).

    The residual add is a MULTI-INPUT JOIN: opaque under W4b-1, so neither
    branch ever relays. Killing the stem's columns makes the skip line
    CONST(0) and starves the branch onto its bias, so the join folds.
    """
    rng = np.random.default_rng(seed)
    stem = NeuralCore(
        id=0, name="stem", input_sources=srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (3, 4))),
        threshold=1.0, latency=0,
    )
    branch = NeuralCore(
        id=1, name="branch",
        input_sources=srcs([(0, j) for j in range(4)] + [(-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (5, 4))),
        threshold=1.0, latency=1,
    )
    add = ComputeOp(
        id=2, name="residual_add",
        input_sources=srcs(
            [(0, j) for j in range(4)] + [(1, j) for j in range(4)]
        ),
        op_type="_operator.add",
        params={
            "module": _AddJoin().eval(),
            "input_shapes": [(4,), (4,)],
        },
        input_shape=(8,), output_shape=(4,),
    )
    head = NeuralCore(
        id=3, name="head",
        input_sources=srcs([(2, j) for j in range(4)] + [(-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (5, 2))),
        threshold=1.0, latency=2,
    )
    return IRGraph(
        nodes=[stem, branch, add, head], output_sources=srcs([(3, 0), (3, 1)])
    )


class _AddJoin(nn.Module):
    """Elementwise residual add of two branches (a real multi-input join)."""

    def forward(self, a, b):
        return a + b


def bias_only_collapse_graph(seed=41):
    """A 1x1 BIAS_ONLY core feeding a consumer, plus a live data path.

    ``bias_core`` has a single always-on axon and a single neuron: it emits one
    constant forever. Folding that constant onto the consumer's carrier frees
    the consumer's axon row, orphans the bias core's only neuron, and the
    liveness pass then DELETES the core outright.
    """
    rng = np.random.default_rng(seed)
    bias_core = NeuralCore(
        id=0, name="bias_core", input_sources=srcs([(-3, 0)]),
        core_matrix=np.array([[0.25]], dtype=np.float64),
        threshold=1.0, latency=0,
    )
    data = NeuralCore(
        id=1, name="data", input_sources=srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (3, 2))),
        threshold=1.0, latency=0,
    )
    head = NeuralCore(
        id=2, name="head",
        input_sources=srcs([(0, 0), (1, 0), (1, 1), (-3, 0)]),
        core_matrix=_nonzero_bias_row(dyadic(rng, (4, 2))),
        threshold=1.0, latency=1,
    )
    return IRGraph(
        nodes=[bias_core, data, head], output_sources=srcs([(2, 0), (2, 1)])
    )
