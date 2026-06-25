"""On-chip LayerNorm mean-centering: the realizable transformer sub-part (D5).

Mean-subtraction ``x - mean(x)`` is the FIXED linear projection
``C = I - (1/N) 11^T``.  It is the one piece of LayerNorm that fits the chip's
static-weight ``clamp(relu(W x + b))`` crossbar primitive.  The variance
division does not (it is scale-invariant -> not linear; see
``attention_mappability.py``), so this maps the centering and leaves the
reciprocal-std to a downstream host op or a learned per-channel scale fold.

Because the crossbar applies a ReLU, a signed projection ``C x`` (whose entries
take both signs) is realized as TWO rails:

    pos = clamp(relu( C x), 0, theta)   # keeps  (C x)+
    neg = clamp(relu(-C x), 0, theta)   # keeps  (C x)-

and ``C x = pos - neg``.  Both rails are genuine on-chip ``NeuralCore``s.  The
``pos - neg`` subtraction is a param-free merge (the same family as the
residual Tier-1 merge); the lock here measures the rails' bit-exactness.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore


def build_centering_matrix(num_features: int) -> np.ndarray:
    """Return ``C = I - (1/N) 11^T`` so that ``C @ x == x - mean(x)``."""
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")
    n = int(num_features)
    return np.eye(n, dtype=float) - np.ones((n, n), dtype=float) / n


class OnchipLayerNormCentering:
    """Maps LayerNorm mean-centering to a two-rail on-chip ``NeuralCore`` pair.

    The emitted :class:`IRGraph` has exactly two NeuralCores (positive /
    negative rail) and no host ``ComputeOp``: the centering is fully on chip.
    """

    POSITIVE_RAIL_ID = 0
    NEGATIVE_RAIL_ID = 1

    def __init__(self, num_features: int, clamp_ceiling: float = 1.0e6):
        if clamp_ceiling <= 0:
            raise ValueError(f"clamp_ceiling must be positive, got {clamp_ceiling}")
        self.num_features = int(num_features)
        self.clamp_ceiling = float(clamp_ceiling)
        self._centering = build_centering_matrix(self.num_features)

    def exact_input_bound(self) -> float:
        """Centered magnitude above which a rail saturates (lock no longer exact)."""
        return self.clamp_ceiling

    def _chip_input_sources(self) -> np.ndarray:
        return np.array(
            [IRSource(node_id=-2, index=i) for i in range(self.num_features)],
            dtype=object,
        )

    def _rail_core(self, node_id: int, name: str, sign: float) -> NeuralCore:
        # NeuralCore.execute computes inputs @ core_matrix, so core_matrix is
        # C^T to realize y = C x.  ``sign`` selects the +C / -C rail; the ReLU
        # then keeps that rail's nonnegative half.
        core_matrix = (sign * self._centering).T.copy()
        return NeuralCore(
            id=node_id,
            name=name,
            input_sources=self._chip_input_sources(),
            core_matrix=core_matrix,
            threshold=1.0,
            activation_scale=torch.tensor(self.clamp_ceiling),
            latency=0,
        )

    def reconstruct_centered(
        self, buffers: dict[int, "torch.Tensor"]
    ) -> "torch.Tensor":
        """Param-free two-rail merge ``pos - neg`` = the centered activation.

        ``buffers`` maps each rail's node id to its executed ``(B, N)`` output
        (the ReLU'd rail values).  The subtraction recombines them into the
        signed centered vector.
        """
        pos = buffers[self.POSITIVE_RAIL_ID]
        neg = buffers[self.NEGATIVE_RAIL_ID]
        return pos - neg

    def to_ir_graph(self) -> IRGraph:
        """Build the two-rail on-chip centering graph."""
        pos = self._rail_core(self.POSITIVE_RAIL_ID, "ln_center_pos", +1.0)
        neg = self._rail_core(self.NEGATIVE_RAIL_ID, "ln_center_neg", -1.0)
        # output_sources points at both rails (consumer subtracts neg from pos);
        # we expose the positive rail's columns as the nominal output slot so
        # the graph has a live, non-host output target.
        output_sources = np.array(
            [IRSource(node_id=self.POSITIVE_RAIL_ID, index=i)
             for i in range(self.num_features)],
            dtype=object,
        )
        return IRGraph(nodes=[pos, neg], output_sources=output_sources)
