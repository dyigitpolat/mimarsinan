"""On-chip LayerNorm mean-centering: the realizable transformer sub-part (D5)."""

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
        # NeuralCore.execute computes inputs @ core_matrix, so core_matrix is C^T; ``sign`` selects the +C / -C rail.
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
        """Param-free two-rail merge ``pos - neg`` = the centered activation."""
        pos = buffers[self.POSITIVE_RAIL_ID]
        neg = buffers[self.NEGATIVE_RAIL_ID]
        return pos - neg

    def to_ir_graph(self) -> IRGraph:
        """Build the two-rail on-chip centering graph."""
        pos = self._rail_core(self.POSITIVE_RAIL_ID, "ln_center_pos", +1.0)
        neg = self._rail_core(self.NEGATIVE_RAIL_ID, "ln_center_neg", -1.0)
        output_sources = np.array(
            [IRSource(node_id=self.POSITIVE_RAIL_ID, index=i)
             for i in range(self.num_features)],
            dtype=object,
        )
        return IRGraph(nodes=[pos, neg], output_sources=output_sources)
