"""Deployed per-neuron survival: which output neurons of each perceptron survive pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore

__all__ = ["DeployedNeuronSurvival", "derive_deployed_neuron_survival"]


@dataclass(frozen=True)
class DeployedNeuronSurvival:
    """Immutable per-perceptron surviving original output-neuron indices in the deployed mapping.

    A perceptron absent from the map, or already at full width, is passed through unchanged.
    """

    survivors: Mapping[int, np.ndarray]

    def project(self, records: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Select each per-perceptron record's surviving-neuron columns; ``(batch, N)`` -> ``(batch, M)``.

        Identity (byte-identical no-op) for perceptrons with no survival entry or already at full width.
        """
        projected: Dict[int, np.ndarray] = {}
        for pi, vals in records.items():
            surviving = self.survivors.get(pi)
            if surviving is None or vals.shape[1] <= len(surviving):
                projected[pi] = vals
                continue
            # Survivors are the M largest values per sample: pruned NF neurons contribute exactly 0, and the multiset compare is permutation-invariant.
            m = len(surviving)
            projected[pi] = np.sort(vals, axis=1)[:, vals.shape[1] - m:]
        return projected


def _deployed_output_width(node: NeuralCore) -> int | None:
    """The core's deployed (post-compaction) output-neuron count, or ``None`` if unknown."""
    if node.core_matrix is not None:
        return int(node.core_matrix.shape[1])
    if node.weight_row_slice is not None:
        start, end = node.weight_row_slice
        return int(end) - int(start)
    return None


def derive_deployed_neuron_survival(ir_graph: IRGraph) -> DeployedNeuronSurvival:
    """Reconstruct per-perceptron surviving original output-neuron indices from the pruned ir_graph.

    The pruned ir_graph is the deployment authority. ``pre_pruning_col_mask is None`` (no pruning)
    yields full survival, so the result is an identity projection when nothing was pruned.
    """
    accumulated: Dict[int, list[int]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.psum_role not in (None, "accum"):
            continue
        perceptron_index = node.perceptron_index
        if perceptron_index is None or perceptron_index < 0:
            continue

        mask = node.pre_pruning_col_mask
        out_slice = node.perceptron_output_slice
        if out_slice is not None:
            start, end = int(out_slice[0]), int(out_slice[1])
        else:
            width = len(mask) if mask is not None else _deployed_output_width(node)
            if width is None:
                continue
            start, end = 0, int(width)

        if mask is None:
            kept = list(range(start, end))
        else:
            assert len(mask) == end - start, (
                f"pre_pruning_col_mask length {len(mask)} != output tile width "
                f"{end - start} for perceptron {perceptron_index} core {node.name}"
            )
            kept = [start + j for j, pruned in enumerate(mask) if not pruned]

        deployed_width = _deployed_output_width(node)
        assert deployed_width is None or len(kept) == deployed_width, (
            f"survivor count {len(kept)} != deployed output width {deployed_width} for "
            f"perceptron {perceptron_index} core {node.name} (pruning reconstruction is "
            f"inconsistent — bank-backed cores are not yet reconstructable this way)"
        )

        accumulated.setdefault(perceptron_index, []).extend(kept)

    return DeployedNeuronSurvival(
        survivors={
            pi: np.array(sorted(idxs), dtype=np.int64)
            for pi, idxs in accumulated.items()
        }
    )
