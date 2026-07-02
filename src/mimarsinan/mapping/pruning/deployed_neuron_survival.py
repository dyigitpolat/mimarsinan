"""The deployed per-neuron reality: which output neurons of each perceptron survive.

Pruning removes output neurons from a deployment two ways — whole liveness-dead cores are
dropped and zeroed columns are compacted — so the deployed mapping carries ``M <= N`` of a
perceptron's ``N`` original output neurons. The analytical NF forward, by contrast, keeps
all ``N`` (pruned ones live at 0.0). Per-neuron behavioral gates therefore cannot compare
NF against the deployed executor without first agreeing on *which* neurons are deployed.

:class:`DeployedNeuronSurvival` is that agreement made explicit: an immutable
``perceptron_index -> surviving original-neuron indices`` reality, reconstructed from the
pruned ir_graph (the deployment authority) and injected into the gates so they project
their full NF records onto the neurons that are actually deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore

__all__ = ["DeployedNeuronSurvival", "derive_deployed_neuron_survival"]


@dataclass(frozen=True)
class DeployedNeuronSurvival:
    """Per-perceptron surviving original output-neuron indices in the deployed mapping.

    ``survivors[perceptron_index]`` is the sorted array of original output-neuron indices
    that survive into the deployed (pruned) mapping. A perceptron absent from the map — or
    whose survivor count already equals a record's width — is passed through unchanged.
    """

    survivors: Mapping[int, np.ndarray]

    def project(self, records: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Select each per-perceptron record's surviving-neuron columns.

        ``records[pi]`` is ``(batch, N_original)``; returns ``(batch, M_surviving)``.
        Identity for perceptrons with no survival entry or already at full width — so the
        no-pruning path is byte-identical and this is a safe no-op wherever pruning did not
        remove neurons.
        """
        projected: Dict[int, np.ndarray] = {}
        for pi, vals in records.items():
            surviving = self.survivors.get(pi)
            if surviving is None or vals.shape[1] <= len(surviving):
                projected[pi] = vals
                continue
            # Pruned/dead neurons contribute exactly 0 to the analytical NF (weights
            # zeroed; a liveness-dead neuron cannot fire). The per-perceptron records are
            # compared as sorted multisets, so the deployed survivor value-set is the NF
            # value-set with the (N - M) pruned ZEROS removed — recovered by keeping the M
            # largest values per sample. This is permutation-invariant, so it holds even
            # when the deployed neuron order differs from the NF's flattened order (the
            # mixer's channel x spatial layout under a token-mix transpose).
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
    """Reconstruct the surviving original output-neuron indices per perceptron.

    The pruned ir_graph is the deployment authority: liveness-dead cores are simply absent,
    and each surviving owned-matrix core carries ``pre_pruning_col_mask`` (``True`` =
    pruned) over its un-reindexed ``perceptron_output_slice`` tile. Survivors of a tile are
    ``[start + j for j, pruned in enumerate(mask) if not pruned]``; a perceptron's tiles are
    unioned. ``pre_pruning_col_mask is None`` (no pruning) yields the whole slice, so the
    result is full-survival — an identity projection — when nothing was pruned.
    """
    accumulated: Dict[int, list[int]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        # A psum-decomposed perceptron emits partial cores (partial_pos/partial_neg)
        # plus one accumulator; only the accumulator (or a non-psum core) carries the
        # perceptron's output neurons — mirror the SCM record grouping, which counts
        # ``psum_role in (None, "accum")``, so survivors are not multiply attributed.
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
            # A single-tile core covers the whole perceptron (offset 0); its original
            # width is the pre-pruning mask length, else the current matrix width.
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
