"""Structured magnitude pruning as a selectable, default-off deployment transform (D4).

This is the deployment knob the coverage ledger enumerates as ``pruning=dense`` vs
``pruning=pruned``. Unlike the in-loop mask-and-rescale pruning in
:mod:`mimarsinan.transformations.pruning.apply` (which zeros/down-scales weights but
keeps every channel resident, so the mapped softcore shapes are unchanged), this
transform *structurally removes* the lowest-magnitude OUTPUT channels of each
perceptron and propagates the removal into the next layer's INPUT columns. That
shrinks ``out_features`` (neurons) and the downstream ``in_features`` (axons), which
is exactly what :func:`mimarsinan.mapping.verification.capacity.estimate_cores_needed`
counts — so a pruned model maps to FEWER hard cores and therefore fewer reprogram
phases (the 80% weight-DMA cost term D4 attacks).

DEFAULT-OFF / BYTE-IDENTICAL: ``sparsity == 0.0`` returns the input perceptron chain
untouched (``pruned=False``); the dense default path is provably unchanged.

EXEMPTIONS (structural soundness, mirroring the boundary policy of the in-loop
pruner): the LAST perceptron's output channels are never pruned (they are the model
logits — their count is fixed), and the FIRST perceptron's input columns are never
pruned by this pass (they are the network input; only *intermediate* axon counts can
drop, and they drop as a consequence of the upstream layer's output pruning).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import torch
import torch.nn as nn


__all__ = [
    "ChannelPruningResult",
    "kept_output_channels",
    "prune_perceptron_chain",
]


@dataclass(frozen=True)
class ChannelPruningResult:
    """Outcome of a structured channel-pruning pass over a perceptron chain.

    ``kept_output_masks[i]`` is a boolean tensor over perceptron ``i``'s output
    neurons (True = kept); the last layer is all-True (logits are exempt).
    ``pruned`` is False exactly when ``sparsity == 0.0`` (the byte-identical
    default), so a caller can branch the ledger axis on it without re-deriving.
    """

    sparsity: float
    pruned: bool
    kept_output_masks: List[torch.Tensor] = field(default_factory=list)

    @property
    def kept_output_counts(self) -> List[int]:
        return [int(m.sum().item()) for m in self.kept_output_masks]


def _row_magnitude(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-neuron L2 magnitude of an ``(out_features, in_features)`` weight."""
    return weight.detach().abs().pow(2).sum(dim=1).sqrt()


def kept_output_channels(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean keep-mask over output neurons: drop the lowest-magnitude ``sparsity`` fraction.

    Keeps at least one channel; ``floor`` of the prune count so ``sparsity`` rounds
    toward keeping. Ties break by lower index pruned first (``torch.sort`` is stable
    over the magnitude order, deterministic for a fixed weight).
    """
    out_features = weight.shape[0]
    n_prune = int(math.floor(out_features * sparsity))
    n_prune = min(n_prune, out_features - 1)
    mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
    if n_prune <= 0:
        return mask
    importance = _row_magnitude(weight)
    _, order = torch.sort(importance)
    mask[order[:n_prune]] = False
    return mask


def _shrink_linear_outputs(layer: nn.Linear, keep_mask: torch.Tensor) -> nn.Linear:
    """Return a new ``nn.Linear`` keeping only the ``keep_mask`` output rows."""
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
    new_layer = nn.Linear(
        layer.in_features,
        int(keep_idx.numel()),
        bias=layer.bias is not None,
    )
    new_layer.weight = nn.Parameter(
        layer.weight.data.index_select(0, keep_idx).clone(),
        requires_grad=layer.weight.requires_grad,
    )
    if layer.bias is not None:
        new_layer.bias = nn.Parameter(
            layer.bias.data.index_select(0, keep_idx).clone(),
            requires_grad=layer.bias.requires_grad,
        )
    return new_layer.to(layer.weight.device)


def _shrink_linear_inputs(layer: nn.Linear, keep_mask: torch.Tensor) -> nn.Linear:
    """Return a new ``nn.Linear`` keeping only the ``keep_mask`` input columns."""
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
    new_layer = nn.Linear(
        int(keep_idx.numel()),
        layer.out_features,
        bias=layer.bias is not None,
    )
    new_layer.weight = nn.Parameter(
        layer.weight.data.index_select(1, keep_idx).clone(),
        requires_grad=layer.weight.requires_grad,
    )
    if layer.bias is not None:
        new_layer.bias = nn.Parameter(
            layer.bias.data.clone(),
            requires_grad=layer.bias.requires_grad,
        )
    return new_layer.to(layer.weight.device)


def prune_perceptron_chain(
    perceptrons: Sequence[nn.Module],
    sparsity: float,
) -> ChannelPruningResult:
    """Structurally prune output neurons of each perceptron and propagate downstream.

    Mutates each perceptron IN PLACE, replacing ``perceptron.layer`` with a smaller
    ``nn.Linear`` (output rows for the pruned-out neurons removed; input columns
    removed to track the upstream layer's pruned outputs). The chain is assumed
    sequential and dense (``perceptron[i].layer.out_features ==
    perceptron[i+1].layer.in_features``); when that adjacency does not hold the
    downstream input shrink is skipped for that boundary (no-op, structurally safe).

    ``sparsity == 0.0`` is the BYTE-IDENTICAL default: returns immediately, leaving
    every ``nn.Linear`` object (and its parameter tensors) untouched.
    """
    layers = [p.layer for p in perceptrons]
    if sparsity == 0.0 or not layers:
        all_kept = [
            torch.ones(l.out_features, dtype=torch.bool, device=l.weight.device)
            for l in layers
        ]
        return ChannelPruningResult(sparsity=0.0, pruned=False, kept_output_masks=all_kept)

    n = len(layers)
    keep_masks: List[torch.Tensor] = []
    for i, layer in enumerate(layers):
        if i == n - 1:
            keep_masks.append(
                torch.ones(layer.out_features, dtype=torch.bool, device=layer.weight.device)
            )
        else:
            keep_masks.append(kept_output_channels(layer.weight.data, sparsity))

    for i, perceptron in enumerate(perceptrons):
        out_keep = keep_masks[i]
        if not bool(out_keep.all()):
            perceptron.layer = _shrink_linear_outputs(perceptron.layer, out_keep)
            perceptron.output_channels = int(out_keep.sum().item())
        if i > 0:
            in_keep = keep_masks[i - 1]
            if in_keep.numel() == perceptron.layer.in_features and not bool(in_keep.all()):
                perceptron.layer = _shrink_linear_inputs(perceptron.layer, in_keep)
                perceptron.input_features = int(in_keep.sum().item())

    return ChannelPruningResult(
        sparsity=float(sparsity), pruned=True, kept_output_masks=keep_masks
    )
