"""Declared pruning applied to the CANDIDATE model (P) — reuse, not twin.

Two knobs, two disciplines:

- ``prune_sparsity`` (the one-shot structural shrink): the candidate runs THE
  DEPLOYED function (``prune_perceptron_chain``). Its counts are weight-
  independent, so candidate shapes equal deployed shapes by construction.
- ``pruning``/``pruning_fraction`` (the training-time pruning tuner): the
  candidate applies the mask floor-count shrink — the same ``floor(f * dim)``
  formula (``mask_prune_count``), the same IO exemptions, the cross-layer
  propagation folded to its deterministic CONSERVATIVE bound. The IR
  cascade's extra harvest is weights-dependent, so the deployed program is
  never LARGER than the candidate's model: a stated upper bound the study's
  fidelity report measures.

Foreign ``prune_criterion`` values (seed masks + cascade) keep candidate
shapes untouched — the same upper-bound statement, at zero elimination.
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.pruning.boundary_policy import (
    build_boundary_ir_graph,
    compute_perceptron_io_exemption_indices,
)
from mimarsinan.transformations.pruning.magnitude import (
    prune_perceptron_chain,
    prune_perceptron_chain_by_counts,
)
from mimarsinan.transformations.pruning.masks import mask_prune_count

ROW_COL_L1 = "row_col_l1"


def _shrink_normalization(perceptron, keep) -> None:
    """Slice a perceptron's normalization to the kept output channels.

    The DEPLOYED shrink runs post-fusion (plain Linears, no norms); the
    candidate model is pre-fusion, so its BatchNorm/LayerNorm must follow the
    kept channels or the shrunk chain cannot forward.
    """
    norm = getattr(perceptron, "normalization", None)
    if norm is None or isinstance(norm, nn.Identity) or bool(keep.all()):
        return
    kept = int(keep.sum().item())
    if isinstance(norm, nn.BatchNorm1d):
        shrunk = nn.BatchNorm1d(
            kept, eps=norm.eps, momentum=norm.momentum,
            affine=norm.affine,
            track_running_stats=norm.track_running_stats,
        )
        with torch.no_grad():
            if norm.affine:
                shrunk.weight.copy_(norm.weight[keep])
                shrunk.bias.copy_(norm.bias[keep])
            if norm.track_running_stats:
                assert (norm.running_mean is not None
                        and norm.running_var is not None
                        and shrunk.running_mean is not None
                        and shrunk.running_var is not None)
                shrunk.running_mean.copy_(norm.running_mean[keep])
                shrunk.running_var.copy_(norm.running_var[keep])
        if norm.affine:
            shrunk = shrunk.to(norm.weight.device)
        shrunk.train(norm.training)  # a fresh module defaults to train mode
        perceptron.normalization = shrunk
        return
    if isinstance(norm, nn.LayerNorm) and len(norm.normalized_shape) == 1:
        shrunk = nn.LayerNorm(kept, eps=norm.eps,
                              elementwise_affine=norm.elementwise_affine)
        with torch.no_grad():
            if norm.elementwise_affine:
                shrunk.weight.copy_(norm.weight[keep])
                shrunk.bias.copy_(norm.bias[keep])
        shrunk.train(norm.training)
        perceptron.normalization = shrunk
        return
    raise ValueError(
        f"cannot shrink a {type(norm).__name__} normalization with the pruned "
        f"channels; the candidate pruning twin only knows BatchNorm1d/LayerNorm"
    )


def tuner_shrink_counts(
    dims: Sequence[Tuple[int, int]],
    fraction: float,
    exempt_input_layers: Set[int],
    exempt_output_layers: Set[int],
) -> Tuple[List[int], List[int]]:
    """(row_counts, col_counts) of the tuner's GUARANTEED elimination.

    Per layer ``k = floor(f * dim)`` (the committed-mask formula), zeroed on
    the IO-exempt layers; a consumer adjacent to a pruned producer loses
    ``max(its own col count, the producer's row count)`` — the deterministic
    end of the mask-intersection interval.
    """
    row_counts: List[int] = []
    col_counts: List[int] = []
    for i, (in_features, out_features) in enumerate(dims):
        k_rows = (
            0 if i in exempt_output_layers
            else mask_prune_count(out_features, fraction)
        )
        k_cols = (
            0 if i in exempt_input_layers
            else mask_prune_count(in_features, fraction)
        )
        if i > 0 and dims[i - 1][1] == in_features:
            k_cols = max(k_cols, row_counts[i - 1])
        row_counts.append(k_rows)
        col_counts.append(k_cols)
    return row_counts, col_counts


def apply_declared_pruning(
    model,
    *,
    prune_sparsity: float,
    prune_criterion: str,
    pruning: bool,
    pruning_fraction: float,
    weight_bits: int,
    firing_mode: str,
) -> None:
    """Shrink the candidate model by the RUN's declared pruning, in place.

    ``pruning and pruning_fraction > 0`` mirrors ``DeploymentPlan``'s
    ``pruning_enabled`` derivation. Both knobs zero is the byte-identical
    no-op.
    """
    sparsity = float(prune_sparsity or 0.0)
    if sparsity > 0.0 and str(prune_criterion) == ROW_COL_L1:
        perceptrons = model.get_perceptrons()
        result = prune_perceptron_chain(perceptrons, sparsity)
        for perceptron, keep in zip(perceptrons, result.kept_output_masks):
            _shrink_normalization(perceptron, keep)

    fraction = float(pruning_fraction or 0.0)
    if bool(pruning) and fraction > 0.0:
        perceptrons = model.get_perceptrons()
        exempt_in, exempt_out = compute_perceptron_io_exemption_indices(
            build_boundary_ir_graph(
                model, weight_bits=int(weight_bits),
                firing_mode=str(firing_mode),
            ),
            perceptrons,
        )
        dims = [
            (int(p.layer.in_features), int(p.layer.out_features))
            for p in perceptrons
        ]
        rows, cols = tuner_shrink_counts(dims, fraction, exempt_in, exempt_out)
        kept = prune_perceptron_chain_by_counts(perceptrons, rows, cols)
        for perceptron, keep in zip(perceptrons, kept):
            _shrink_normalization(perceptron, keep)
