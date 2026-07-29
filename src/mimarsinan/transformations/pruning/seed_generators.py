"""Foreign-criterion SEED GENERATORS for the IR pruning cascade (criterion agnosticism).

The cascade consumes committed layer masks (``prune_mask`` / ``prune_row_mask`` /
``prune_col_mask`` / ``prune_bias_mask`` buffers) through the ONE path
``committed_masks`` commit/verify -> ``get_initial_pruning_masks_from_model``.
Every generator here emits per-perceptron :class:`LayerSeedMasks` in exactly
that format, so criteria are interchangeable at the selection seam:

- ``row_col_l1``   — the incumbent: whole rows/cols by weight L1 (reuses
  ``masks.compute_all_pruning_masks``).
- ``activation``   — whole rows/cols by measured activation importance (reuses
  the same machinery fed with ``activation.collect_activation_stats`` output).
- ``partial_column_group`` — Meng-style ELEMENT masks: each weight column is
  partitioned into contiguous ``group_size``-row groups scored by L2 norm and
  the lowest-scoring fraction is zeroed. A group kill does not by itself
  remove a whole row or column — the cascade then harvests whichever whole
  rows/cols the group kills complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from mimarsinan.transformations.pruning.masks import compute_all_pruning_masks

DEFAULT_PRUNE_CRITERION = "row_col_l1"
DEFAULT_PARTIAL_COLUMN_GROUP_SIZE = 8


@dataclass(frozen=True)
class SeedContext:
    """Criterion inputs beyond (perceptrons, fraction); workload-agnostic.

    ``activation_stats`` is the per-perceptron list produced by
    ``activation.collect_activation_stats`` (required by ``activation``);
    ``group_size`` is the rows-per-column-group of ``partial_column_group``;
    the exempt sets are perceptron indices whose model-I/O boundary must stay
    intact (``boundary_policy.compute_perceptron_io_exemption_indices``).
    """

    activation_stats: Optional[Sequence[dict]] = None
    group_size: int = DEFAULT_PARTIAL_COLUMN_GROUP_SIZE
    exempt_input_layers: frozenset = frozenset()
    exempt_output_layers: frozenset = frozenset()


@dataclass(frozen=True)
class LayerSeedMasks:
    """One layer's seed in the committed-mask format (True = pruned).

    ``element_pruned`` is the (out_features, in_features) element mask;
    ``row_pruned`` / ``col_pruned`` are the whole-row / whole-column seeds the
    IR path reads. Invariant (fail loud): a row/col seed implies the full
    element row/col is pruned — the 1-D seeds never claim more than the
    element mask commits.
    """

    element_pruned: torch.Tensor
    row_pruned: torch.Tensor
    col_pruned: torch.Tensor

    def __post_init__(self):
        out_f, in_f = self.element_pruned.shape
        if self.row_pruned.shape != (out_f,) or self.col_pruned.shape != (in_f,):
            raise ValueError(
                f"seed mask shape mismatch: element {tuple(self.element_pruned.shape)} "
                f"vs row {tuple(self.row_pruned.shape)} / col {tuple(self.col_pruned.shape)}"
            )
        if bool((self.row_pruned & ~self.element_pruned.all(dim=1)).any()):
            raise ValueError(
                "row seed claims a row whose element mask is not fully pruned"
            )
        if bool((self.col_pruned & ~self.element_pruned.all(dim=0)).any()):
            raise ValueError(
                "col seed claims a column whose element mask is not fully pruned"
            )


def structured_seed_masks_from_keep(
    row_keep_masks: Sequence[torch.Tensor],
    col_keep_masks: Sequence[torch.Tensor],
) -> List[LayerSeedMasks]:
    """Adapt (row_keep, col_keep) mask pairs (True = kept) to seed masks.

    The SSOT for the committed-buffer convention shared with the pruning
    tuner's ``register_prune_buffers``: pruned = ~keep, element = row|col union.
    """
    seeds: List[LayerSeedMasks] = []
    for rm, cm in zip(row_keep_masks, col_keep_masks):
        row_pruned = ~rm
        col_pruned = ~cm
        seeds.append(LayerSeedMasks(
            element_pruned=row_pruned.unsqueeze(1) | col_pruned.unsqueeze(0),
            row_pruned=row_pruned,
            col_pruned=col_pruned,
        ))
    return seeds


def _structured_masks(perceptrons, fraction: float, context: SeedContext,
                      activation_stats) -> List[LayerSeedMasks]:
    pairs = compute_all_pruning_masks(
        perceptrons,
        fraction,
        set(context.exempt_input_layers),
        set(context.exempt_output_layers),
        activation_stats=activation_stats,
    )
    return structured_seed_masks_from_keep(
        [rm for rm, _ in pairs], [cm for _, cm in pairs]
    )


def row_col_l1_seed_masks(perceptrons, fraction: float,
                          context: SeedContext) -> List[LayerSeedMasks]:
    """Whole rows/cols by weight L1 with cross-layer propagation (incumbent)."""
    return _structured_masks(perceptrons, fraction, context, activation_stats=None)


def activation_seed_masks(perceptrons, fraction: float,
                          context: SeedContext) -> List[LayerSeedMasks]:
    """Whole rows/cols by measured activation importance (foreign criterion)."""
    if context.activation_stats is None:
        raise ValueError(
            "the 'activation' prune criterion requires collected activation "
            "stats (SeedContext.activation_stats); refusing to fall back to "
            "weight L1 silently"
        )
    return _structured_masks(
        perceptrons, fraction, context, activation_stats=list(context.activation_stats)
    )


def partial_column_group_seed_masks(perceptrons, fraction: float,
                                    context: SeedContext) -> List[LayerSeedMasks]:
    """Meng-style partial-column-group element seeds (foreign criterion).

    Each column is partitioned into contiguous ``group_size``-row groups
    (the tail group may be shorter); all groups of a layer are scored by L2
    norm and the ``floor(n_groups * fraction)`` lowest-scoring die (stable
    sort: ties break toward lower (group, column) index — deterministic).
    Exempt layers get no kills. The 1-D seeds are exactly the COMPLETED
    rows/cols of the element mask.
    """
    group_size = int(context.group_size)
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    seeds: List[LayerSeedMasks] = []
    for i, p in enumerate(perceptrons):
        weight = p.layer.weight.data
        out_f, in_f = weight.shape
        element = torch.zeros(out_f, in_f, dtype=torch.bool, device=weight.device)
        exempt = (
            i in context.exempt_input_layers or i in context.exempt_output_layers
        )
        if not exempt:
            n_groups = math.ceil(out_f / group_size)
            padded = torch.zeros(
                n_groups * group_size, in_f,
                dtype=weight.dtype, device=weight.device,
            )
            padded[:out_f] = weight.detach() ** 2
            # (n_groups, in_f) group L2 scores
            scores = padded.view(n_groups, group_size, in_f).sum(dim=1).sqrt()
            flat = scores.flatten()
            n_kill = int(math.floor(flat.numel() * fraction))
            if n_kill > 0:
                order = torch.argsort(flat, stable=True)
                kill_flat = torch.zeros_like(flat, dtype=torch.bool)
                kill_flat[order[:n_kill]] = True
                kill_groups = kill_flat.view(n_groups, in_f)
                element = kill_groups.repeat_interleave(group_size, dim=0)[:out_f]
        seeds.append(LayerSeedMasks(
            element_pruned=element,
            row_pruned=element.all(dim=1),
            col_pruned=element.all(dim=0),
        ))
    return seeds


SEED_GENERATORS: Dict[
    str, Callable[[Sequence, float, SeedContext], List[LayerSeedMasks]]
] = {
    "row_col_l1": row_col_l1_seed_masks,
    "activation": activation_seed_masks,
    "partial_column_group": partial_column_group_seed_masks,
}

PRUNE_CRITERIA: Tuple[str, ...] = tuple(SEED_GENERATORS)


def generate_seed_masks(criterion: str, perceptrons, fraction: float,
                        context: Optional[SeedContext] = None) -> List[LayerSeedMasks]:
    """Dispatch to the named criterion's generator; unknown names fail loud."""
    generator = SEED_GENERATORS.get(criterion)
    if generator is None:
        raise ValueError(
            f"unknown prune criterion {criterion!r}; known criteria: "
            f"{sorted(SEED_GENERATORS)}"
        )
    return generator(perceptrons, float(fraction), context or SeedContext())


def install_seed_masks(perceptrons, seeds: Sequence[LayerSeedMasks]) -> None:
    """Register each layer's seed as the committed prune-mask buffers.

    Overwrites any previously installed masks (re-registering a buffer name is
    an overwrite in torch). The commit/verify path
    (``committed_masks.commit_perceptron_pruning`` /
    ``verify_committed_pruning``) and the IR seed extraction
    (``get_initial_pruning_masks_from_model``) read exactly these buffers.
    """
    if len(perceptrons) != len(seeds):
        raise ValueError(
            f"seed count {len(seeds)} does not match perceptron count "
            f"{len(perceptrons)}"
        )
    for p, seed in zip(perceptrons, seeds):
        layer = p.layer
        out_f, in_f = layer.weight.shape
        if seed.element_pruned.shape != (out_f, in_f):
            raise ValueError(
                f"seed element mask {tuple(seed.element_pruned.shape)} does not "
                f"match layer weight {(out_f, in_f)}"
            )
        layer.register_buffer("prune_mask", seed.element_pruned.clone())
        layer.register_buffer("prune_row_mask", seed.row_pruned.clone())
        layer.register_buffer("prune_col_mask", seed.col_pruned.clone())
        if getattr(layer, "bias", None) is not None:
            layer.register_buffer("prune_bias_mask", seed.row_pruned.clone())
