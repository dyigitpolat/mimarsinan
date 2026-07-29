"""W0.8b finding 5 -- activation importances DECIDE which channels are deleted.

W0.8's coverage argument put ``collect_activation_stats`` in the provably-safe
group on the claim that a NaN importance cannot masquerade as a good number. That
claim is false, and the falsification is mechanical:

  * a non-finite batch makes every ``x.abs().mean(dim=0)`` entry NaN, so the whole
    importance vector is NaN;
  * ``compute_pruning_masks_from_activations`` (and the tuner's mask builder) ranks
    channels with ``sort()``, and sorting an all-NaN vector RETURNS AN ORDER -- it
    does not raise and it does not produce a NaN mask. The bottom-k it deletes is
    just the tensor's index order;
  * masks are booleans, so nothing downstream is non-finite either. Weights are
    zeroed and the run continues.

The prune is therefore a silent, arbitrary, permanent structural decision made on a
corrupted read -- strictly worse than a wrong printed number. The path is guarded
now, and the two tests below are the ones that had to fail first: the falsification
of the old justification, and the guard that closes it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.data_handling import batch_integrity
from mimarsinan.transformations.pruning.activation import (
    collect_activation_stats,
    compute_pruning_masks_from_activations,
)


class _Model(nn.Module):
    def __init__(self, in_f=4, out_f=3):
        super().__init__()
        self.p = _Perceptron(in_f, out_f)

    def get_perceptrons(self):
        return [self.p]

    def forward(self, x):
        return self.p.layer(x)


class _Perceptron(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.layer = nn.Linear(in_f, out_f, bias=False)


class _ReplayLoader:
    """Successive PASSES yield different batch lists: poisoned once, then correct."""

    def __init__(self, passes):
        self._passes = list(passes)
        self.passes_taken = 0

    def __iter__(self):
        batches = self._passes[min(self.passes_taken, len(self._passes) - 1)]
        self.passes_taken += 1
        return iter(batches)


def _clean(n=4, in_f=4):
    x = torch.arange(1.0, n * in_f + 1.0).reshape(n, in_f)
    return x, torch.zeros(n, dtype=torch.long)


def _nan(n=4, in_f=4):
    x, y = _clean(n, in_f)
    return torch.full_like(x, float("nan")), y


class TestNanImportancesDoMasqueradeAsAGoodNumber:
    """The refuted justification, stated as an executable fact."""

    def test_sorting_an_all_nan_importance_vector_returns_an_order(self):
        importance = torch.full((5,), float("nan"))
        _, order = importance.sort()
        assert sorted(order.tolist()) == [0, 1, 2, 3, 4]

    def test_a_nan_importance_yields_a_finite_boolean_prune_mask(self):
        """Nothing downstream is non-finite, so nothing downstream can notice."""
        p = _Perceptron(4, 4)
        stats = {
            "input_importance": torch.full((4,), float("nan")),
            "output_importance": torch.full((4,), float("nan")),
        }
        row_mask, col_mask = compute_pruning_masks_from_activations(stats, p, 0.5)
        assert row_mask.dtype is torch.bool and col_mask.dtype is torch.bool
        assert int(row_mask.sum()) == 2 and int(col_mask.sum()) == 2


class TestImportanceIsMeasuredOnVerifiedReads:
    def test_a_poisoned_probe_batch_is_repaired_before_it_ranks_a_channel(self):
        loader = _ReplayLoader([[_nan()], [_clean()]])
        stats = collect_activation_stats(_Model(), loader, "cpu", num_batches=1)
        assert bool(torch.isfinite(stats[0]["input_importance"]).all())
        assert bool(torch.isfinite(stats[0]["output_importance"]).all())

    def test_persistent_corruption_fails_loud_instead_of_pruning_arbitrarily(self):
        loader = _ReplayLoader([[_nan()]])
        with pytest.raises(batch_integrity.CorruptedBatchError):
            collect_activation_stats(_Model(), loader, "cpu", num_batches=1)

    def test_a_poisoned_batch_never_reaches_the_hooks(self):
        """The accumulators are additive: one NaN forward poisons every later
        batch's contribution too, so the batch must be rejected BEFORE the
        forward, not after it."""
        loader = _ReplayLoader([[_nan(), _clean()], [_clean(), _clean()]])
        stats = collect_activation_stats(_Model(), loader, "cpu", num_batches=2)
        assert bool(torch.isfinite(stats[0]["input_importance"]).all())

    def test_clean_importances_are_value_identical(self):
        """Byte-identical default: a clean probe is read once and unchanged."""
        model = _Model()
        x, _ = _clean()
        loader = _ReplayLoader([[(x, torch.zeros(4, dtype=torch.long))]])
        stats = collect_activation_stats(model, loader, "cpu", num_batches=1)
        torch.testing.assert_close(stats[0]["input_importance"], x.abs().mean(dim=0))
        assert loader.passes_taken == 1
