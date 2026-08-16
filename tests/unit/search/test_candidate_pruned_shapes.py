"""P — pruning produces deterministic mappable shapes at candidate time, and
is a declared run parameter, never a search axis.

Two knobs, two twins:
- ``prune_sparsity`` (one-shot structural shrink): the candidate applies THE
  DEPLOYED function itself (``prune_perceptron_chain``) — its counts are
  weight-independent, so candidate shapes == deployed shapes by construction.
- ``pruning_fraction`` (the training-time pruning tuner): the candidate applies
  the mask floor-count shrink — the SAME ``k = floor(f * dim)`` formula, the
  SAME IO exemptions, the cross-layer propagation folded to its deterministic
  conservative bound (``max`` of the col count and the producer's row count).
  The IR cascade's extra harvest is weights-dependent, so the deployed program
  is never LARGER than the candidate's model: a stated upper bound.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.transformations.pruning.magnitude import (
    prune_perceptron_chain,
    prune_perceptron_chain_by_counts,
)
from mimarsinan.transformations.pruning.masks import (
    compute_all_pruning_masks,
    mask_prune_count,
)


class _Perceptron(nn.Module):
    """Minimal perceptron stand-in: the transforms only touch ``.layer``."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=True)
        self.input_features = in_features
        self.output_channels = out_features


def _chain(*dims):
    torch.manual_seed(7)
    return [
        _Perceptron(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
    ]


class TestTheMaskCountFormula:
    """The deployed tuner's per-layer elimination counts ARE the floor rule."""

    def test_realized_mask_counts_equal_the_formula(self):
        chain = _chain(64, 16, 16, 4)
        masks = compute_all_pruning_masks(
            chain, 0.25, exempt_input_layers={0}, exempt_output_layers={2},
        )
        rows = [row for row, _ in masks]
        cols = [col for _, col in masks]
        for i, p in enumerate(chain):
            out_f, _in_f = p.layer.weight.shape
            expect_rows = 0 if i == 2 else mask_prune_count(out_f, 0.25)
            assert int((~rows[i]).sum()) == expect_rows, f"layer {i} rows"
        # Un-exempt col masks eliminate AT LEAST the floor count and at most
        # floor + the producer's row count (the propagation intersection).
        for i in (1, 2):
            in_f = chain[i].layer.weight.shape[1]
            k_c = mask_prune_count(in_f, 0.25)
            k_r_prev = mask_prune_count(chain[i - 1].layer.weight.shape[0], 0.25)
            eliminated = int((~cols[i]).sum())
            assert max(k_c, k_r_prev) <= eliminated <= k_c + k_r_prev, f"layer {i}"

    def test_the_formula_is_the_floor(self):
        assert mask_prune_count(16, 0.25) == 4
        assert mask_prune_count(10, 0.25) == 2
        assert mask_prune_count(3, 0.25) == 0
        assert mask_prune_count(16, 0.0) == 0


class TestTunerShrinkCounts:
    """The propagation max-rule discriminates exactly where the consumer's own
    col count is exempt (or zero) while its producer pruned rows — for plain
    adjacent dims floor(f*in) == floor(f*out_prev) and the max is silent."""

    def test_propagation_reaches_an_input_exempt_consumer(self):
        from mimarsinan.search.problems.joint.candidate_pruning import (
            tuner_shrink_counts,
        )

        rows, cols = tuner_shrink_counts(
            [(8, 16), (16, 8)], 0.25,
            exempt_input_layers={1}, exempt_output_layers=set(),
        )
        assert rows == [4, 2]
        # cols[1] is input-exempt (own count 0) but its producer pruned 4
        # rows: the deployed mask intersection eliminates those wires anyway.
        assert cols == [2, 4]

    def test_non_adjacent_dims_do_not_propagate(self):
        from mimarsinan.search.problems.joint.candidate_pruning import (
            tuner_shrink_counts,
        )

        _rows, cols = tuner_shrink_counts(
            [(8, 16), (20, 8)], 0.25,
            exempt_input_layers={1}, exempt_output_layers=set(),
        )
        assert cols[1] == 0  # a host boundary breaks the mask propagation


class TestTheChainRule:
    """The one-shot shrink's counts: min(floor(out*f), out-1), last exempt."""

    def test_realized_chain_counts_equal_the_rule(self):
        chain = _chain(64, 16, 16, 4)
        prune_perceptron_chain(chain, 0.25)
        assert chain[0].layer.out_features == 16 - 4
        assert chain[1].layer.out_features == 16 - 4
        assert chain[2].layer.out_features == 4      # last layer exempt
        assert chain[1].layer.in_features == 12      # adjacency shrink
        assert chain[2].layer.in_features == 12
        assert chain[0].layer.in_features == 64      # network input untouched


class TestCountsBasedShrink:
    def test_shrinks_exactly_the_requested_counts(self):
        chain = _chain(64, 16, 16, 4)
        prune_perceptron_chain_by_counts(chain, [4, 4, 0], [0, 4, 4])
        assert [p.layer.out_features for p in chain] == [12, 12, 4]
        assert [p.layer.in_features for p in chain] == [64, 12, 12]

    def test_zero_counts_are_the_byte_identical_noop(self):
        chain = _chain(8, 8, 4)
        layers_before = [p.layer for p in chain]
        prune_perceptron_chain_by_counts(chain, [0, 0], [0, 0])
        assert [p.layer for p in chain] == layers_before

    def test_the_shrunk_chain_still_forwards(self):
        chain = _chain(64, 16, 16, 4)
        prune_perceptron_chain_by_counts(chain, [4, 4, 0], [0, 4, 4])
        x = torch.zeros((2, 64))
        for p in chain:
            x = p.layer(x)
        assert x.shape == (2, 4)

    def test_a_count_that_leaves_nothing_refuses(self):
        chain = _chain(8, 4)
        with pytest.raises(ValueError, match="at least one"):
            prune_perceptron_chain_by_counts(chain, [4], [8])


class TestDeSearch:
    """Pruning is a declared run parameter; its accuracy impact is unmodeled,
    so promoting it to a decision variable is refused BY NAME."""

    @pytest.mark.parametrize("key", ["pruning_fraction", "prune_sparsity"])
    def test_pruning_keys_are_refused_as_option_axes(self, key):
        with pytest.raises(ValueError, match="accuracy"):
            build_option_axes([key])

    def test_the_refusal_names_the_key(self):
        with pytest.raises(ValueError, match="pruning_fraction"):
            build_option_axes(["pruning_fraction"])

    def test_other_axes_stay_declarable(self):
        axes = build_option_axes(["weight_bits"])
        assert [axis.key for axis in axes] == ["weight_bits"]

    def test_the_dead_searched_pruning_reader_is_gone(self):
        from mimarsinan.search.problems.joint.problem import JointArchHwProblem

        assert not hasattr(JointArchHwProblem, "candidate_pruning_fraction")


class TestCandidateShapesMoveOnTheLiveProblem:
    """The live path: declared pruning moves the candidate's mapped shapes."""

    def _problem(self, **overrides):
        from unit.search.test_candidate_fragments_live_path import _cfg, _problem

        cfg = _cfg(activity_factor=0.05)
        return _problem(cfg, ["param_utilization_pct"], **overrides)

    def _model_dims(self, problem):
        model, _ = problem._candidate_model(
            {}, problem.fixed_platform_constraints, problem.encoding_placement,
        )
        return [
            (p.layer.in_features, p.layer.out_features)
            for p in model.get_perceptrons()
        ]

    def test_the_one_shot_knob_shrinks_by_the_chain_rule(self):
        dense = self._model_dims(self._problem())
        pruned = self._model_dims(self._problem(prune_sparsity=0.25))
        assert pruned != dense
        # Every non-last layer's outputs shrink by min(floor(out*f), out-1);
        # the last layer's outputs are exempt.
        for i, ((_, dense_out), (_, pruned_out)) in enumerate(
            zip(dense, pruned)
        ):
            if i == len(dense) - 1:
                assert pruned_out == dense_out
            else:
                expected = dense_out - min(
                    int(np.floor(dense_out * 0.25)), dense_out - 1,
                )
                assert pruned_out == expected, f"layer {i}"

    def test_the_tuner_knob_shrinks_by_the_floor_counts(self):
        dense = self._model_dims(self._problem())
        pruned = self._model_dims(
            self._problem(pruning=True, pruning_fraction=0.25),
        )
        assert pruned != dense
        # First layer's inputs are network-input-exempt; the last layer's
        # outputs are output-exempt; interior dims shrink by the floor rule
        # with the propagation folded to its conservative max.
        assert pruned[0][0] == dense[0][0]
        assert pruned[-1][1] == dense[-1][1]
        for i in range(len(dense) - 1):
            k_r = mask_prune_count(dense[i][1], 0.25)
            assert pruned[i][1] == dense[i][1] - k_r, f"layer {i} out"
            k_c = mask_prune_count(dense[i + 1][0], 0.25)
            assert pruned[i + 1][0] == dense[i + 1][0] - max(k_c, k_r), (
                f"layer {i + 1} in"
            )

    def test_the_shrink_is_deterministic(self):
        first = self._model_dims(self._problem(pruning=True, pruning_fraction=0.25))
        second = self._model_dims(self._problem(pruning=True, pruning_fraction=0.25))
        assert first == second

    def test_a_model_without_the_chain_api_stays_unpruned(self):
        """Conv models expose perceptrons only post-conversion (where the
        deployed shrink runs); the candidate keeps its shapes — the same
        stated upper bound, measured by fidelity."""
        from mimarsinan.search.problems.joint.candidate_pruning import (
            apply_declared_pruning,
        )

        bare = torch.nn.Linear(4, 4)
        apply_declared_pruning(
            bare, prune_sparsity=0.5, prune_criterion="row_col_l1",
            pruning=True, pruning_fraction=0.5, weight_bits=8,
            firing_mode="Default",
        )
        assert bare.out_features == 4 and bare.in_features == 4

    def test_a_foreign_criterion_stays_unpruned_as_an_upper_bound(self):
        dense = self._model_dims(self._problem())
        foreign = self._model_dims(
            self._problem(prune_sparsity=0.25, prune_criterion="activation"),
        )
        assert foreign == dense

    def test_the_candidate_census_moves_with_the_shapes(self):
        base = self._problem()
        pruned = self._problem(prune_sparsity=0.25)
        x = (np.asarray(base.xl) + np.asarray(base.xu)) / 2.0
        dense_view = base.candidate_layout(base.decode(x)).view
        pruned_view = pruned.candidate_layout(pruned.decode(x)).view
        assert (
            pruned_view.quantities.get("total_params").value
            < dense_view.quantities.get("total_params").value
        )
