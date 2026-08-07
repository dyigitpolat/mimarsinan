"""Foreign-criterion seed generators for the IR pruning cascade (W3b).

The criterion-agnosticism claim needs the cascade seedable by criterion
classes demonstrably distinct from row/col L1. These tests pin, per
generator: declared mask STRUCTURE, prune RATE, DETERMINISM; that the three
criteria produce DIFFERENT seed sets on one fixture (non-identity is the
point); and that every criterion's masks flow through the ONE committed-mask
path (`committed_masks` commit/verify and
`get_initial_pruning_masks_from_model`).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.ir_pruning_masks import (
    get_initial_pruning_masks_from_model,
)
from mimarsinan.transformations.pruning.committed_masks import (
    commit_perceptron_pruning,
    verify_committed_pruning,
)
from mimarsinan.transformations.pruning.seed_generators import (
    DEFAULT_PRUNE_CRITERION,
    PRUNE_CRITERIA,
    LayerSeedMasks,
    SeedContext,
    generate_seed_masks,
    install_seed_masks,
    structured_seed_masks_from_keep,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _perceptron(out_f: int, in_f: int, weight=None, seed: int = 0):
    torch.manual_seed(seed)
    layer = nn.Linear(in_f, out_f)
    if weight is not None:
        with torch.no_grad():
            layer.weight.data = weight.clone()
    return SimpleNamespace(layer=layer)


def _chain(widths, seed: int = 0):
    """Sequential dense chain with strictly positive, all-distinct magnitudes."""
    torch.manual_seed(seed)
    perceptrons = []
    for i in range(len(widths) - 1):
        w = torch.rand(widths[i + 1], widths[i]) + 0.1
        perceptrons.append(_perceptron(widths[i + 1], widths[i], weight=w, seed=seed + i))
    return perceptrons


def _reversed_l1_stats(perceptrons):
    """Activation stats that rank importance OPPOSITE to weight L1."""
    stats = []
    for p in perceptrons:
        w = p.layer.weight.data
        stats.append({
            "output_importance": -w.abs().sum(dim=1),
            "input_importance": -w.abs().sum(dim=0),
        })
    return stats


# --------------------------------------------------------------------------- #
# Criterion registry surface                                                  #
# --------------------------------------------------------------------------- #
class TestCriterionRegistry:
    def test_registry_names_the_three_criteria(self):
        assert PRUNE_CRITERIA == ("row_col_l1", "activation", "partial_column_group")

    def test_default_criterion_is_row_col_l1(self):
        assert DEFAULT_PRUNE_CRITERION == "row_col_l1"

    def test_unknown_criterion_fails_loud(self):
        perceptrons = _chain([8, 8])
        with pytest.raises(ValueError, match="unknown prune criterion"):
            generate_seed_masks("global_magnitude", perceptrons, 0.5)


# --------------------------------------------------------------------------- #
# A. row/col L1 (the incumbent, exposed as a seed generator)                  #
# --------------------------------------------------------------------------- #
class TestRowColL1Seeds:
    def test_structure_is_full_row_col_union(self):
        perceptrons = _chain([16, 16, 16], seed=1)
        seeds = generate_seed_masks("row_col_l1", perceptrons, 0.25)
        assert len(seeds) == 2
        for seed in seeds:
            outer = seed.row_pruned.unsqueeze(1) | seed.col_pruned.unsqueeze(0)
            assert torch.equal(seed.element_pruned, outer)

    def test_rate_rows_exact_cols_within_propagation_band(self):
        f = 0.25
        perceptrons = _chain([16, 12, 16], seed=2)
        seeds = generate_seed_masks("row_col_l1", perceptrons, f)
        # rows: exactly floor(f * out_f) per layer
        assert int(seeds[0].row_pruned.sum()) == math.floor(f * 12)
        assert int(seeds[1].row_pruned.sum()) == math.floor(f * 16)
        # cols, layer 0: exactly floor(f * in_f); layer 1: own floor plus at
        # most the upstream pruned rows (cross-layer propagation is a union)
        assert int(seeds[0].col_pruned.sum()) == math.floor(f * 16)
        own = math.floor(f * 12)
        upstream = int(seeds[0].row_pruned.sum())
        assert own <= int(seeds[1].col_pruned.sum()) <= own + upstream

    def test_prunes_smallest_l1_rows(self):
        w = torch.tensor([
            [0.01, 0.01, 0.01, 0.01],
            [2.00, 2.00, 2.00, 2.00],
            [0.02, 0.02, 0.02, 0.02],
            [1.00, 1.00, 1.00, 1.00],
        ])
        perceptrons = [_perceptron(4, 4, weight=w)]
        seeds = generate_seed_masks("row_col_l1", perceptrons, 0.5)
        assert seeds[0].row_pruned.tolist() == [True, False, True, False]

    def test_exempt_layers_are_not_pruned(self):
        perceptrons = _chain([16, 16, 16], seed=3)
        ctx = SeedContext(
            exempt_input_layers=frozenset({0}), exempt_output_layers=frozenset({1}),
        )
        seeds = generate_seed_masks("row_col_l1", perceptrons, 0.5, ctx)
        assert not seeds[0].col_pruned.any()  # model input exempt
        assert not seeds[1].row_pruned.any()  # model output exempt

    def test_determinism(self):
        perceptrons = _chain([16, 16, 16], seed=4)
        a = generate_seed_masks("row_col_l1", perceptrons, 0.5)
        b = generate_seed_masks("row_col_l1", perceptrons, 0.5)
        for sa, sb in zip(a, b):
            assert torch.equal(sa.element_pruned, sb.element_pruned)


# --------------------------------------------------------------------------- #
# B. activation-importance structured seeds                                   #
# --------------------------------------------------------------------------- #
class TestActivationSeeds:
    def test_missing_stats_fail_loud(self):
        perceptrons = _chain([8, 8])
        with pytest.raises(ValueError, match="activation"):
            generate_seed_masks("activation", perceptrons, 0.5)

    def test_structure_is_full_row_col_union(self):
        perceptrons = _chain([16, 16, 16], seed=5)
        ctx = SeedContext(activation_stats=_reversed_l1_stats(perceptrons))
        seeds = generate_seed_masks("activation", perceptrons, 0.25, ctx)
        for seed in seeds:
            outer = seed.row_pruned.unsqueeze(1) | seed.col_pruned.unsqueeze(0)
            assert torch.equal(seed.element_pruned, outer)

    def test_rate_rows_exact(self):
        f = 0.25
        perceptrons = _chain([16, 16, 16], seed=6)
        ctx = SeedContext(activation_stats=_reversed_l1_stats(perceptrons))
        seeds = generate_seed_masks("activation", perceptrons, f, ctx)
        for seed in seeds:
            assert int(seed.row_pruned.sum()) == math.floor(f * 16)

    def test_ranking_follows_stats_not_weight_l1(self):
        """Anti-correlated stats prune the OPPOSITE rows to the L1 criterion."""
        perceptrons = _chain([16, 16], seed=7)
        ctx = SeedContext(activation_stats=_reversed_l1_stats(perceptrons))
        act = generate_seed_masks("activation", perceptrons, 0.5, ctx)
        l1 = generate_seed_masks("row_col_l1", perceptrons, 0.5)
        assert not (act[0].row_pruned & l1[0].row_pruned).any()

    def test_determinism(self):
        perceptrons = _chain([16, 16], seed=8)
        ctx = SeedContext(activation_stats=_reversed_l1_stats(perceptrons))
        a = generate_seed_masks("activation", perceptrons, 0.5, ctx)
        b = generate_seed_masks("activation", perceptrons, 0.5, ctx)
        assert torch.equal(a[0].element_pruned, b[0].element_pruned)


# --------------------------------------------------------------------------- #
# C. partial-column-group (Meng-style) element seeds                          #
# --------------------------------------------------------------------------- #
class TestPartialColumnGroupSeeds:
    def test_kills_are_unions_of_aligned_row_groups_per_column(self):
        gs = 4
        perceptrons = _chain([16, 16], seed=9)
        ctx = SeedContext(group_size=gs)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.5, ctx)
        el = seeds[0].element_pruned
        grouped = el.view(16 // gs, gs, 16)
        per_group_any = grouped.any(dim=1)
        per_group_all = grouped.all(dim=1)
        assert torch.equal(per_group_any, per_group_all), (
            "every killed element must belong to a fully killed aligned group"
        )

    def test_rate_is_floor_of_group_count(self):
        gs, f = 4, 0.3
        perceptrons = _chain([16, 16], seed=10)
        ctx = SeedContext(group_size=gs)
        seeds = generate_seed_masks("partial_column_group", perceptrons, f, ctx)
        n_groups = (16 // gs) * 16
        killed_groups = int(seeds[0].element_pruned.sum()) // gs
        assert killed_groups == math.floor(n_groups * f)

    def test_scores_are_group_l2_lowest_killed(self):
        """Column 2's two groups are the smallest by L2 -> exactly they die."""
        w = torch.ones(4, 4)
        w[:, 2] = 0.001
        perceptrons = [_perceptron(4, 4, weight=w)]
        ctx = SeedContext(group_size=2)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.25, ctx)
        expected = torch.zeros(4, 4, dtype=torch.bool)
        expected[:, 2] = True
        assert torch.equal(seeds[0].element_pruned, expected)
        assert seeds[0].col_pruned.tolist() == [False, False, True, False]
        assert not seeds[0].row_pruned.any()

    def test_masks_are_element_level_not_whole_row_col(self):
        """A group kill alone must NOT read as a whole pruned row or column."""
        w = torch.ones(4, 4)
        w[0:2, 0] = 0.001  # group (g=0, col=0)
        w[2:4, 3] = 0.002  # group (g=1, col=3)
        perceptrons = [_perceptron(4, 4, weight=w)]
        ctx = SeedContext(group_size=2)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.25, ctx)
        assert int(seeds[0].element_pruned.sum()) == 4  # two 2-row groups
        assert not seeds[0].row_pruned.any()
        assert not seeds[0].col_pruned.any()

    def test_completed_rows_are_harvested_as_row_seeds(self):
        """When one group-band of rows dies across ALL columns, those whole
        rows complete and surface in ``row_pruned`` (the cascade's harvest)."""
        gs = 2
        w = torch.ones(8, 4)
        w[0:gs, :] = 0.001  # rows 0..1 tiny in every column -> 4 dead groups
        perceptrons = [_perceptron(8, 4, weight=w)]
        ctx = SeedContext(group_size=gs)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.25, ctx)
        assert seeds[0].row_pruned.tolist() == [True, True] + [False] * 6
        assert not seeds[0].col_pruned.any()

    def test_non_divisible_out_features_keep_short_tail_group(self):
        gs = 4
        perceptrons = _chain([8, 10], seed=11)  # 10 rows -> groups of 4,4,2
        ctx = SeedContext(group_size=gs)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.5, ctx)
        el = seeds[0].element_pruned
        assert el.shape == (10, 8)
        # every killed element sits in a fully killed aligned group (tail = 2 rows)
        for j in range(8):
            for g, (lo, hi) in enumerate([(0, 4), (4, 8), (8, 10)]):
                block = el[lo:hi, j]
                assert block.all() or not block.any()

    def test_exempt_layers_get_no_kills(self):
        perceptrons = _chain([16, 16, 16], seed=12)
        ctx = SeedContext(
            group_size=4,
            exempt_input_layers=frozenset({0}),
            exempt_output_layers=frozenset({1}),
        )
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.5, ctx)
        assert not seeds[0].element_pruned.any()
        assert not seeds[1].element_pruned.any()

    def test_invalid_group_size_fails_loud(self):
        perceptrons = _chain([8, 8])
        with pytest.raises(ValueError, match="group_size"):
            generate_seed_masks(
                "partial_column_group", perceptrons, 0.5, SeedContext(group_size=0)
            )

    def test_determinism(self):
        perceptrons = _chain([16, 16], seed=13)
        ctx = SeedContext(group_size=4)
        a = generate_seed_masks("partial_column_group", perceptrons, 0.5, ctx)
        b = generate_seed_masks("partial_column_group", perceptrons, 0.5, ctx)
        assert torch.equal(a[0].element_pruned, b[0].element_pruned)


# --------------------------------------------------------------------------- #
# Distinctness: the whole point of the unit                                   #
# --------------------------------------------------------------------------- #
class TestCriterionDistinctness:
    def test_three_criteria_produce_pairwise_different_seed_sets(self):
        perceptrons = _chain([16, 16, 16], seed=42)
        stats = _reversed_l1_stats(perceptrons)
        by_criterion = {
            "row_col_l1": generate_seed_masks("row_col_l1", perceptrons, 0.5),
            "activation": generate_seed_masks(
                "activation", perceptrons, 0.5, SeedContext(activation_stats=stats)
            ),
            "partial_column_group": generate_seed_masks(
                "partial_column_group", perceptrons, 0.5, SeedContext(group_size=4)
            ),
        }
        names = list(by_criterion)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                different = any(
                    not torch.equal(sa.element_pruned, sb.element_pruned)
                    for sa, sb in zip(by_criterion[a], by_criterion[b])
                )
                assert different, f"{a} and {b} produced identical seed sets"


# --------------------------------------------------------------------------- #
# Committed-mask flow: install -> commit/verify -> IR seed extraction         #
# --------------------------------------------------------------------------- #
def _make_source_array(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object
    )


def _two_core_graph(in_f: int, out_f: int):
    src0 = _make_source_array([(-2, j) for j in range(in_f)] + [(-3, 0)])
    core0 = NeuralCore(
        id=0, name="c0", input_sources=src0,
        core_matrix=np.ones((in_f + 1, out_f)), threshold=1.0, latency=0,
        perceptron_index=0,
    )
    src1 = _make_source_array([(0, j) for j in range(out_f)] + [(-3, 0)])
    core1 = NeuralCore(
        id=1, name="c1", input_sources=src1,
        core_matrix=np.ones((out_f + 1, out_f)), threshold=1.0, latency=0,
        perceptron_index=1,
    )
    return IRGraph(
        nodes=[core0, core1],
        output_sources=_make_source_array([(1, j) for j in range(out_f)]),
    )


class TestCommittedMaskFlow:
    @pytest.mark.parametrize("criterion", PRUNE_CRITERIA)
    def test_install_commit_verify_zeroes_exactly_the_masked_entries(self, criterion):
        perceptrons = _chain([16, 16], seed=21)
        ctx = SeedContext(
            activation_stats=_reversed_l1_stats(perceptrons), group_size=4,
        )
        seeds = generate_seed_masks(criterion, perceptrons, 0.5, ctx)
        install_seed_masks(perceptrons, seeds)
        for p in perceptrons:
            commit_perceptron_pruning(p)
        verify_committed_pruning(perceptrons, where="test_seed_generators")
        w = perceptrons[0].layer.weight.data
        el = seeds[0].element_pruned
        assert el.any(), "fixture must actually prune something"
        assert (w[el] == 0.0).all()
        assert (w[~el] != 0.0).all(), "unmasked weights must be untouched"
        bias = perceptrons[0].layer.bias.data
        assert (bias[seeds[0].row_pruned] == 0.0).all()

    def test_installed_masks_flow_into_ir_seed_extraction(self):
        """Completed columns/rows surface as IR seeds via the 1-D mask path of
        ``get_initial_pruning_masks_from_model``; partial groups do NOT."""
        in_f = out_f = 4
        # perceptron 0: column 2 fully killed (both groups); perceptron 1:
        # two partial group kills, nothing completes.
        w0 = torch.ones(out_f, in_f)
        w0[:, 2] = 0.001
        w1 = torch.ones(out_f, out_f)
        w1[0:2, 0] = 0.001
        w1[2:4, 3] = 0.002
        perceptrons = [
            _perceptron(out_f, in_f, weight=w0),
            _perceptron(out_f, out_f, weight=w1),
        ]
        ctx = SeedContext(group_size=2)
        seeds = generate_seed_masks("partial_column_group", perceptrons, 0.25, ctx)
        install_seed_masks(perceptrons, seeds)

        graph = _two_core_graph(in_f, out_f)
        model = SimpleNamespace(get_perceptrons=lambda: perceptrons)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, graph)

        assert set(initial_node) == {0, 1}
        ir_rows0, ir_cols0 = initial_node[0]
        # IR rows = model input columns (+ bias axon, never seeded)
        assert list(ir_rows0) == [False, False, True, False, False]
        assert list(ir_cols0) == [False] * out_f
        ir_rows1, ir_cols1 = initial_node[1]
        assert list(ir_rows1) == [False] * out_f + [False]
        assert list(ir_cols1) == [False] * out_f

    def test_install_is_idempotent_and_overwrites_previous_masks(self):
        perceptrons = _chain([16, 16], seed=22)
        first = generate_seed_masks(
            "partial_column_group", perceptrons, 0.5, SeedContext(group_size=4)
        )
        install_seed_masks(perceptrons, first)
        second = generate_seed_masks("row_col_l1", perceptrons, 0.25)
        install_seed_masks(perceptrons, second)
        layer = perceptrons[0].layer
        assert torch.equal(layer.prune_mask, second[0].element_pruned)
        assert torch.equal(layer.prune_row_mask, second[0].row_pruned)
        assert torch.equal(layer.prune_col_mask, second[0].col_pruned)

    def test_length_mismatch_fails_loud(self):
        perceptrons = _chain([8, 8])
        seeds = generate_seed_masks("row_col_l1", perceptrons, 0.5)
        with pytest.raises(ValueError, match="seed"):
            install_seed_masks(perceptrons, seeds[:-1] if len(seeds) > 1 else [])


class TestLayerSeedMasksInvariants:
    def test_row_seed_without_full_element_row_fails_loud(self):
        el = torch.zeros(4, 4, dtype=torch.bool)
        row = torch.tensor([True, False, False, False])
        col = torch.zeros(4, dtype=torch.bool)
        with pytest.raises(ValueError, match="row"):
            LayerSeedMasks(element_pruned=el, row_pruned=row, col_pruned=col)

    def test_col_seed_without_full_element_col_fails_loud(self):
        el = torch.zeros(4, 4, dtype=torch.bool)
        col = torch.tensor([True, False, False, False])
        row = torch.zeros(4, dtype=torch.bool)
        with pytest.raises(ValueError, match="col"):
            LayerSeedMasks(element_pruned=el, row_pruned=row, col_pruned=col)


class TestStructuredSeedsFromKeepMasks:
    def test_matches_tuner_buffer_convention(self):
        """The keep-mask adapter is the SSOT behind the tuner's
        ``register_prune_buffers``: pruned = ~keep, element = row|col union."""
        rm = torch.tensor([True, False, True, True])
        cm = torch.tensor([True, True, False, True])
        [seed] = structured_seed_masks_from_keep([rm], [cm])
        assert torch.equal(seed.row_pruned, ~rm)
        assert torch.equal(seed.col_pruned, ~cm)
        expected = (~rm).unsqueeze(1) | (~cm).unsqueeze(0)
        assert torch.equal(seed.element_pruned, expected)
