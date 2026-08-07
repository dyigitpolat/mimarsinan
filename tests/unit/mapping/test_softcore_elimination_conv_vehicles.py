"""[W6c] The softcore-elimination table on CONVERTER-BUILT CONV vehicles.

Conv vehicles are the paper's headline, and they are exactly where the W6b
lexical group naming died: a bare ``nn.Sequential`` names its cores
``_0_pos4_7_g0`` / ``_4_col0`` / ``_6_col0``, every one of which peels to the
empty string, so the whole program degenerated into ONE BLANK ROW carrying the
totals of three unrelated layers. A table that blanks on the headline case is
useless, so both vehicles below are pinned end to end:

- the SAME conv vehicle the liveness-transfer DoD uses
  (``test_liveness_transfer_real_vehicles::_deep_conv_vehicle``): bank-backed
  conv over 64 positions plus two owned FCs;
- a WIDER multi-position conv: 256 weight-shared instances per conv layer, and
  a first FC wide enough that the mapper OUTPUT-TILES it into
  ``_col0_tile_{s}_{e}`` fragments — the split/coalesce suffixes that used to
  shatter one layer into one group per column.

Every assertion is arithmetic that closes: the rows partition the instances,
the cells and surviving cells add up to the program totals, and each row's
``surviving`` is the sum of its instances' own ``(R-r)(C-c)``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_arms,
)
from mimarsinan.mapping.softcore_elimination import (
    facts_from_pruning_result,
    render_softcore_elimination_markdown,
    report_from_arms,
)


def _map(model, input_shape, *, max_axons, max_neurons) -> IRGraph:
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import (
        fuse_into_perceptron,
    )

    fused = convert_torch_model(
        model, input_shape, 10, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for perceptron in fused.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    repr_ = fused.get_mapper_repr()
    repr_.assign_perceptron_indices()
    return IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=max_axons, max_neurons=max_neurons,
    ).map(repr_)


CH1_FLAT = list(range(16, 32))
CH1_ONLY_HIDDEN = (3, 7)


def _deep_conv_vehicle() -> IRGraph:
    """The repo's conv vehicle, byte-for-byte as the liveness DoD builds it."""
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2), nn.Flatten(),
        nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10),
    ).eval()
    with torch.no_grad():
        fc1 = model[4]
        keep = torch.zeros(64)
        keep[CH1_FLAT] = 1.0
        for j in CH1_ONLY_HIDDEN:
            fc1.weight[j] *= keep
            fc1.bias[j] = 0.0
    return _map(model, (1, 8, 8), max_axons=256, max_neurons=64)


def _wide_conv_vehicle() -> IRGraph:
    """A wider multi-position conv: 256 shared instances per conv layer, and a
    first FC the mapper must output-tile."""
    torch.manual_seed(11)
    model = nn.Sequential(
        nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(),
        nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2), nn.Flatten(),
        nn.Linear(24 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 10),
    ).eval()
    with torch.no_grad():
        model[2].weight[5] = 0.0          # conv2 output channel 5 is dead
        model[2].bias[5] = 0.0
        model[6].weight[:, 0:64] = 0.0    # fc1 ignores conv2 channel 0's plane
        model[8].weight[:, 3] = 0.0       # fc2 ignores fc1 unit 3
    return _map(model, (3, 16, 16), max_axons=2048, max_neurons=16)


@pytest.fixture(scope="module")
def deep_report():
    graph = _deep_conv_vehicle()
    fc2 = next(
        n for n in graph.nodes
        if isinstance(n, NeuralCore)
        and n.core_matrix is not None and n.core_matrix.shape == (17, 10)
    )
    arms = compute_elimination_arms(
        graph,
        initial_pruned_per_node={
            fc2.id: ([i == 5 for i in range(17)], [False] * 10)
        },
        initial_pruned_per_bank={0: ([False] * 10, [j == 1 for j in range(4)])},
        elimination_propagation="cascade",
    )
    return graph, arms, report_from_arms(graph, arms)


@pytest.fixture(scope="module")
def wide_report():
    graph = _wide_conv_vehicle()
    arms = compute_elimination_arms(graph, elimination_propagation="cascade")
    return graph, arms, report_from_arms(graph, arms)


def _rows(report):
    return {row.group: row for row in report.view.groups}


class TestTheRepoConvVehicleGetsOneRowPerMappedLayer:
    """W6b produced ONE row labelled `''` here. Three source layers, three
    rows, each carrying the module path the converter mapped it from."""

    def test_three_named_rows_not_one_blank_one(self, deep_report):
        _, _, report = deep_report
        rows = _rows(report)
        assert sorted(rows) == ["_0", "_4", "_6"]
        assert "" not in rows

    def test_each_row_carries_its_real_geometry(self, deep_report):
        _, _, report = deep_report
        rows = _rows(report)
        assert (rows["_0"].instances, rows["_0"].dimensions) == (64, "10x4")
        assert (rows["_4"].instances, rows["_4"].dimensions) == (1, "65x16")
        assert (rows["_6"].instances, rows["_6"].dimensions) == (1, "17x10")

    def test_the_conv_row_is_the_weight_shared_bank_instances(
        self, deep_report
    ):
        graph, _, report = deep_report
        banked = [
            n for n in graph.nodes
            if isinstance(n, NeuralCore) and n.weight_bank_id is not None
        ]
        assert _rows(report)["_0"].instances == len(banked) == 64

    def test_the_rows_partition_the_program(self, deep_report):
        _, _, report = deep_report
        rows = list(report.view.groups)
        assert sum(r.instances for r in rows) == report.total.instances == 66
        assert sum(r.cells for r in rows) == report.total.cells == 3_770
        assert sum(r.surviving for r in rows) == report.total.surviving

    def test_every_row_matches_its_instances_hand_arithmetic(self, deep_report):
        """Each row is exactly the sum over the instances that structurally
        belong to it — recomputed here straight off the facts."""
        graph, arms, report = deep_report
        facts = facts_from_pruning_result(graph, arms.final)
        by_key: dict = {}
        for inst in facts.instances:
            by_key.setdefault(inst.layer_key, []).append(inst)
        expected = sorted(
            (len(g), sum(i.cells for i in g), sum(i.surviving for i in g))
            for g in by_key.values()
        )
        assert expected == sorted(
            (row.instances, row.cells, row.surviving)
            for row in report.view.layers
        )
        for row in report.view.layers:
            assert row.surviving <= row.cells

    def test_the_table_renders_with_no_blank_label(self, deep_report):
        _, _, report = deep_report
        text = render_softcore_elimination_markdown(report)
        assert "| _0 | 64 | 10x4 |" in text
        assert "|  | " not in text.split("**TOTAL**")[0]


class TestTheWiderMultiPositionConv:
    """256 weight-shared instances per conv layer, plus the output-tiled FC
    whose `_col0_tile_{s}_{e}` fragments used to shatter into one group each."""

    def test_four_rows_one_per_source_layer(self, wide_report):
        _, _, report = wide_report
        assert sorted(_rows(report)) == ["_0", "_2", "_6", "_8"]

    def test_the_conv_layers_keep_all_their_shared_instances(self, wide_report):
        _, _, report = wide_report
        rows = _rows(report)
        assert (rows["_0"].instances, rows["_0"].dimensions) == (256, "28x12")
        assert (rows["_2"].instances, rows["_2"].dimensions) == (256, "109x24")

    def test_the_output_tiled_fc_is_one_row_not_one_per_tile(self, wide_report):
        graph, _, report = wide_report
        tiles = [
            n for n in graph.nodes
            if isinstance(n, NeuralCore) and "_tile_" in str(n.name)
        ]
        assert len(tiles) == 2, "the vehicle must actually be output-tiled"
        assert _rows(report)["_6"].instances == len(tiles)

    def test_the_rows_partition_the_program(self, wide_report):
        _, _, report = wide_report
        rows = list(report.view.groups)
        assert sum(r.instances for r in rows) == report.total.instances == 515
        assert sum(r.cells for r in rows) == report.total.cells == 805_226
        assert sum(r.surviving for r in rows) == report.total.surviving

    def test_the_dead_conv_channel_shows_up_in_the_conv_row(self, wide_report):
        _, _, report = wide_report
        row = _rows(report)["_2"]
        assert row.cols_eliminated_per_instance >= 1.0, (
            "conv2's zeroed output channel must be eliminated in every "
            "position instance"
        )

    def test_per_arm_columns_stay_monotone_per_row(self, wide_report):
        _, _, report = wide_report
        for name in _rows(report):
            fractions = [
                next(
                    row.eliminated_fraction
                    for row in report.views[arm].groups if row.group == name
                )
                for arm in report.arms
            ]
            assert fractions == sorted(fractions), (name, fractions)
