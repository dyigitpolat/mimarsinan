"""[W6b] Weight-cell elimination in MAPPED SOFTCORES — the headline metric.

The method targets the weights that occupy crossbar softcores, so "what
fraction of the weight CELLS in mapped softcores did we eliminate" must be a
measured per-run artifact. Every number here is checkable by hand:

- the AS-MAPPED view counts each mapped instance's own ``a x n`` cells
  (surviving = (R-r)(C-c) per instance, so a shared matrix on N crossbars
  counts N times);
- the PHYSICAL view counts shared storage ONCE and kills a bank row/column
  only when it is dead for EVERY sharing instance (the W3c intersection rule).

The two answer different questions, and the asymmetric-deadness vehicle below
pins that they genuinely differ.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_arms,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
)
from mimarsinan.mapping.softcore_elimination import (
    GEOMETRY_AS_STORED,
    SOFTCORE_ELIMINATION_RECORD_FILENAME,
    SOFTCORE_ELIMINATION_TABLE_FILENAME,
    SoftcoreEliminationError,
    render_softcore_elimination_markdown,
    report_from_arms,
    report_from_pruned_ir_graph,
    softcore_group_name,
    softcore_layer_name,
    summarize_softcore_elimination,
    write_softcore_elimination_markdown,
    write_softcore_elimination_record,
)


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object
    )


class TestGroupNamingIsDerivedFromTheNames:
    """No per-workload string table: the positional suffixes a mapper appends
    are peeled off lexically, and repeat indices collapse to one table row."""

    @pytest.mark.parametrize(
        "name,layer,group",
        [
            ("blocks_0_fc1_col36", "blocks_0_fc1", "blocks_*_fc1"),
            ("blocks_6_fc2_col64", "blocks_6_fc2", "blocks_*_fc2"),
            ("patch_embed_pos0_7_g0", "patch_embed", "patch_embed"),
            ("head_col0", "head", "head"),
            ("conv2_tile3", "conv2", "conv2"),
            ("layer_2_conv_0", "layer_2_conv", "layer_*_conv"),
            ("fc", "fc", "fc"),
            ("0", "0", "*"),
        ],
    )
    def test_layer_and_group_derivation(self, name, layer, group):
        assert softcore_layer_name(name) == layer
        assert softcore_group_name(name) == group

    def test_a_positional_word_inside_a_name_is_never_stripped(self):
        assert softcore_layer_name("column_proj") == "column_proj"
        assert softcore_layer_name("group_norm_fc") == "group_norm_fc"


def _bank_graph(masks_a, masks_b):
    """Two instances of ONE 4x4 shared bank, each with its own kill masks."""
    bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
    cores = []
    for i, (name, (rows, cols)) in enumerate(
        (("mlp_fc_col0", masks_a), ("mlp_fc_col1", masks_b))
    ):
        core = NeuralCore(
            id=i, name=name,
            input_sources=_src([(-2, j) for j in range(4)]),
            weight_bank_id=0, weight_row_slice=(0, 4),
        )
        core.pruned_row_mask = list(rows)
        core.pruned_col_mask = list(cols)
        cores.append(core)
    return IRGraph(
        nodes=cores, output_sources=_src([(1, 0)]), weight_banks={0: bank}
    )


class TestAsymmetricSharedBank:
    """A bank shared by instances with ASYMMETRIC deadness — the case where
    per-crossbar occupancy and physical weight storage must disagree.

    Instance A kills rows {0,1} and column {0}; instance B kills row {0} and
    columns {0,1}. By hand:
      as-mapped  A: (4-2)(4-1) = 6 surviving of 16;
                 B: (4-1)(4-2) = 6 surviving of 16;  => 12/32 survive (62.5% gone)
      physical   rows dead in BOTH = {0}; columns dead in EVERY covering view
                 = {0};  => (4-1)(4-1) = 9 surviving of 16 (43.75% gone)
    """

    def _report(self):
        graph = _bank_graph(
            ([True, True, False, False], [True, False, False, False]),
            ([True, False, False, False], [True, True, False, False]),
        )
        return report_from_pruned_ir_graph(graph)

    def test_as_mapped_counts_every_instance_separately(self):
        group = self._report().group("mlp_fc")
        assert (group.instances, group.axons, group.neurons) == (2, 4, 4)
        assert group.rows_eliminated == 3, "2 rows in A + 1 row in B"
        assert group.cols_eliminated == 3, "1 col in A + 2 cols in B"
        assert group.cells == 32
        assert group.surviving == 12
        assert group.eliminated == 20
        assert group.eliminated_fraction == pytest.approx(20 / 32)

    def test_physical_view_applies_the_intersection_rule(self):
        storage = self._report().storage_total
        assert storage.banks == 1
        assert storage.bank_cells_before == 16
        assert storage.bank_cells_after == 9, "only row 0 and column 0 die"
        assert storage.bank_eliminated_fraction == pytest.approx(7 / 16)

    def test_the_two_views_genuinely_differ_and_are_both_recorded(self):
        record = self._report().to_dict()
        assert record["total"]["eliminated_fraction"] == pytest.approx(20 / 32)
        assert record["storage_total"]["bank_eliminated_fraction"] == \
            pytest.approx(7 / 16)
        assert record["total"]["eliminated_fraction"] != \
            record["storage_total"]["bank_eliminated_fraction"]

    def test_symmetric_deadness_collapses_the_views_onto_one_fraction(self):
        graph = _bank_graph(
            ([True, True, False, False], [True, False, False, False]),
            ([True, True, False, False], [True, False, False, False]),
        )
        report = report_from_pruned_ir_graph(graph)
        assert report.total.eliminated_fraction == pytest.approx(
            report.storage_total.bank_eliminated_fraction
        )

    def test_a_mask_in_the_wrong_coordinates_fails_loud(self):
        graph = _bank_graph(
            ([True, True, False, False], [True, False, False, False]),
            ([True, False, False, False], [True, True, False, False]),
        )
        graph.nodes[1].pruned_row_mask = [True, False]
        with pytest.raises(SoftcoreEliminationError, match="bank coordinates"):
            report_from_pruned_ir_graph(graph)


class TestOwnedCoreArithmetic:
    def test_hand_checkable_owned_matrix(self):
        core = NeuralCore(
            id=0, name="fc1_col0",
            input_sources=_src([(-2, j) for j in range(5)]),
            core_matrix=np.ones((5, 3), dtype=np.float64),
        )
        core.pruned_row_mask = [True, True, False, False, False]
        core.pruned_col_mask = [True, False, False]
        graph = IRGraph(nodes=[core], output_sources=_src([(0, 1)]))
        report = report_from_pruned_ir_graph(graph)

        group = report.group("fc1")
        assert (group.cells, group.surviving) == (15, 3 * 2)
        assert group.eliminated_fraction == pytest.approx(9 / 15)
        storage = report.storage_total
        assert (storage.banks, storage.owned_matrices) == (0, 1)
        assert (storage.cells_before, storage.cells_after) == (15, 6)

    def test_as_stored_geometry_reads_the_post_compaction_masks(self):
        """The retrospective knob: a core whose matrix IR compaction already
        shrank keeps its pre-elimination dimensions in ``pre_pruning_*``."""
        core = NeuralCore(
            id=0, name="head_col0",
            input_sources=_src([(-2, j) for j in range(2)]),
            core_matrix=np.ones((2, 3), dtype=np.float64),
        )
        core.pre_pruning_row_mask = [True, True, False, False]
        core.pre_pruning_col_mask = [False, False, False]
        core.pruned_row_mask = [False, False]
        core.pruned_col_mask = [False, False, False]
        graph = IRGraph(nodes=[core], output_sources=_src([(0, 0)]))

        pre = report_from_pruned_ir_graph(graph)
        assert (pre.total.cells, pre.total.surviving) == (12, 6)
        stored = report_from_pruned_ir_graph(graph, geometry=GEOMETRY_AS_STORED)
        assert (stored.total.cells, stored.total.surviving) == (6, 6)
        assert stored.total.eliminated_fraction == 0.0


def _boundary_chain():
    """in_fc (model-input axons) -> mid_fc -> out_fc (model-output neurons).

    Every core carries an all-zero line that value-based seeding kills on
    sight, with CONTROLS so the exemptions are not vacuously satisfied:

    - ``in_fc`` row 2 is all-zero but reads model input data -> must SURVIVE,
      while ``mid_fc`` row 2 is all-zero and reads an ordinary always-on axon
      -> must DIE (the control for the row exemption);
    - ``out_fc`` column 1 is all-zero but is a model-output logit -> must
      SURVIVE, while ``in_fc`` column 2 is all-zero with no consumer -> must
      DIE (the control for the column exemption).
    """
    w_in = np.array(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    w_mid = np.array([[5.0, 0.0], [0.0, 6.0], [0.0, 0.0]], dtype=np.float64)
    w_out = np.array([[7.0, 0.0], [8.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    in_fc = NeuralCore(
        id=0, name="in_fc_col0",
        input_sources=_src([(-2, 0), (-2, 1), (-2, 2)]),
        core_matrix=w_in, threshold=1.0, latency=0,
    )
    mid_fc = NeuralCore(
        id=1, name="mid_fc_col0",
        input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=w_mid, threshold=1.0, latency=1,
    )
    out_fc = NeuralCore(
        id=2, name="out_fc_col0",
        input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=w_out, threshold=1.0, latency=2,
    )
    return IRGraph(
        nodes=[in_fc, mid_fc, out_fc], output_sources=_src([(2, 0), (2, 1)])
    )


class TestExemptionVisibility:
    """The boundary policy must be READABLE off the table: a model-input-fed
    core shows 0 eliminated rows, a model-output core shows 0 eliminated
    columns, even where the weights themselves are all-zero."""

    def _report(self):
        graph = _boundary_chain()
        arms = compute_elimination_arms(graph)
        return report_from_arms(graph, arms)

    def test_model_input_fed_core_shows_zero_eliminated_rows(self):
        report = self._report()
        assert report.group("in_fc").rows_eliminated == 0

    def test_the_row_control_dies_so_the_exemption_is_not_vacuous(self):
        report = self._report()
        assert report.group("mid_fc").rows_eliminated == 1

    def test_model_output_core_shows_zero_eliminated_columns(self):
        report = self._report()
        assert report.group("out_fc").cols_eliminated == 0

    def test_the_column_control_dies_so_the_exemption_is_not_vacuous(self):
        report = self._report()
        assert report.group("in_fc").cols_eliminated == 1

    def test_the_denominator_is_the_full_mapped_geometry(self):
        report = self._report()
        assert report.total.cells == (3 * 3) + (3 * 2) + (3 * 2)
        assert report.total.instances == 3


def _emergent_chain():
    """A -> B -> C with seed A.col0: closure couples one hop, the cascade
    fixpoint keeps going (starvation of B.col0, then C.row0)."""
    w_a = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]], dtype=np.float64)
    w_b = np.array([[10.0, 0.0], [0.0, 11.0], [0.0, 1.0]], dtype=np.float64)
    w_c = np.array([[5.0, 0.0], [0.0, 6.0], [0.0, 1.0]], dtype=np.float64)
    a = NeuralCore(
        id=0, name="stage_0_fc_col0", input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=w_a, threshold=1.0, latency=0,
    )
    b = NeuralCore(
        id=1, name="stage_1_fc_col0", input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=w_b, threshold=1.0, latency=1,
    )
    c = NeuralCore(
        id=2, name="stage_2_fc_col0", input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=w_c, threshold=1.0, latency=2,
    )
    graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0), (2, 1)]))
    return graph, {0: ([False, False, False], [True, False])}


class TestPerArmColumns:
    """The same table under masked / closure / cascade, so it doubles as the
    C1 evidence: each arm's column is what THAT arm would have reclaimed."""

    def _report(self, mode=ELIMINATION_PROPAGATION_CASCADE):
        graph, seeds = _emergent_chain()
        arms = compute_elimination_arms(
            graph, initial_pruned_per_node=seeds, elimination_propagation=mode
        )
        return report_from_arms(graph, arms)

    def test_every_run_arm_is_present_and_monotone(self):
        report = self._report()
        assert report.arms == (
            ELIMINATION_PROPAGATION_MASKED,
            ELIMINATION_PROPAGATION_CLOSURE,
            ELIMINATION_PROPAGATION_CASCADE,
        )
        fractions = [
            report.views[arm].total.eliminated_fraction for arm in report.arms
        ]
        assert fractions[0] < fractions[1] < fractions[2], (
            "closure must add coupling and the cascade must add emergent "
            f"propagation on this vehicle; got {fractions}"
        )

    def test_arms_share_one_denominator(self):
        report = self._report()
        cells = {report.views[arm].total.cells for arm in report.arms}
        assert len(cells) == 1, "the arm columns must be comparable"

    def test_masked_arm_reports_only_the_arm_it_ran(self):
        report = self._report(mode=ELIMINATION_PROPAGATION_MASKED)
        assert report.arms == (ELIMINATION_PROPAGATION_MASKED,)
        assert report.deployed_arm == ELIMINATION_PROPAGATION_MASKED

    def test_the_seed_only_arm_matches_the_hand_count(self):
        report = self._report(mode=ELIMINATION_PROPAGATION_MASKED)
        group = report.group("stage_*_fc")
        assert group.instances == 3
        assert group.cells == 3 * (3 * 2)
        # Only A.col0 is killed: A survives 3*(2-1)=3 cells, B and C untouched.
        assert group.surviving == 3 + 6 + 6

    def test_record_carries_every_arm(self):
        record = self._report().to_dict()
        assert set(record["per_arm"]) == set(record["arms"])
        assert record["deployed_arm"] == ELIMINATION_PROPAGATION_CASCADE
        for arm_record in record["per_arm"].values():
            assert "groups" in arm_record and "storage_total" in arm_record


class TestSerializationAndRendering:
    def _report(self):
        graph, seeds = _emergent_chain()
        return report_from_arms(
            graph, compute_elimination_arms(graph, initial_pruned_per_node=seeds)
        )

    def test_json_record_round_trips(self, tmp_path):
        path = write_softcore_elimination_record(self._report(), str(tmp_path))
        assert path.endswith(SOFTCORE_ELIMINATION_RECORD_FILENAME)
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        assert record["total"]["cells"] == 18
        assert record["groups"][0]["group"] == "stage_*_fc"
        assert record["geometry"] == "pre_elimination"

    def test_markdown_is_a_drop_in_table(self, tmp_path):
        report = self._report()
        path = write_softcore_elimination_markdown(report, str(tmp_path))
        assert path.endswith(SOFTCORE_ELIMINATION_TABLE_FILENAME)
        text = open(path, encoding="utf-8").read()
        assert "| softcore group | instances | a×n |" in text
        assert "| **TOTAL** |" in text
        assert "Physical weight storage" in text
        assert "Per-arm cell elimination" in text
        assert text == render_softcore_elimination_markdown(report)

    def test_summary_is_one_line_with_both_views(self):
        line = summarize_softcore_elimination(self._report())
        assert line.startswith("[SoftcoreElimination]")
        assert "\n" not in line
        assert "bank_cells=" in line and "eliminated=" in line

    def test_missing_group_fails_loud(self):
        with pytest.raises(KeyError, match="no softcore group"):
            self._report().group("does_not_exist")
