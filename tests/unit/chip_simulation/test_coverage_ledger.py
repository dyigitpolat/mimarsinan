"""Frontier E1 — the hypervolume coverage ledger: genericity as a MEASURED fraction.

Locks the mechanism the program plan (E1) calls for:

* a typed HYPERVOLUME AXIS MODEL — the product of deployment axes, each classified
  ORTHOGONAL (covered marginally) vs INTERACTING (tested jointly), with the cheap-
  screening justification (``encoding_placement`` offload≡subsume under signed-IF ⇒
  collapse the axis);
* a FULL-TUPLE HypervolumeCell key that extends the (firing × sync × backend)
  ``CertificationCell`` to the full config tuple, and round-trips through a canonical
  string;
* ``coverage_report(rows, claimed_subproduct=None)`` — a GROUP BY over the ledger by
  cell-key + the ``deployment_validity`` tier that reports, for each cell in the
  claimed sub-product, a status in {VALID, VALID_FLAGGED, INVALID, untested}, the
  measured COVERAGE FRACTION, the named UNTESTED frontier, and the RESEARCH-GAP
  frontier (the union of ``research_gap_ops`` over VALID_FLAGGED cells).
"""

import json

import pytest

from mimarsinan.chip_simulation.coverage_ledger import (
    AXES,
    AxisKind,
    CoverageStatus,
    HypervolumeAxis,
    HypervolumeCell,
    classify_validity_tier,
    claimed_subproduct,
    collapse_orthogonal_axes,
    coverage_report,
    row_to_cell,
    row_to_cells,
)
from mimarsinan.chip_simulation.certification import CertificationCell


# --------------------------------------------------------------------------- #
# HypervolumeAxis — the typed product of deployment axes + classification.
# --------------------------------------------------------------------------- #

class TestAxisModel:
    def test_axes_are_named_and_classified(self):
        # Every axis in the program plan's product is modeled and carries a kind.
        names = {a.name for a in AXES}
        expected = {
            "firing",
            "sync",
            "encoding_placement",
            "quantization",
            "pruning",
            "backend",
            "mapping_strategy",
            "S",
            "vehicle",
            "dataset",
            "regime",
        }
        assert expected <= names
        for axis in AXES:
            assert axis.kind in (AxisKind.ORTHOGONAL, AxisKind.INTERACTING)

    def test_firing_and_sync_are_interacting(self):
        # (firing × sync) is tested JOINTLY (the death-cascade is a firing×sync result).
        assert HypervolumeAxis.get("firing").kind is AxisKind.INTERACTING
        assert HypervolumeAxis.get("sync").kind is AxisKind.INTERACTING

    def test_quantization_is_interacting_with_firing(self):
        # quantization × firing is tested jointly (quant interacts with the firing law).
        axis = HypervolumeAxis.get("quantization")
        assert axis.kind is AxisKind.INTERACTING
        assert "firing" in axis.interacts_with

    def test_encoding_placement_is_orthogonal_and_collapses(self):
        # The checkpoint's cheap-screen result: offload≡subsume under signed-IF ⇒
        # the axis is ORTHOGONAL and collapses to a single representative value.
        axis = HypervolumeAxis.get("encoding_placement")
        assert axis.kind is AxisKind.ORTHOGONAL
        assert axis.collapsed is True
        assert axis.justification  # the cheap-screening reason is documented

    def test_collapse_drops_collapsed_axes_to_one_value(self):
        collapsed = collapse_orthogonal_axes(AXES)
        names = {a.name for a in collapsed}
        # The collapsed encoding_placement axis is gone from the active product.
        assert "encoding_placement" not in names
        # Interacting axes survive.
        assert {"firing", "sync"} <= names


# --------------------------------------------------------------------------- #
# HypervolumeCell — the FULL-TUPLE cell key.
# --------------------------------------------------------------------------- #

class TestHypervolumeCell:
    def _cell(self, **over):
        base = dict(
            firing="ttfs_cycle_based",
            sync="cascaded",
            backend="sanafe",
            dataset="mnist",
            vehicle="deep_cnn",
            regime="from_scratch",
            quantization="wq",
            pruning="dense",
            mapping_strategy="packed",
            s="4",
        )
        base.update(over)
        return HypervolumeCell(**base)

    def test_key_round_trips(self):
        cell = self._cell()
        assert HypervolumeCell.from_key(cell.cell_key) == cell

    def test_key_extends_certification_cell(self):
        cell = self._cell()
        # The (firing × sync × backend) sub-coordinate is a real CertificationCell.
        cert = cell.cert_cell
        assert isinstance(cert, CertificationCell)
        assert cert.cell_key == "ttfs_cycle_based/cascaded@sanafe"
        # The full-tuple key is STRICTLY richer than the cert key.
        assert cell.cell_key != cert.cell_key
        assert cert.cell_key in cell.cell_key

    def test_collapsed_axis_is_absent_from_the_key(self):
        # encoding_placement is collapsed, so it is NOT a coordinate of the cell key.
        cell = self._cell()
        assert "subsume" not in cell.cell_key
        assert "offload" not in cell.cell_key

    def test_distinct_cells_have_distinct_keys(self):
        a = self._cell(dataset="mnist")
        b = self._cell(dataset="fmnist")
        assert a.cell_key != b.cell_key


# --------------------------------------------------------------------------- #
# Tier normalization — the ledger's free-form validity strings → canonical tier.
# --------------------------------------------------------------------------- #

class TestTierNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("VALID_on_chip_majority", CoverageStatus.VALID),
            ("VALID_on_chip_majority_rc0", CoverageStatus.VALID),
            ("VALID_clean_rc0_all_12_runs__on_chip_majority", CoverageStatus.VALID),
            ("VALID_FLAGGED_placement", CoverageStatus.VALID_FLAGGED),
            ("INVALID_host_majority", CoverageStatus.INVALID),
            ("INVALID_rc1_crash", CoverageStatus.INVALID),
            (None, None),
            ("", None),
            ("FINALIZED_rc0", None),  # not a validity tier → not a science cell
        ],
    )
    def test_normalizes_ledger_tier_strings(self, raw, expected):
        assert classify_validity_tier(raw) == expected


# --------------------------------------------------------------------------- #
# coverage_report — the GROUP BY tally over a synthetic ledger.
# --------------------------------------------------------------------------- #

def _row(model, dataset, schedule, tier, **extra):
    row = {
        "model": model,
        "dataset": dataset,
        "schedule": schedule,
        "spiking_mode": "ttfs_cycle_based",
        "backend": "sanafe",
        "deployment_validity": tier,
    }
    row.update(extra)
    return row


class TestCoverageReport:
    def _synthetic_ledger(self):
        # Four science cells across the (vehicle × dataset × sync) sub-product, one
        # per tier, plus a non-science row that must be ignored.
        return [
            _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority"),
            _row("deep_cnn", "fmnist", "cascaded", "VALID_FLAGGED_placement",
                 research_gap_ops=["MultiheadAttention"]),
            _row("deep_mlp", "mnist", "cascaded", "INVALID_host_majority"),
            _row("deep_cnn", "mnist", "synchronized", "VALID_on_chip_majority"),
            # Non-science row: no validity tier → contributes nothing.
            _row("deep_cnn", "kmnist", "cascaded", "FINALIZED_rc0"),
        ]

    def test_tier_tally_is_correct(self):
        report = coverage_report(self._synthetic_ledger())
        assert report.tier_counts[CoverageStatus.VALID] == 2
        assert report.tier_counts[CoverageStatus.VALID_FLAGGED] == 1
        assert report.tier_counts[CoverageStatus.INVALID] == 1
        # The non-science row is not counted as a covered cell.
        assert report.covered_cell_count == 4

    def test_coverage_fraction_over_a_claimed_subproduct(self):
        ledger = self._synthetic_ledger()
        # Claim the product {deep_cnn} × {mnist, fmnist} × {cascaded, synchronized}.
        claimed = claimed_subproduct(
            vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist"],
            sync=["cascaded", "synchronized"],
        )
        report = coverage_report(ledger, claimed_subproduct=claimed)
        # 4 claimed cells; covered = (mnist,cascaded), (fmnist,cascaded),
        # (mnist,synchronized) = 3. (fmnist,synchronized) is untested.
        assert report.claimed_cell_count == 4
        assert report.covered_claimed_count == 3
        assert report.coverage_fraction == pytest.approx(3 / 4)

    def test_untested_frontier_is_named(self):
        ledger = self._synthetic_ledger()
        claimed = claimed_subproduct(
            vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist"],
            sync=["cascaded", "synchronized"],
        )
        report = coverage_report(ledger, claimed_subproduct=claimed)
        untested_keys = {c.cell_key for c in report.untested_frontier}
        # The one untested cell is deep_cnn × fmnist × synchronized.
        assert len(report.untested_frontier) == 1
        only = next(iter(report.untested_frontier))
        assert only.vehicle == "deep_cnn"
        assert only.dataset == "fmnist"
        assert only.sync == "synchronized"
        # And it is reported as status untested in the per-cell table.
        assert report.status_for(only) is CoverageStatus.UNTESTED
        assert untested_keys

    def test_research_gap_frontier_unions_flagged_ops(self):
        ledger = self._synthetic_ledger()
        report = coverage_report(ledger)
        # The lone VALID_FLAGGED cell carries MultiheadAttention as its gap op.
        assert "MultiheadAttention" in report.research_gap_frontier

    def test_research_gap_frontier_dedupes_across_flagged_cells(self):
        ledger = [
            _row("deep_cnn", "mnist", "cascaded", "VALID_FLAGGED_placement",
                 research_gap_ops=["MultiheadAttention", "LayerNorm"]),
            _row("deep_cnn", "fmnist", "cascaded", "VALID_FLAGGED_placement",
                 research_gap_ops=["MultiheadAttention"]),
        ]
        report = coverage_report(ledger)
        assert report.research_gap_frontier == ["LayerNorm", "MultiheadAttention"]

    def test_status_for_uncovered_claimed_cell_is_untested(self):
        # An empty ledger → every claimed cell is untested; fraction 0.
        claimed = claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        report = coverage_report([], claimed_subproduct=claimed)
        assert report.coverage_fraction == 0.0
        assert report.claimed_cell_count == 1
        assert report.covered_claimed_count == 0
        assert len(report.untested_frontier) == 1

    def test_dual_mean_row_covers_two_cells_in_the_tally(self):
        # One arch_dataset row reporting both schedule means → two covered cells.
        row = _row("deep_cnn", "mnist", None, "VALID_on_chip_majority")
        row["cascaded_deployed_mean"] = 0.98
        row["synchronized_deployed_mean"] = 0.99
        report = coverage_report([row])
        assert report.covered_cell_count == 2
        assert report.tier_counts[CoverageStatus.VALID] == 2

    def test_worst_tier_wins_when_a_cell_has_conflicting_rows(self):
        # Two rows land on the SAME cell with different tiers; the report keeps the
        # worst (most conservative) tier for that cell.
        ledger = [
            _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority"),
            _row("deep_cnn", "mnist", "cascaded", "VALID_FLAGGED_placement",
                 research_gap_ops=["LayerNorm"]),
        ]
        report = coverage_report(ledger)
        assert report.covered_cell_count == 1
        cells = list(report.cell_status.keys())
        assert report.cell_status[cells[0]] is CoverageStatus.VALID_FLAGGED
        # And the flagged op still surfaces in the research-gap frontier.
        assert "LayerNorm" in report.research_gap_frontier

    def test_invalid_demoted_cell_drops_its_flag_from_the_frontier(self):
        # A cell tested both FLAGGED and INVALID resolves to INVALID (conservative);
        # its flag op is therefore NOT a flagged-cell research target.
        ledger = [
            _row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op",
                 research_gap_ops=["MultiheadAttention"]),
            _row("vit_b", "mnist", "cascaded", "INVALID_host_majority"),
        ]
        report = coverage_report(ledger)
        cells = list(report.cell_status.keys())
        assert report.cell_status[cells[0]] is CoverageStatus.INVALID
        assert report.research_gap_frontier == []


# --------------------------------------------------------------------------- #
# Orthogonality collapse end-to-end in the claimed product.
# --------------------------------------------------------------------------- #

class TestOrthogonalityCollapse:
    def test_claimed_product_ignores_collapsed_encoding_placement(self):
        # Asking for both encoding placements collapses to a single representative;
        # the claimed product does NOT double in size.
        claimed_one = claimed_subproduct(vehicle=["deep_cnn"], dataset=["mnist"])
        claimed_both = claimed_subproduct(
            vehicle=["deep_cnn"],
            dataset=["mnist"],
            encoding_placement=["subsume", "offload"],
        )
        assert len(claimed_one) == len(claimed_both)

    def test_report_serializes_to_json(self):
        ledger = [
            _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority"),
            _row("deep_cnn", "fmnist", "cascaded", "VALID_FLAGGED_placement",
                 research_gap_ops=["MultiheadAttention"]),
        ]
        report = coverage_report(ledger)
        blob = json.dumps(report.to_dict())
        back = json.loads(blob)
        assert back["covered_cell_count"] == 2
        assert "MultiheadAttention" in back["research_gap_frontier"]


# --------------------------------------------------------------------------- #
# row_to_cell — mapping a heterogeneous ledger row onto a cell.
# --------------------------------------------------------------------------- #

class TestRowToCell:
    def test_non_sync_schedule_token_normalizes_to_none(self):
        # Run-status leakage in the `schedule` field (finalize_attempt, offload, all)
        # must NOT spawn a spurious sync cell — it normalizes to sync=none.
        for junk in ("finalize_attempt", "offload", "all", "blocked_attempt", None):
            cell = row_to_cell(_row("deep_mlp", "mnist", junk, "VALID_on_chip_majority"))
            assert cell is not None
            assert cell.sync == "none"

    def test_valid_sync_tokens_survive(self):
        for sync in ("cascaded", "synchronized"):
            cell = row_to_cell(_row("deep_cnn", "mnist", sync, "VALID_on_chip_majority"))
            assert cell.sync == sync

    def test_non_science_row_maps_to_none(self):
        assert row_to_cell(_row("deep_cnn", "mnist", "cascaded", "FINALIZED_rc0")) is None

    def test_row_without_model_maps_to_none(self):
        row = _row(None, "mnist", "cascaded", "VALID_on_chip_majority")
        assert row_to_cell(row) is None

    def test_dataset_aliases_normalize(self):
        a = row_to_cell(_row("deep_cnn", "FashionMNIST", "cascaded", "VALID_on_chip_majority"))
        b = row_to_cell(_row("deep_cnn", "fmnist", "cascaded", "VALID_on_chip_majority"))
        assert a.dataset == b.dataset == "fmnist"

    def test_dual_mean_row_expands_to_both_sync_cells(self):
        # An arch_dataset row reporting BOTH schedule means covers BOTH sync cells.
        row = _row("deep_cnn", "mnist", None, "VALID_on_chip_majority")
        row["cascaded_deployed_mean"] = 0.98
        row["synchronized_deployed_mean"] = 0.99
        cells = row_to_cells(row)
        syncs = {c.sync for c in cells}
        assert syncs == {"cascaded", "synchronized"}

    def test_single_schedule_row_yields_one_cell(self):
        row = _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority")
        cells = row_to_cells(row)
        assert len(cells) == 1
        assert cells[0].sync == "cascaded"

    def test_row_to_cell_returns_the_first_of_row_to_cells(self):
        row = _row("deep_cnn", "mnist", "cascaded", "VALID_on_chip_majority")
        assert row_to_cell(row) == row_to_cells(row)[0]


# --------------------------------------------------------------------------- #
# Research-gap vs placement-fixable frontiers — the two flag categories.
# --------------------------------------------------------------------------- #

class TestFlagFrontiers:
    def test_placement_fixable_ops_are_not_research_gaps(self):
        ledger = [
            _row("deep_cnn", "mnist", "cascaded", "VALID_FLAGGED_placement",
                 placement_fixable_ops=["Linear"]),
        ]
        report = coverage_report(ledger)
        # A placement-fixable encoder is reported separately, NOT as a research gap.
        assert report.research_gap_frontier == []
        assert "Linear" in report.placement_fixable_frontier

    def test_live_ledger_placement_suffix_falls_back_to_placement_frontier(self):
        # A live-ledger flagged row carries NO structured ops, only the tier suffix.
        ledger = [_row("deep_mlp", "mnist", "cascaded", "VALID_FLAGGED_placement")]
        report = coverage_report(ledger)
        # The _placement suffix → placement-fixable, not a research gap.
        assert report.research_gap_frontier == []
        assert report.placement_fixable_frontier  # non-empty

    def test_live_ledger_non_placement_flag_is_a_research_gap(self):
        ledger = [_row("vit_b", "mnist", "cascaded", "VALID_FLAGGED_unsupported_op")]
        report = coverage_report(ledger)
        assert report.research_gap_frontier  # non-empty
        assert report.placement_fixable_frontier == []
