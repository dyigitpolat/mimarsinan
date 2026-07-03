"""The coverage_ledger façade must keep exposing the complete pre-split public API.

coverage_ledger.py is a compatibility façade over hypervolume_axes, hypervolume_cells,
coverage_rows, and coverage_reporting; every importer's name must keep resolving.
"""

import datetime

import mimarsinan.chip_simulation.coverage_ledger as coverage_ledger

# Every name imported from coverage_ledger anywhere in src/, tests/, or scripts/
# (audited via `grep -rn "coverage_ledger import"` before the split).
NAMES_IMPORTED_ANYWHERE = (
    "AXES",
    "AttributionFidelity",
    "AxisKind",
    "CoverageReport",
    "CoverageStatus",
    "FlagMetadata",
    "HypervolumeAxis",
    "HypervolumeCell",
    "KNOWN_CRACKED_REGIONS",
    "PLACEMENT_FIXABLE_DEFAULT_OWNER",
    "PLACEMENT_FIXABLE_FIX_PATH",
    "ScreeningStatus",
    "claimed_subproduct",
    "classify_validity_tier",
    "collapse_orthogonal_axes",
    "collapsed_axis_representatives",
    "coverage_report",
    "honest_claimed_subproduct",
    "interacting_axes",
    "row_to_cell",
    "row_to_cells",
)

# Pre-split public names no importer names today — still part of the API contract.
UNIMPORTED_PUBLIC_NAMES = ("AXIS_WILDCARD", "active_axes", "cell_covers")

FULL_PUBLIC_API = tuple(sorted(NAMES_IMPORTED_ANYWHERE + UNIMPORTED_PUBLIC_NAMES))


class TestFacadeSurface:
    def test_all_is_exactly_the_pre_split_public_api(self):
        assert sorted(coverage_ledger.__all__) == list(FULL_PUBLIC_API)

    def test_every_public_name_is_exposed(self):
        missing = [n for n in FULL_PUBLIC_API if not hasattr(coverage_ledger, n)]
        assert missing == []

    def test_star_import_exposes_exactly_the_public_api(self):
        namespace = {}
        exec(
            "from mimarsinan.chip_simulation.coverage_ledger import *", namespace
        )
        exported = {n for n in namespace if not n.startswith("__")}
        assert exported == set(FULL_PUBLIC_API)

    def test_every_importer_name_resolves_via_from_import(self):
        namespace = {}
        exec(
            "from mimarsinan.chip_simulation.coverage_ledger import "
            + ", ".join(NAMES_IMPORTED_ANYWHERE),
            namespace,
        )
        assert set(NAMES_IMPORTED_ANYWHERE) <= set(namespace)


class TestFacadeDelegation:
    def _assert_reexports(self, impl, names):
        for name in names:
            assert getattr(coverage_ledger, name) is getattr(impl, name), name

    def test_axis_model_names_are_the_hypervolume_axes_objects(self):
        from mimarsinan.chip_simulation import hypervolume_axes as impl

        self._assert_reexports(
            impl,
            (
                "AXES",
                "AXIS_WILDCARD",
                "AxisKind",
                "HypervolumeAxis",
                "ScreeningStatus",
                "active_axes",
                "collapse_orthogonal_axes",
                "collapsed_axis_representatives",
                "interacting_axes",
            ),
        )

    def test_cell_names_are_the_hypervolume_cells_objects(self):
        from mimarsinan.chip_simulation import hypervolume_cells as impl

        self._assert_reexports(
            impl,
            (
                "HypervolumeCell",
                "cell_covers",
                "claimed_subproduct",
                "honest_claimed_subproduct",
            ),
        )

    def test_row_names_are_the_coverage_rows_objects(self):
        from mimarsinan.chip_simulation import coverage_rows as impl

        self._assert_reexports(
            impl,
            (
                "CoverageStatus",
                "PLACEMENT_FIXABLE_DEFAULT_OWNER",
                "PLACEMENT_FIXABLE_FIX_PATH",
                "classify_validity_tier",
                "row_to_cell",
                "row_to_cells",
            ),
        )

    def test_reporting_names_are_the_coverage_reporting_objects(self):
        from mimarsinan.chip_simulation import coverage_reporting as impl

        self._assert_reexports(
            impl,
            (
                "AttributionFidelity",
                "CoverageReport",
                "FlagMetadata",
                "KNOWN_CRACKED_REGIONS",
                "coverage_report",
            ),
        )


class TestCoverageRowsModuleApi:
    """The helpers promoted to coverage_rows module API when they crossed the module boundary."""

    def test_tier_severity_orders_invalid_worst(self):
        from mimarsinan.chip_simulation.coverage_rows import TIER_SEVERITY

        s = coverage_ledger.CoverageStatus
        assert TIER_SEVERITY[s.INVALID] > TIER_SEVERITY[s.VALID_FLAGGED] > TIER_SEVERITY[s.VALID]

    def test_parse_ledger_timestamp_reads_iso_and_epoch(self):
        from mimarsinan.chip_simulation.coverage_rows import parse_ledger_timestamp

        assert parse_ledger_timestamp("2026-06-01") == datetime.date(2026, 6, 1)
        epoch = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
        assert parse_ledger_timestamp(epoch) == datetime.date(2026, 6, 1)
        assert parse_ledger_timestamp(None) is None
        assert parse_ledger_timestamp("not-a-date") is None

    def test_mine_flagged_ops_prefers_structured_fields(self):
        from mimarsinan.chip_simulation.coverage_rows import mine_flagged_ops

        gaps, placement = mine_flagged_ops(
            {"research_gap_ops": ["MultiheadAttention"], "placement_fixable_ops": []}
        )
        assert gaps == ["MultiheadAttention"]
        assert placement == []

    def test_mine_flagged_ops_derives_placement_from_tier_suffix(self):
        from mimarsinan.chip_simulation.coverage_rows import (
            is_placement_fixable_flag,
            mine_flagged_ops,
        )

        row = {"deployment_validity": "VALID_FLAGGED_placement"}
        gaps, placement = mine_flagged_ops(row)
        assert gaps == []
        assert placement == ["encoding_layer(placement)"]
        assert is_placement_fixable_flag(row)

    def test_flag_owner_and_ts_fall_back_across_field_aliases(self):
        from mimarsinan.chip_simulation.coverage_rows import flag_owner_of, flag_ts_of

        assert flag_owner_of({"flag_owner": "team:x"}) == "team:x"
        assert flag_owner_of({"owner": "  "}) is None
        assert flag_ts_of({"ts": "2026-06-01"}) == "2026-06-01"
        assert flag_ts_of({}) is None


class TestFacadeEndToEnd:
    def _valid_row(self):
        return {
            "model": "deep_cnn",
            "dataset": "mnist",
            "schedule": "cascaded",
            "spiking_mode": "ttfs_cycle_based",
            "backend": "sanafe",
            "deployment_validity": "VALID_on_chip_majority",
        }

    def test_row_flows_through_the_facade_to_a_full_coverage_report(self):
        claimed = coverage_ledger.claimed_subproduct()
        report = coverage_ledger.coverage_report(
            [self._valid_row()], claimed_subproduct=claimed
        )
        assert report.coverage_fraction == 1.0
        assert report.tier_counts[coverage_ledger.CoverageStatus.VALID] == 1
        assert report.untested_frontier == []

    def test_row_to_cell_round_trips_through_the_cell_key(self):
        cell = coverage_ledger.row_to_cell(self._valid_row())
        assert cell is not None
        assert coverage_ledger.HypervolumeCell.from_key(cell.cell_key) == cell
