"""Frontier E6 — the certification protocol that REPLACES byte-identity as the Fix-B gate.

Locks the mechanism:

* the cell key is (firing × sync × backend) and round-trips through its canonical
  ``mode[/schedule]@backend`` string;
* the frozen-floor FORMAT serializes/deserializes losslessly (and fails loud on a
  format-version or field drift);
* :func:`freeze_cell` records the current numbers immutably;
* the GATE (:func:`certify`) PASSes iff ``deployed >= floor − eps`` AND
  ``wall_clock <= budget``, FAILs naming the regressing side, and reports
  MISSING_FLOOR (never a silent pass) for a cell with no frozen floor.
"""

import json

import pytest

from mimarsinan.chip_simulation.certification import (
    DEFAULT_ACCURACY_EPS,
    FLOOR_BOOK_FORMAT_VERSION,
    CertificationCell,
    CertificationFloorBook,
    CertificationStatus,
    RegressionFloor,
    certify,
    freeze_cell,
    load_floor_book,
    save_floor_book,
)


# --------------------------------------------------------------------------- #
# CertificationCell — the (firing × sync × backend) coordinate.
# --------------------------------------------------------------------------- #

class TestCertificationCell:
    def test_cell_key_without_sync(self):
        cell = CertificationCell(firing="lif", sync=None, backend="nevresim")
        assert cell.cell_key == "lif@nevresim"

    def test_cell_key_with_sync(self):
        cell = CertificationCell(
            firing="ttfs_cycle_based", sync="cascaded", backend="sanafe"
        )
        assert cell.cell_key == "ttfs_cycle_based/cascaded@sanafe"

    def test_cell_key_round_trips(self):
        for cell in (
            CertificationCell("lif", None, "nevresim"),
            CertificationCell("ttfs_quantized", None, "sanafe"),
            CertificationCell("ttfs_cycle_based", "synchronized", "hcm"),
        ):
            assert CertificationCell.from_key(cell.cell_key) == cell

    def test_from_key_rejects_missing_backend(self):
        with pytest.raises(ValueError, match="@backend"):
            CertificationCell.from_key("lif")

    def test_variant_defaults_none_keeps_canonical_key(self):
        # Default variant => the key is byte-identical to the pre-variant format,
        # so adding the field changes no existing floor-book key.
        assert CertificationCell("lif", None, "nevresim").variant is None
        assert CertificationCell("lif", None, "nevresim").cell_key == "lif@nevresim"

    def test_variant_disambiguates_same_recipe_cell(self):
        # Two deployment configs that share a (firing × sync) recipe cell but have
        # distinct floors (e.g. plain vs pruned LIF) get distinct keys.
        plain = CertificationCell("lif", None, "nevresim", variant="rate")
        pruned = CertificationCell("lif", None, "nevresim", variant="pruned_scheduled")
        assert plain.cell_key == "lif@nevresim#rate"
        assert pruned.cell_key == "lif@nevresim#pruned_scheduled"
        assert plain.cell_key != pruned.cell_key

    def test_variant_key_round_trips(self):
        for cell in (
            CertificationCell("lif", None, "nevresim", variant="rate"),
            CertificationCell("ttfs_cycle_based", "cascaded", "sanafe", variant="nobias"),
        ):
            assert CertificationCell.from_key(cell.cell_key) == cell

    def test_from_mode_policy_reuses_mode_naming(self):
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )

        policy = policy_for_spiking_mode("ttfs_cycle_based", "cascaded")
        cell = CertificationCell.from_mode_policy(policy, backend="nevresim")
        assert cell.cell_key == "ttfs_cycle_based/cascaded@nevresim"


# --------------------------------------------------------------------------- #
# RegressionFloor — the frozen baseline + its derived thresholds.
# --------------------------------------------------------------------------- #

class TestRegressionFloor:
    def test_accuracy_floor_subtracts_eps(self):
        floor = RegressionFloor(deployed_accuracy=0.95, wall_clock_s=70.0, eps=0.01)
        assert floor.accuracy_floor() == pytest.approx(0.94)

    def test_wall_clock_budget_from_slack(self):
        floor = RegressionFloor(
            deployed_accuracy=0.95, wall_clock_s=100.0, wall_clock_slack=0.5
        )
        assert floor.wall_clock_budget() == pytest.approx(150.0)

    def test_absolute_budget_wins_over_slack(self):
        floor = RegressionFloor(
            deployed_accuracy=0.95,
            wall_clock_s=100.0,
            wall_clock_slack=0.5,
            wall_clock_budget_s=80.0,
        )
        assert floor.wall_clock_budget() == pytest.approx(80.0)

    def test_from_dict_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="unknown fields"):
            RegressionFloor.from_dict(
                {"deployed_accuracy": 0.9, "wall_clock_s": 1.0, "bogus": 1}
            )


# --------------------------------------------------------------------------- #
# CertificationFloorBook — the frozen FORMAT (JSON round-trip + drift guard).
# --------------------------------------------------------------------------- #

class TestFloorBookFormat:
    def _book(self):
        book = CertificationFloorBook()
        book = freeze_cell(
            book,
            CertificationCell("lif", None, "nevresim"),
            deployed_accuracy=0.9784,
            wall_clock_s=60.0,
            eps=0.0,
            provenance={"commit": "deadbee"},
        )
        book = freeze_cell(
            book,
            CertificationCell("ttfs_cycle_based", "cascaded", "nevresim"),
            deployed_accuracy=0.95,
            wall_clock_s=70.0,
            eps=0.01,
            wall_clock_slack=0.5,
        )
        return book

    def test_round_trips_through_json(self, tmp_path):
        book = self._book()
        path = tmp_path / "floor.json"
        save_floor_book(book, str(path))
        reloaded = load_floor_book(str(path))
        assert reloaded.to_dict() == book.to_dict()

    def test_serialized_keys_are_canonical_cell_keys(self, tmp_path):
        book = self._book()
        path = tmp_path / "floor.json"
        save_floor_book(book, str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format_version"] == FLOOR_BOOK_FORMAT_VERSION
        assert set(data["floors"]) == {
            "lif@nevresim",
            "ttfs_cycle_based/cascaded@nevresim",
        }

    def test_provenance_is_preserved(self, tmp_path):
        book = self._book()
        path = tmp_path / "floor.json"
        save_floor_book(book, str(path))
        reloaded = load_floor_book(str(path))
        floor = reloaded.floor_for(CertificationCell("lif", None, "nevresim"))
        assert floor.provenance["commit"] == "deadbee"

    def test_format_version_drift_fails_loud(self):
        with pytest.raises(ValueError, match="format_version"):
            CertificationFloorBook.from_dict(
                {"format_version": FLOOR_BOOK_FORMAT_VERSION + 1, "floors": {}}
            )


# --------------------------------------------------------------------------- #
# freeze_cell — recording the current numbers immutably.
# --------------------------------------------------------------------------- #

class TestFreezeCell:
    def test_freeze_is_immutable_and_additive(self):
        empty = CertificationFloorBook()
        cell = CertificationCell("lif", None, "nevresim")
        frozen = freeze_cell(empty, cell, deployed_accuracy=0.97, wall_clock_s=60.0)
        assert not empty.has_floor(cell)  # original untouched
        assert frozen.has_floor(cell)
        assert frozen.floor_for(cell).deployed_accuracy == pytest.approx(0.97)

    def test_re_freeze_overwrites_only_that_cell(self):
        book = CertificationFloorBook()
        a = CertificationCell("lif", None, "nevresim")
        b = CertificationCell("ttfs_quantized", None, "sanafe")
        book = freeze_cell(book, a, deployed_accuracy=0.9, wall_clock_s=10.0)
        book = freeze_cell(book, b, deployed_accuracy=0.8, wall_clock_s=20.0)
        book = freeze_cell(book, a, deployed_accuracy=0.95, wall_clock_s=12.0)
        assert book.floor_for(a).deployed_accuracy == pytest.approx(0.95)
        assert book.floor_for(b).deployed_accuracy == pytest.approx(0.8)

    def test_default_eps_is_zero(self):
        book = freeze_cell(
            CertificationFloorBook(),
            CertificationCell("lif", None, "nevresim"),
            deployed_accuracy=0.9,
            wall_clock_s=10.0,
        )
        floor = book.floor_for(CertificationCell("lif", None, "nevresim"))
        assert floor.eps == DEFAULT_ACCURACY_EPS


# --------------------------------------------------------------------------- #
# certify — the GATE: pass / fail / missing-floor.
# --------------------------------------------------------------------------- #

class TestCertifyGate:
    def _book(self):
        return freeze_cell(
            CertificationFloorBook(),
            CertificationCell("ttfs_cycle_based", "cascaded", "nevresim"),
            deployed_accuracy=0.95,
            wall_clock_s=100.0,
            eps=0.01,
            wall_clock_slack=0.5,  # budget = 150s
        )

    def _cell(self):
        return CertificationCell("ttfs_cycle_based", "cascaded", "nevresim")

    def test_pass_within_both_tolerances(self):
        verdict = certify(
            self._cell(),
            deployed_accuracy=0.945,  # >= 0.94 floor
            wall_clock_s=120.0,  # <= 150 budget
            floor_book=self._book(),
        )
        assert verdict.status is CertificationStatus.PASS
        assert verdict.passed
        assert verdict.accuracy_ok and verdict.wall_clock_ok

    def test_pass_exactly_on_both_boundaries(self):
        verdict = certify(
            self._cell(),
            deployed_accuracy=0.94,  # == floor − eps
            wall_clock_s=150.0,  # == budget
            floor_book=self._book(),
        )
        assert verdict.status is CertificationStatus.PASS

    def test_fail_on_accuracy_regression(self):
        verdict = certify(
            self._cell(),
            deployed_accuracy=0.93,  # < 0.94 floor
            wall_clock_s=100.0,
            floor_book=self._book(),
        )
        assert verdict.status is CertificationStatus.FAIL
        assert not verdict.accuracy_ok
        assert verdict.wall_clock_ok
        assert "accuracy" in verdict.reason

    def test_fail_on_wall_clock_regression(self):
        verdict = certify(
            self._cell(),
            deployed_accuracy=0.96,  # accuracy improved
            wall_clock_s=200.0,  # but blew the budget
            floor_book=self._book(),
        )
        assert verdict.status is CertificationStatus.FAIL
        assert verdict.accuracy_ok
        assert not verdict.wall_clock_ok
        assert "wall-clock" in verdict.reason

    def test_fail_on_both(self):
        verdict = certify(
            self._cell(),
            deployed_accuracy=0.5,
            wall_clock_s=999.0,
            floor_book=self._book(),
        )
        assert verdict.status is CertificationStatus.FAIL
        assert not verdict.accuracy_ok
        assert not verdict.wall_clock_ok

    def test_missing_floor_is_not_a_pass(self):
        # An unrelated cell is frozen; the cell under test has no floor.
        book = freeze_cell(
            CertificationFloorBook(),
            CertificationCell("lif", None, "nevresim"),
            deployed_accuracy=0.97,
            wall_clock_s=60.0,
        )
        verdict = certify(
            self._cell(),  # ttfs cascaded — not in the book
            deployed_accuracy=0.99,  # would pass any floor
            wall_clock_s=1.0,
            floor_book=book,
        )
        assert verdict.status is CertificationStatus.MISSING_FLOOR
        assert not verdict.passed
        assert verdict.floor is None
        assert "no frozen floor" in verdict.reason

    def test_gate_keys_by_backend(self):
        # Same (firing × sync), different backend → a distinct cell with no floor.
        book = self._book()  # ...@nevresim
        sanafe_cell = CertificationCell("ttfs_cycle_based", "cascaded", "sanafe")
        verdict = certify(
            sanafe_cell,
            deployed_accuracy=0.99,
            wall_clock_s=1.0,
            floor_book=book,
        )
        assert verdict.status is CertificationStatus.MISSING_FLOOR


# --------------------------------------------------------------------------- #
# Freeze → gate full loop (the protocol end-to-end on the certification API).
# --------------------------------------------------------------------------- #

class TestFreezeThenGate:
    def test_a_run_at_the_frozen_numbers_certifies(self):
        cell = CertificationCell("lif", None, "nevresim")
        book = freeze_cell(
            CertificationFloorBook(), cell,
            deployed_accuracy=0.9784, wall_clock_s=60.0,
        )
        verdict = certify(
            cell, deployed_accuracy=0.9784, wall_clock_s=60.0, floor_book=book
        )
        assert verdict.passed

    def test_freeze_persist_reload_then_gate(self, tmp_path):
        cell = CertificationCell("ttfs_cycle_based", "cascaded", "nevresim")
        book = freeze_cell(
            CertificationFloorBook(), cell,
            deployed_accuracy=0.95, wall_clock_s=70.0, eps=0.02,
        )
        path = tmp_path / "floor.json"
        save_floor_book(book, str(path))
        reloaded = load_floor_book(str(path))
        # Fix B's fast recipe: a touch lower accuracy, dramatically faster.
        verdict = certify(
            cell, deployed_accuracy=0.94, wall_clock_s=12.0, floor_book=reloaded
        )
        assert verdict.passed
