"""Wave-2 A2/A3 — the artifact-backed FAITHFULNESS collapses of backend + mapping_strategy.

Locks the keystone of this unit: two FAITHFULNESS axes collapse on a measured
parity/fidelity artifact, the cell-key mechanics fold them correctly, the SEMANTIC
knobs (pruning/regime) DO NOT collapse, placement-fixable flags auto-own, and the
deep_cnn denominator re-prices to the measured value. An adversarial verifier WILL
try to refute these — every claim is a TEST, never weakened.

The distinction the tests encode:

* ``backend`` / ``mapping_strategy`` are FAITHFULNESS axes — different simulators /
  packings of the SAME deployment contract. They collapse on a measured
  PARITY/FIDELITY artifact (faithful sims & equivalent packings agree on the DEPLOYED
  VALUE; a disagreement is a BUG, not an interaction). The collapse is FIDELITY-ONLY:
  capability and cost/utilization are NOT collapsed.
* ``pruning`` / ``regime`` are SEMANTIC knobs (they change the trained result) — they
  CANNOT collapse on a fidelity artifact; they stay ``ASSERTED_UNSCREENED``.
"""

import json
import os

import numpy as np
import pytest

from mimarsinan.chip_simulation.coverage_ledger import (
    AXES,
    PLACEMENT_FIXABLE_DEFAULT_OWNER,
    PLACEMENT_FIXABLE_FIX_PATH,
    HypervolumeAxis,
    HypervolumeCell,
    ScreeningStatus,
    collapsed_axis_representatives,
    coverage_report,
    honest_claimed_subproduct,
    row_to_cell,
)
from mimarsinan.chip_simulation.coverage_ci import (
    assert_axes_screening_sound,
    assert_no_aged_unowned_flags,
)
from mimarsinan.chip_simulation.cross_sim_parity import (
    assert_cross_sim_screen_sound,
)


# --------------------------------------------------------------------------- #
# (A2) backend — a FAITHFULNESS axis that collapses on the cross-sim screen.
# (A3) mapping_strategy — a FAITHFULNESS axis that collapses on the torch↔sim lock.
# --------------------------------------------------------------------------- #

class TestFaithfulnessCollapse:
    @pytest.mark.parametrize("name", ["backend", "mapping_strategy"])
    def test_axis_is_screened_collapsed_with_a_representative_and_artifact(self, name):
        axis = HypervolumeAxis.get(name)
        assert axis.screening_status is ScreeningStatus.SCREENED_COLLAPSED
        assert axis.collapsed is True
        assert axis.representative in axis.values
        assert axis.screening_artifact.strip()

    def test_backend_artifact_points_at_the_cross_sim_screen(self):
        artifact = HypervolumeAxis.get("backend").screening_artifact
        assert "backend_cross_sim_screen.md" in artifact
        # The multi-backend parity locks are cited as evidence.
        assert "test_scm_hcm_sim_parity.py" in artifact
        assert "nf_scm_parity" in artifact.lower()
        # lava INAPPLICABLE for TTFS (LIF-only capability gap).
        assert "lava" in artifact.lower() and "lif-only" in artifact.lower()

    def test_mapping_strategy_artifact_points_at_the_torch_sim_fidelity_lock(self):
        artifact = HypervolumeAxis.get("mapping_strategy").screening_artifact
        assert "mapping_strategy_fidelity_screen.md" in artifact
        assert "test_torch_sim_fidelity.py" in artifact
        # The coalescing GAP-1 caveat is recorded (value-domain only attribution).
        assert "coalescing" in artifact.lower()
        assert "value_domain_only" in artifact.lower() or "value-domain" in artifact.lower()

    @pytest.mark.parametrize("name", ["backend", "mapping_strategy"])
    def test_collapse_is_scoped_fidelity_only_not_cost(self, name):
        # The load-bearing scope: FIDELITY-ONLY — capability + cost/utilization NOT
        # collapsed (the encoding_placement precedent).
        scope = HypervolumeAxis.get(name).screening_artifact.lower()
        assert "fidelity-only" in scope
        assert "cost" in scope

    def test_axes_screening_sound_passes_with_the_new_collapses(self):
        # Every SCREENED_COLLAPSED axis carries an artifact (defense-in-depth CI gate).
        assert_axes_screening_sound(AXES)  # does not raise


# --------------------------------------------------------------------------- #
# The SEMANTIC knobs must NOT collapse — they change the trained result.
# --------------------------------------------------------------------------- #

class TestSemanticKnobsDoNotCollapse:
    @pytest.mark.parametrize("name", ["pruning", "regime"])
    def test_semantic_knob_stays_asserted_unscreened(self, name):
        axis = HypervolumeAxis.get(name)
        assert axis.screening_status is ScreeningStatus.ASSERTED_UNSCREENED
        assert axis.collapsed is False


# --------------------------------------------------------------------------- #
# CELL-KEY MECHANICS — the load-bearing collapse plumbing (tests-first).
# --------------------------------------------------------------------------- #

class TestCellKeyMechanics:
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
            depth="8",
        )
        base.update(over)
        return HypervolumeCell(**base)

    def test_representatives_map_is_the_ssot(self):
        reps = collapsed_axis_representatives(AXES)
        assert reps["backend"] == HypervolumeAxis.get("backend").representative
        assert reps["mapping_strategy"] == HypervolumeAxis.get("mapping_strategy").representative
        # encoding_placement is collapsed too (the precedent).
        assert reps["encoding_placement"] == "subsume"
        # The semantic knobs are NOT in the collapsed-rep map.
        assert "pruning" not in reps and "regime" not in reps

    def test_two_cells_differing_only_in_backend_share_one_cell_key(self):
        # backend is in the cert prefix — collapsing it must fold the cert key so two
        # rows differing ONLY in backend map to the SAME cell.
        a = self._cell(backend="nevresim")
        b = self._cell(backend="sanafe")
        c = self._cell(backend="hcm")
        assert a.cell_key == b.cell_key == c.cell_key
        # The folded cert prefix uses the representative.
        rep = HypervolumeAxis.get("backend").representative
        assert f"@{rep}" in a.cell_key
        assert a.cert_cell.backend == rep

    def test_two_cells_differing_only_in_mapping_strategy_share_one_cell_key(self):
        # mapping_strategy is an extending axis — collapse_orthogonal_axes drops it from
        # _CELL_AXES, so it must NOT appear in the key and rows differing only in it
        # collapse to one cell.
        a = self._cell(mapping_strategy="identity")
        b = self._cell(mapping_strategy="packed")
        c = self._cell(mapping_strategy="coalesced")
        assert a.cell_key == b.cell_key == c.cell_key
        assert "mapping_strategy=" not in a.cell_key

    def test_from_key_round_trips_with_collapsed_mapping_strategy(self):
        # from_key reads no mapping_strategy segment (dropped) — it must default to the
        # representative so the round-trip reconstructs the canonical cell.
        cell = self._cell(mapping_strategy="packed")  # = representative
        back = HypervolumeCell.from_key(cell.cell_key)
        assert back == cell
        assert back.mapping_strategy == HypervolumeAxis.get("mapping_strategy").representative

    def test_from_key_round_trips_with_collapsed_backend(self):
        cell = self._cell(backend="sanafe")  # = representative
        assert HypervolumeCell.from_key(cell.cell_key) == cell

    def test_noncollapsed_axes_still_separate_cells(self):
        # The collapse must NOT over-merge: a different dataset / depth / quantization /
        # pruning / regime is still a DISTINCT cell.
        base = self._cell()
        for axis, value in (
            ("dataset", "fmnist"),
            ("depth", "4"),
            ("quantization", "none"),
            ("pruning", "pruned"),
            ("regime", "pretrained"),
        ):
            other = self._cell(**{axis: value})
            assert base.cell_key != other.cell_key, axis

    def test_live_shaped_rows_differing_only_in_backend_or_strategy_collapse(self):
        # Two ledger-shaped rows differing ONLY in backend (or mapping_strategy) map to
        # the same covered cell — the collapse holds end-to-end through row_to_cell.
        def row(**extra):
            r = {
                "model": "deep_cnn", "dataset": "mnist", "schedule": "cascaded",
                "spiking_mode": "ttfs_cycle_based",
                "deployment_validity": "VALID_on_chip_majority", "S": 4,
            }
            r.update(extra)
            return r

        assert row_to_cell(row(backend="nevresim")).cell_key == \
            row_to_cell(row(backend="sanafe")).cell_key
        assert row_to_cell(row(mapping_strategy="identity")).cell_key == \
            row_to_cell(row(mapping_strategy="coalesced")).cell_key


# --------------------------------------------------------------------------- #
# RE-PRICING — the measured denominator with the two new collapses.
# --------------------------------------------------------------------------- #

class TestReprice:
    def test_backend_and_mapping_strategy_no_longer_enlarge_the_denominator(self):
        # Each collapsed faithfulness axis, pinned to ALL its values, leaves the honest
        # denominator UNCHANGED (it contributes one representative).
        base = honest_claimed_subproduct(
            firing=["ttfs_cycle_based"], vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist", "kmnist", "svhn", "cifar10"],
            sync=["cascaded", "synchronized"],
        )
        with_backends = honest_claimed_subproduct(
            firing=["ttfs_cycle_based"], vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist", "kmnist", "svhn", "cifar10"],
            sync=["cascaded", "synchronized"],
            backend=["nevresim", "sanafe", "hcm", "lava"],
            mapping_strategy=["packed", "identity", "neuron_split", "coalesced"],
        )
        assert len(base) == len(with_backends)

    def test_deep_cnn_firing_pinned_denominator_is_160(self):
        # The MEASURED re-priced denominator: firing pinned ttfs_cycle_based, 5 datasets
        # × 2 syncs × quantization(4) × pruning(2) × regime(2) = 160 (backend &
        # mapping_strategy collapsed). Old denominator (×4×4 for the two faithfulness
        # axes) was 2560; the collapse shrinks it to 160.
        claim = honest_claimed_subproduct(
            firing=["ttfs_cycle_based"], vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist", "kmnist", "svhn", "cifar10"],
            sync=["cascaded", "synchronized"],
        )
        assert len(claim) == 160
        # The shrink factor is exactly backend(4) × mapping_strategy(4) = 16.
        assert 2560 == 160 * 16

    def test_repriced_fraction_rises_without_changing_covered_cells(self):
        # A deep_cnn ledger covering 6 cells: the collapse shrinks the denominator (160)
        # so the honest fraction RISES to 6/160, while the covered-cell tally is
        # unchanged (the collapse re-prices the CLAIM, not the coverage).
        def cnn(dataset):
            r = {
                "model": "deep_cnn", "dataset": dataset, "schedule": None,
                "spiking_mode": "ttfs_cycle_based", "backend": "sanafe",
                "deployment_validity": "VALID_on_chip_majority", "S": 4,
            }
            r["cascaded_deployed_mean"] = 0.98
            r["synchronized_deployed_mean"] = 0.99
            return r
        ledger = [cnn("mnist"), cnn("fmnist"), cnn("kmnist")]  # 6 covered cells
        claim = honest_claimed_subproduct(
            firing=["ttfs_cycle_based"], vehicle=["deep_cnn"],
            dataset=["mnist", "fmnist", "kmnist", "svhn", "cifar10"],
            sync=["cascaded", "synchronized"],
        )
        report = coverage_report(ledger, claimed_subproduct=claim)
        assert report.covered_claimed_count == 6
        assert report.claimed_subproduct_size == 160
        assert report.coverage_fraction == pytest.approx(6 / 160)


# --------------------------------------------------------------------------- #
# FLAG OWNERSHIP — placement-fixable flags auto-own (not drift, a KNOWN fix).
# --------------------------------------------------------------------------- #

def _placement_row(**extra):
    r = {
        "model": "deep_mlp", "dataset": "mnist", "schedule": "cascaded",
        "spiking_mode": "lif", "backend": "sanafe",
        "deployment_validity": "VALID_FLAGGED_placement",
    }
    r.update(extra)
    return r


class TestPlacementFlagOwnership:
    def test_placement_fixable_flag_gets_the_default_owner_and_fix_path(self):
        # A placement-fixable flag with NO explicit owner now reports the standing
        # placement-offload owner + the encoding-offload fix-path (NOT UNOWNED).
        report = coverage_report([_placement_row(ts="2026-01-01")], now_ts="2026-06-01")
        meta = report.flag_metadata
        assert meta
        entry = meta[0]
        assert entry.owner == PLACEMENT_FIXABLE_DEFAULT_OWNER
        assert entry.fix_path == PLACEMENT_FIXABLE_FIX_PATH
        assert entry.is_unowned is False

    def test_structured_placement_op_flag_also_auto_owns(self):
        row = _placement_row(deployment_validity="VALID_FLAGGED_misc",
                              placement_fixable_ops=["Linear"], ts="2026-01-01")
        report = coverage_report([row], now_ts="2026-06-01")
        entry = report.flag_metadata[0]
        assert entry.owner == PLACEMENT_FIXABLE_DEFAULT_OWNER
        assert entry.fix_path == PLACEMENT_FIXABLE_FIX_PATH

    def test_explicit_owner_wins_over_the_default(self):
        row = _placement_row(flag_owner="dyigit", ts="2026-01-01")
        report = coverage_report([row], now_ts="2026-06-01")
        entry = report.flag_metadata[0]
        assert entry.owner == "dyigit"  # explicit owner is not overwritten
        assert entry.fix_path == PLACEMENT_FIXABLE_FIX_PATH  # still placement-fixable

    def test_research_gap_flag_does_not_auto_own(self):
        # A real research-gap flag (unsupported host op) gets NO default owner — it is a
        # genuine open research target, not a known-fix placement flag.
        row = {
            "model": "vit_b", "dataset": "mnist", "schedule": "cascaded",
            "spiking_mode": "ttfs_cycle_based", "backend": "sanafe",
            "deployment_validity": "VALID_FLAGGED_unsupported_op", "ts": "2026-01-01",
        }
        report = coverage_report([row], now_ts="2026-06-01")
        entry = report.flag_metadata[0]
        assert entry.owner is None
        assert entry.fix_path is None
        assert entry.is_unowned is True

    def test_a_flag_owning_both_a_gap_and_placement_is_not_auto_owned(self):
        # A row that owes a real research gap AND a placement fix is NOT auto-resolvable
        # — it keeps no default owner (the research gap is the blocker).
        row = _placement_row(
            deployment_validity="VALID_FLAGGED_mixed",
            placement_fixable_ops=["Linear"], research_gap_ops=["MultiheadAttention"],
            ts="2026-01-01",
        )
        report = coverage_report([row], now_ts="2026-06-01")
        entry = report.flag_metadata[0]
        assert entry.owner is None
        assert entry.fix_path is None

    def test_aged_placement_flags_stay_green_under_the_aging_guard(self):
        # An aged placement-fixable flag is auto-owned, so the no-aged-unowned guard
        # stays GREEN (the flag is not rotting — it has a standing owner + fix-path).
        report = coverage_report([_placement_row(ts="2026-01-01")], now_ts="2026-06-01")
        assert report.flag_metadata[0].age_days > 90
        assert_no_aged_unowned_flags(report, max_age_days=90)  # does not raise

    def test_fix_path_survives_serialization(self):
        report = coverage_report([_placement_row(ts="2026-01-01")], now_ts="2026-06-01")
        d = report.to_dict()
        meta = d["flag_metadata"][0]
        assert meta["fix_path"] == PLACEMENT_FIXABLE_FIX_PATH
        assert meta["owner"] == PLACEMENT_FIXABLE_DEFAULT_OWNER


# --------------------------------------------------------------------------- #
# The COMMITTED artifacts — they exist, are JSON-sound, and justify the collapse.
# --------------------------------------------------------------------------- #

_FINDINGS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docs", "research", "findings"
)


class TestCommittedArtifacts:
    def test_backend_cross_sim_screen_json_is_sound_and_justifies_collapse(self):
        path = os.path.join(_FINDINGS, "backend_cross_sim_screen.json")
        assert os.path.isfile(path), f"missing committed artifact {path}"
        with open(path, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        # The honesty gate the coverage screen calls before trusting the artifact.
        assert_cross_sim_screen_sound(artifact)
        assert artifact["justifies_collapse"] is True
        # Every applicable pair AGREEs with a recorded measured diff; lava/ttfs is
        # INAPPLICABLE (LIF-only capability gap).
        states = {o["state"] for o in artifact["outcomes"]}
        assert "agree" in states
        assert "inapplicable" in states
        agree_diffs = [
            o["max_abs_diff"] for o in artifact["outcomes"] if o["state"] == "agree"
        ]
        assert agree_diffs and all(d is not None for d in agree_diffs)
        # The deployed-value parity is bit-exact in the screened corner.
        assert max(agree_diffs) == 0.0

    def test_backend_screen_spans_both_modes_and_both_packings(self):
        path = os.path.join(_FINDINGS, "backend_cross_sim_screen.json")
        with open(path, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        cells = " ".join(o["cell"] for o in artifact["outcomes"])
        assert "lif" in cells and "ttfs_cycle_based" in cells
        assert "identity" in cells and "neuron_split" in cells

    @pytest.mark.parametrize(
        "fname",
        ["backend_cross_sim_screen.md", "mapping_strategy_fidelity_screen.md"],
    )
    def test_markdown_artifact_exists_and_scopes_fidelity_only(self, fname):
        path = os.path.join(_FINDINGS, fname)
        assert os.path.isfile(path), f"missing committed artifact {path}"
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().lower()
        assert "fidelity-only" in text
        # Capability + cost explicitly NOT collapsed.
        assert "cost" in text and "capability" in text
