"""Tests for the semantic-axis equivalence-SCREEN instrument (Wave-9 A2).

Tests-first: a synthetic ledger where two cell-populations differ ONLY in a
semantic axis value (pruning=dense vs pruned; regime=from_scratch vs pretrained):

* a SMALL |Δacc| within the band AND the same validity tier ⇒ EQUIVALENT
  (collapsible) with the MEASURED Δ recorded;
* a LARGE |Δacc| past the band ⇒ INTERACTING (stays ENUMERATED) with the
  MEASURED Δ — an upgrade from ASSERTED_UNSCREENED, not faked;
* too few paired cells (only one axis value present) ⇒ INSUFFICIENT_DATA;
* a tier MISMATCH at a small Δ ⇒ INTERACTING (equivalence needs BOTH);
* the honesty gate ``assert_semantic_screen_sound`` passes a good artifact and
  RAISES on a collapse claim that lacks a measured Δ or the min cell count.

The live-ledger helpers honestly return INSUFFICIENT_DATA while the F3 dual-regime
rows are draining and while no pruned rows exist — the instrument ships now; the
AXIS flip in coverage_ledger is the later consume step.
"""

from __future__ import annotations

import json
import os

import pytest

from mimarsinan.chip_simulation.semantic_axis_screen import (
    SemanticAxisState,
    SemanticPairOutcome,
    SemanticScreenError,
    assert_semantic_screen_sound,
    screen_live_pruning,
    screen_live_regime,
    screen_semantic_axis,
    write_semantic_screen,
)


# --- synthetic ledger rows: two populations differing ONLY in the semantic axis -

def _row(*, pruning="dense", regime="from_scratch", dataset, deployed_acc, tier,
         model="deep_cnn", schedule="synchronized", S=4, depth=8):
    """A science-valid ledger row carrying a single-schedule deployed accuracy."""
    return {
        "model": model,
        "dataset": dataset,
        "schedule": schedule,
        "spiking_mode": "lif",
        "backend": "sanafe",
        "S": S,
        "depth": depth,
        "pruning": pruning,
        "regime": regime,
        "deployment_validity": tier,
        "deployed_acc": deployed_acc,
    }


# --- EQUIVALENT: small Δ, same tier ⇒ collapsible -----------------------------

def test_equivalent_small_delta_same_tier():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.970, tier="VALID_on_chip_majority"),
        _row(pruning="pruned", dataset="mnist", deployed_acc=0.968, tier="VALID_on_chip_majority"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.state is SemanticAxisState.EQUIVALENT
    assert o.delta_pp == pytest.approx(0.2, abs=1e-9)  # |0.970-0.968|*100
    assert o.value_a == "dense" and o.value_b == "pruned"
    assert o.tier_a == o.tier_b


def test_equivalent_at_the_band_edge_is_inclusive():
    rows = [
        _row(regime="from_scratch", dataset="mnist", deployed_acc=0.970, tier="VALID"),
        _row(regime="pretrained", dataset="mnist", deployed_acc=0.960, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("regime", rows, tol_pp=1.0)
    o = outcomes[0]
    assert o.delta_pp == pytest.approx(1.0, abs=1e-9)
    assert o.state is SemanticAxisState.EQUIVALENT  # |Δ| == tol is WITHIN the band


# --- INTERACTING: large Δ ⇒ enumerated, with the measured Δ -------------------

def test_interacting_large_delta_is_enumerated_with_measured_delta():
    rows = [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    o = outcomes[0]
    assert o.state is SemanticAxisState.INTERACTING
    assert o.delta_pp == pytest.approx(18.0, abs=1e-9)
    # The Δ is MEASURED (an upgrade from ASSERTED_UNSCREENED), not asserted.
    assert o.reason is not None and "18.0" in o.reason


def test_interacting_on_tier_mismatch_even_when_delta_small():
    # Equivalence needs BOTH: same tier AND |Δ| within band. A tier flip alone interacts.
    rows = [
        _row(regime="from_scratch", dataset="svhn", deployed_acc=0.900, tier="VALID"),
        _row(regime="pretrained", dataset="svhn", deployed_acc=0.899, tier="VALID_FLAGGED_placement"),
    ]
    outcomes = screen_semantic_axis("regime", rows, tol_pp=1.0)
    o = outcomes[0]
    assert o.state is SemanticAxisState.INTERACTING
    assert o.delta_pp == pytest.approx(0.1, abs=1e-9)
    assert "tier" in o.reason.lower()


# --- INSUFFICIENT_DATA: only one axis value present in a group -----------------

def test_insufficient_data_when_no_pair():
    # Two dense rows on different datasets: each forms its OWN group (dataset is a
    # different coordinate) and NEITHER has a pruned counterpart to pair with.
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.97, tier="VALID"),
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    assert len(outcomes) == 2
    for o in outcomes:
        assert o.state is SemanticAxisState.INSUFFICIENT_DATA
        assert o.delta_pp is None
        assert "1" in o.reason  # only one distinct axis value seen in the group


def test_insufficient_data_on_empty_rows():
    outcomes = screen_semantic_axis("regime", [], tol_pp=1.0)
    assert len(outcomes) == 1
    assert outcomes[0].state is SemanticAxisState.INSUFFICIENT_DATA


# --- mixed: one paired group EQUIVALENT, one unpaired ⇒ both reported ----------

def test_mixed_groups_report_paired_and_unpaired_separately():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.970, tier="VALID"),
        _row(pruning="pruned", dataset="mnist", deployed_acc=0.969, tier="VALID"),
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),  # unpaired
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    states = {o.group_key: o.state for o in outcomes}
    assert SemanticAxisState.EQUIVALENT in states.values()
    # The unpaired cifar10 group is reported INSUFFICIENT_DATA, not silently dropped.
    assert SemanticAxisState.INSUFFICIENT_DATA in states.values()


def test_unknown_axis_is_rejected():
    with pytest.raises(SemanticScreenError, match="not a SEMANTIC"):
        screen_semantic_axis("backend", [], tol_pp=1.0)


# --- the artifact: JSON-able, deterministic (no timestamp), diffable ----------

def _equivalent_rows():
    return [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.970, tier="VALID"),
        _row(pruning="pruned", dataset="mnist", deployed_acc=0.969, tier="VALID"),
    ]


def test_artifact_is_jsonable_and_has_no_timestamp():
    outcomes = screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0)
    artifact = write_semantic_screen(
        "pruning", outcomes, tol_pp=1.0, methodology="paired by other coords",
    )
    dumped = json.dumps(artifact, sort_keys=True)
    assert json.loads(dumped) == artifact
    assert "timestamp" not in dumped.lower()
    assert artifact["axis"] == "pruning"
    assert "tol_pp" in artifact
    assert "outcomes" in artifact
    assert "methodology" in artifact


def test_artifact_is_deterministic_across_calls():
    a = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0),
        tol_pp=1.0, methodology="m",
    )
    b = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0),
        tol_pp=1.0, methodology="m",
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- the honesty gate: passes a good artifact, RAISES on a faked collapse ------

def test_soundness_passes_on_a_good_equivalent_artifact():
    outcomes = screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0)
    artifact = write_semantic_screen(
        "pruning", outcomes, tol_pp=1.0, methodology="m",
        justifies_collapse=True, min_equivalent_cells=1,
    )
    assert_semantic_screen_sound(artifact)  # no raise


def test_soundness_raises_on_equivalent_without_a_measured_delta():
    outcomes = screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0)
    artifact = write_semantic_screen("pruning", outcomes, tol_pp=1.0, methodology="m")
    for entry in artifact["outcomes"]:
        if entry["state"] == SemanticAxisState.EQUIVALENT.value:
            entry["delta_pp"] = None
            break
    with pytest.raises(SemanticScreenError, match="EQUIVALENT.*delta"):
        assert_semantic_screen_sound(artifact)


def test_soundness_raises_on_interacting_without_a_measured_delta():
    rows = [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", rows, tol_pp=1.0),
        tol_pp=1.0, methodology="m",
    )
    for entry in artifact["outcomes"]:
        if entry["state"] == SemanticAxisState.INTERACTING.value:
            entry["delta_pp"] = None
            break
    with pytest.raises(SemanticScreenError, match="INTERACTING.*delta"):
        assert_semantic_screen_sound(artifact)


def test_soundness_raises_on_malformed_state():
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0),
        tol_pp=1.0, methodology="m",
    )
    artifact["outcomes"][0]["state"] = "MAYBE"
    with pytest.raises(SemanticScreenError, match="malformed|state"):
        assert_semantic_screen_sound(artifact)


def test_soundness_raises_when_collapse_claimed_with_an_interacting_outcome():
    # A collapse cannot rest on an axis that has a MEASURED interaction.
    rows = _equivalent_rows() + [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", rows, tol_pp=1.0),
        tol_pp=1.0, methodology="m", justifies_collapse=True, min_equivalent_cells=1,
    )
    with pytest.raises(SemanticScreenError, match="collapse|INTERACTING"):
        assert_semantic_screen_sound(artifact)


def test_soundness_raises_when_collapse_claimed_below_min_cell_count():
    # One EQUIVALENT cell but min_equivalent_cells=2 ⇒ not enough evidence to collapse.
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0),
        tol_pp=1.0, methodology="m", justifies_collapse=True, min_equivalent_cells=2,
    )
    with pytest.raises(SemanticScreenError, match="min.*cell|cell.*count|enough"):
        assert_semantic_screen_sound(artifact)


def test_soundness_raises_when_collapse_claimed_over_insufficient_data_only():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.97, tier="VALID"),
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
    ]
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", rows, tol_pp=1.0),
        tol_pp=1.0, methodology="m", justifies_collapse=True, min_equivalent_cells=1,
    )
    with pytest.raises(SemanticScreenError):
        assert_semantic_screen_sound(artifact)


def test_soundness_allows_non_collapse_artifact_with_interactions():
    # Recording a MEASURED interaction is honest as long as no collapse is claimed.
    rows = [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", rows, tol_pp=1.0),
        tol_pp=1.0, methodology="m", justifies_collapse=False,
    )
    assert_semantic_screen_sound(artifact)  # no raise


# --- the outcome record round-trips to/from its dict form ---------------------

def test_outcome_to_dict_is_typed():
    o = SemanticPairOutcome(
        axis="pruning",
        group_key="vehicle=deep_cnn|dataset=mnist",
        value_a="dense",
        value_b="pruned",
        state=SemanticAxisState.EQUIVALENT,
        delta_pp=0.2,
        tol_pp=1.0,
        tier_a="VALID",
        tier_b="VALID",
        reason=None,
    )
    d = o.to_dict()
    assert d["state"] == "equivalent"
    assert d["axis"] == "pruning"
    assert d["value_a"] == "dense" and d["value_b"] == "pruned"
    assert d["delta_pp"] == 0.2


# --- the LIVE-ledger helpers: honest INSUFFICIENT_DATA while data drains -------
# The campaign ledger lives in the MAIN repo runtime tree, not the worktree git
# state — resolve it cwd-robustly and skip the live checks when it is absent
# (mirrors tests/unit/chip_simulation/test_pareto.py's pattern).
_LEDGER_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs", "campaign", "ledger.jsonl"),
    "/home/yigit/repos/research_stuff/mimarsinan/runs/campaign/ledger.jsonl",
]


def _real_ledger_path():
    for cand in _LEDGER_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


def _live_rows():
    with open(_real_ledger_path()) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_live_pruning_is_insufficient_data_no_pruned_rows():
    # No pruned rows exist in the live ledger yet ⇒ honest INSUFFICIENT_DATA.
    artifact = screen_live_pruning(_live_rows(), tol_pp=1.0)
    assert artifact["axis"] == "pruning"
    assert artifact["justifies_collapse"] is False
    states = {o["state"] for o in artifact["outcomes"]}
    assert SemanticAxisState.EQUIVALENT.value not in states
    assert_semantic_screen_sound(artifact)  # an honest, non-collapse artifact is sound


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_live_regime_drains_honestly():
    # The F3 dual-regime rows are draining: either too few pairs (INSUFFICIENT_DATA)
    # or a MEASURED outcome — never a fabricated collapse.
    artifact = screen_live_regime(_live_rows(), tol_pp=1.0)
    assert artifact["axis"] == "regime"
    assert artifact["justifies_collapse"] is False
    assert_semantic_screen_sound(artifact)


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_live_helpers_never_assert_collapse():
    # The instrument ships the MEASURED state; it must not flip the axis here.
    for artifact in (screen_live_regime(_live_rows(), tol_pp=1.0),
                     screen_live_pruning(_live_rows(), tol_pp=1.0)):
        assert artifact["justifies_collapse"] is False
