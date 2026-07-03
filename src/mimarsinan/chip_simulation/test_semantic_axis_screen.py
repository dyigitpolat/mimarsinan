"""Tests for the semantic-axis equivalence-SCREEN instrument."""

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


def test_equivalent_small_delta_same_tier():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.970, tier="VALID_on_chip_majority"),
        _row(pruning="pruned", dataset="mnist", deployed_acc=0.968, tier="VALID_on_chip_majority"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.state is SemanticAxisState.EQUIVALENT
    assert o.delta_pp == pytest.approx(0.2, abs=1e-9)
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
    assert o.state is SemanticAxisState.EQUIVALENT


def test_interacting_large_delta_is_enumerated_with_measured_delta():
    rows = [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    o = outcomes[0]
    assert o.state is SemanticAxisState.INTERACTING
    assert o.delta_pp == pytest.approx(18.0, abs=1e-9)
    assert o.reason is not None and "18.0" in o.reason


def test_interacting_on_tier_mismatch_even_when_delta_small():
    rows = [
        _row(regime="from_scratch", dataset="svhn", deployed_acc=0.900, tier="VALID"),
        _row(regime="pretrained", dataset="svhn", deployed_acc=0.899, tier="VALID_FLAGGED_placement"),
    ]
    outcomes = screen_semantic_axis("regime", rows, tol_pp=1.0)
    o = outcomes[0]
    assert o.state is SemanticAxisState.INTERACTING
    assert o.delta_pp == pytest.approx(0.1, abs=1e-9)
    assert "tier" in o.reason.lower()


def test_insufficient_data_when_no_pair():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.97, tier="VALID"),
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    assert len(outcomes) == 2
    for o in outcomes:
        assert o.state is SemanticAxisState.INSUFFICIENT_DATA
        assert o.delta_pp is None
        assert "1" in o.reason


def test_insufficient_data_on_empty_rows():
    outcomes = screen_semantic_axis("regime", [], tol_pp=1.0)
    assert len(outcomes) == 1
    assert outcomes[0].state is SemanticAxisState.INSUFFICIENT_DATA


def test_mixed_groups_report_paired_and_unpaired_separately():
    rows = [
        _row(pruning="dense", dataset="mnist", deployed_acc=0.970, tier="VALID"),
        _row(pruning="pruned", dataset="mnist", deployed_acc=0.969, tier="VALID"),
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
    ]
    outcomes = screen_semantic_axis("pruning", rows, tol_pp=1.0)
    states = {o.group_key: o.state for o in outcomes}
    assert SemanticAxisState.EQUIVALENT in states.values()
    assert SemanticAxisState.INSUFFICIENT_DATA in states.values()


def test_unknown_axis_is_rejected():
    with pytest.raises(SemanticScreenError, match="not a SEMANTIC"):
        screen_semantic_axis("backend", [], tol_pp=1.0)


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


def test_soundness_passes_on_a_good_equivalent_artifact():
    outcomes = screen_semantic_axis("pruning", _equivalent_rows(), tol_pp=1.0)
    artifact = write_semantic_screen(
        "pruning", outcomes, tol_pp=1.0, methodology="m",
        justifies_collapse=True, min_equivalent_cells=1,
    )
    assert_semantic_screen_sound(artifact)


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
    rows = [
        _row(pruning="dense", dataset="cifar10", deployed_acc=0.80, tier="VALID"),
        _row(pruning="pruned", dataset="cifar10", deployed_acc=0.62, tier="VALID"),
    ]
    artifact = write_semantic_screen(
        "pruning", screen_semantic_axis("pruning", rows, tol_pp=1.0),
        tol_pp=1.0, methodology="m", justifies_collapse=False,
    )
    assert_semantic_screen_sound(artifact)


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
    artifact = screen_live_pruning(_live_rows(), tol_pp=1.0)
    assert artifact["axis"] == "pruning"
    assert artifact["justifies_collapse"] is False
    states = {o["state"] for o in artifact["outcomes"]}
    assert SemanticAxisState.EQUIVALENT.value not in states
    assert_semantic_screen_sound(artifact)


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_live_regime_drains_honestly():
    artifact = screen_live_regime(_live_rows(), tol_pp=1.0)
    assert artifact["axis"] == "regime"
    assert artifact["justifies_collapse"] is False
    assert_semantic_screen_sound(artifact)


@pytest.mark.skipif(_real_ledger_path() is None, reason="campaign ledger not present in worktree runtime tree")
def test_live_helpers_never_assert_collapse():
    for artifact in (screen_live_regime(_live_rows(), tol_pp=1.0),
                     screen_live_pruning(_live_rows(), tol_pp=1.0)):
        assert artifact["justifies_collapse"] is False
