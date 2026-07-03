"""Tests for the cross-simulator parity SCREENING instrument."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.cross_sim_parity import (
    CrossSimOutcome,
    CrossSimParityError,
    CrossSimState,
    assert_cross_sim_screen_sound,
    derive_applicability,
    measured_max_abs_diff,
    screen_cell_pair,
    write_cross_sim_screen,
)


def _records(values):
    """One-perceptron normalized record: {perceptron_index: (samples, neurons)}."""
    return {0: np.asarray(values, dtype=np.float64)}


def test_measured_max_abs_diff_is_zero_for_identical_records():
    nf = _records([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    scm = _records([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert measured_max_abs_diff(nf, scm) == 0.0


def test_measured_max_abs_diff_is_order_insensitive():
    nf = _records([[0.3, 0.1, 0.2]])
    scm = _records([[0.1, 0.2, 0.3]])
    assert measured_max_abs_diff(nf, scm) == 0.0


def test_measured_max_abs_diff_quantifies_the_gap():
    nf = _records([[0.1, 0.2, 0.3]])
    scm = _records([[0.1, 0.2, 0.55]])
    assert measured_max_abs_diff(nf, scm) == pytest.approx(0.25)


def test_validated_cell_agrees_with_zero_diff():
    nf = _records([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    scm = _records([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    outcome = screen_cell_pair(
        cell="lif/identity@T8",
        backend_a="nevresim",
        backend_b="hcm",
        spiking_mode="lif",
        nf_record=nf,
        scm_record=scm,
        tolerance=1e-9,
    )
    assert outcome.state is CrossSimState.AGREE
    assert outcome.max_abs_diff == 0.0
    assert outcome.reason is None


def test_within_tolerance_residual_is_agree():
    nf = _records([[0.10, 0.20]])
    scm = _records([[0.10, 0.20 + 1e-12]])
    outcome = screen_cell_pair(
        cell="lif/identity@T8",
        backend_a="nevresim",
        backend_b="hcm",
        spiking_mode="lif",
        nf_record=nf,
        scm_record=scm,
        tolerance=1e-9,
    )
    assert outcome.state is CrossSimState.AGREE
    assert 0.0 < outcome.max_abs_diff <= 1e-9


def test_lava_ttfs_is_inapplicable_from_capabilities():
    outcome = screen_cell_pair(
        cell="ttfs_cycle_based/cascaded/identity@T8",
        backend_a="nevresim",
        backend_b="lava",
        spiking_mode="ttfs_cycle_based",
        nf_record=None,
        scm_record=None,
        tolerance=1e-9,
    )
    assert outcome.state is CrossSimState.INAPPLICABLE
    assert outcome.max_abs_diff is None
    assert "lava" in outcome.reason
    assert "ttfs_cycle_based" in outcome.reason


def test_loihi_ttfs_quantized_is_inapplicable():
    outcome = screen_cell_pair(
        cell="ttfs_quantized/identity@T8",
        backend_a="hcm",
        backend_b="loihi",
        spiking_mode="ttfs_quantized",
        nf_record=None,
        scm_record=None,
        tolerance=1e-9,
    )
    assert outcome.state is CrossSimState.INAPPLICABLE
    assert "loihi" in outcome.reason


def test_lava_lif_is_applicable():
    applicable, reason = derive_applicability("lava", "lif")
    assert applicable is True
    assert reason is None


def test_derive_applicability_names_the_unsupported_backend_and_mode():
    applicable, reason = derive_applicability("lava", "ttfs")
    assert applicable is False
    assert "lava" in reason and "ttfs" in reason


def test_perturbed_pair_disagrees_with_quantified_gap():
    nf = _records([[0.10, 0.20, 0.30]])
    scm = _records([[0.10, 0.20, 0.42]])
    outcome = screen_cell_pair(
        cell="lif/identity@T8",
        backend_a="nevresim",
        backend_b="hcm",
        spiking_mode="lif",
        nf_record=nf,
        scm_record=scm,
        tolerance=1e-9,
        disagree_reason="injected perturbation for the screen test",
    )
    assert outcome.state is CrossSimState.DISAGREE
    assert outcome.max_abs_diff == pytest.approx(0.12)
    assert outcome.reason == "injected perturbation for the screen test"


def test_disagree_without_a_reason_still_quantifies_but_flags_unreasoned():
    nf = _records([[0.10]])
    scm = _records([[0.90]])
    outcome = screen_cell_pair(
        cell="lif/identity@T8",
        backend_a="nevresim",
        backend_b="hcm",
        spiking_mode="lif",
        nf_record=nf,
        scm_record=scm,
        tolerance=1e-9,
    )
    assert outcome.state is CrossSimState.DISAGREE
    assert outcome.max_abs_diff == pytest.approx(0.80)
    assert outcome.reason is None


def _good_outcomes():
    return [
        screen_cell_pair(
            cell="lif/identity@T8",
            backend_a="nevresim", backend_b="hcm", spiking_mode="lif",
            nf_record=_records([[0.1, 0.2]]), scm_record=_records([[0.1, 0.2]]),
            tolerance=1e-9,
        ),
        screen_cell_pair(
            cell="ttfs_cycle_based/cascaded/identity@T8",
            backend_a="nevresim", backend_b="lava", spiking_mode="ttfs_cycle_based",
            nf_record=None, scm_record=None, tolerance=1e-9,
        ),
    ]


def test_artifact_is_jsonable_and_has_no_timestamp():
    import json

    artifact = write_cross_sim_screen(
        _good_outcomes(),
        tolerance=1e-9,
        methodology="nevresim≡HCM wraps nf_scm_parity; SANA-FE/Lava capability-derived",
    )
    dumped = json.dumps(artifact, sort_keys=True)
    assert json.loads(dumped) == artifact
    assert "timestamp" not in dumped.lower()
    assert "tolerance" in artifact
    assert "backend_pairs" in artifact
    assert "outcomes" in artifact
    assert "methodology" in artifact


def test_artifact_is_deterministic_across_calls():
    a = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    b = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    import json

    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_artifact_records_the_measured_diff_for_every_agree():
    artifact = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    for entry in artifact["outcomes"]:
        if entry["state"] == CrossSimState.AGREE.value:
            assert entry["max_abs_diff"] is not None


def test_soundness_passes_on_a_good_artifact():
    artifact = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    assert_cross_sim_screen_sound(artifact)


def test_soundness_raises_on_agree_without_a_max_diff():
    artifact = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    for entry in artifact["outcomes"]:
        if entry["state"] == CrossSimState.AGREE.value:
            entry["max_abs_diff"] = None
            break
    with pytest.raises(CrossSimParityError, match="AGREE.*max_abs_diff"):
        assert_cross_sim_screen_sound(artifact)


def test_soundness_raises_on_a_malformed_state():
    artifact = write_cross_sim_screen(_good_outcomes(), tolerance=1e-9, methodology="m")
    artifact["outcomes"][0]["state"] = "MAYBE"
    with pytest.raises(CrossSimParityError, match="malformed|state"):
        assert_cross_sim_screen_sound(artifact)


def test_soundness_raises_when_collapse_claimed_over_unreasoned_disagree():
    outcomes = [
        screen_cell_pair(
            cell="lif/identity@T8",
            backend_a="nevresim", backend_b="hcm", spiking_mode="lif",
            nf_record=_records([[0.1]]), scm_record=_records([[0.9]]),
            tolerance=1e-9,
        ),
    ]
    artifact = write_cross_sim_screen(
        outcomes, tolerance=1e-9, methodology="m", justifies_collapse=True,
    )
    with pytest.raises(CrossSimParityError, match="collapse|DISAGREE"):
        assert_cross_sim_screen_sound(artifact)


def test_soundness_allows_reasoned_disagree_when_not_claiming_collapse():
    outcomes = [
        screen_cell_pair(
            cell="lif/identity@T8",
            backend_a="nevresim", backend_b="hcm", spiking_mode="lif",
            nf_record=_records([[0.1]]), scm_record=_records([[0.9]]),
            tolerance=1e-9, disagree_reason="known WQ tie-flip residual",
        ),
    ]
    artifact = write_cross_sim_screen(
        outcomes, tolerance=1e-9, methodology="m", justifies_collapse=False,
    )
    assert_cross_sim_screen_sound(artifact)


def test_collapse_supported_only_when_all_applicable_pairs_agree():
    artifact = write_cross_sim_screen(
        _good_outcomes(), tolerance=1e-9, methodology="m", justifies_collapse=True,
    )
    assert_cross_sim_screen_sound(artifact)
    assert artifact["justifies_collapse"] is True


def test_outcome_to_dict_is_minimal_and_typed():
    outcome = CrossSimOutcome(
        cell="lif/identity@T8",
        backend_pair=("nevresim", "hcm"),
        state=CrossSimState.AGREE,
        max_abs_diff=0.0,
        tolerance=1e-9,
        reason=None,
    )
    d = outcome.to_dict()
    assert d["state"] == "agree"
    assert d["backend_pair"] == ["nevresim", "hcm"]
    assert d["max_abs_diff"] == 0.0
    assert d["cell"] == "lif/identity@T8"
