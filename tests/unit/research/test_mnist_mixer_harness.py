"""CPU-only contracts for the MNIST mlp_mixer_core diagnostic harness."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.ledger_schema import (
    fastest_successful_baseline_wall_s,
    normalize_planned_ledger_row,
    with_relative_timing,
)
from mimarsinan.research.harness import (
    ACCEPTANCE_DEPLOYED_ACC,
    ACCEPTANCE_RELATIVE_TIME,
    MIXER_DIAGNOSTIC_STEP_NAMES,
    AcceptanceGate,
    build_mnist_mixer_manifest,
    default_recipe_for_cell,
    planned_mnist_mixer_ledger_row,
    recipe_is_certified_for_promotion,
    recipe_registry,
)

REQUIRED_RECIPES = {
    "mixer_lif_best_known",
    "mixer_lif_fast_stabilized",
    "mixer_sync_ttfs_relaxed_parity",
    "mixer_sync_ttfs_qat_minimal",
    "mixer_cascaded_proxy_then_refine_minimal",
    "mixer_cascaded_genuine_blend_fast",
    "mixer_ttfs_quantized_q100_fast",
    "mixer_ttfs_quantized_q100_fast_timing",
    "mixer_ttfs_analytical_control",
    "mixer_controller_baseline",
}

# Genuine-cascade QAT repair recipes (the production port of the proven CIFAR fix)
# plus the faithfulness/timing repairs added by the MNIST mixer fix wave.
REQUIRED_REPAIR_RECIPES = {
    "mixer_lif_genuine_qat",
    "mixer_lif_genuine_qat_a03",
    "mixer_lif_genuine_qat_a07",
    "mixer_sync_ttfs_faithfulness_repair",
    "mixer_cascaded_genuine_qat",
    "mixer_ttfs_quantized_q100_aggressive_timing",
}


def test_repair_recipes_carry_genuine_qat_levers():
    """The genuine-cascade QAT port must freeze BN and re-weight the KD+CE blend
    (kd_ce_alpha/kd_temperature) — the two levers that closed CIFAR conversion."""
    registry = recipe_registry()
    assert REQUIRED_REPAIR_RECIPES <= set(registry)

    for rid in ("mixer_lif_genuine_qat", "mixer_lif_genuine_qat_a03",
                "mixer_lif_genuine_qat_a07", "mixer_sync_ttfs_faithfulness_repair",
                "mixer_cascaded_genuine_qat"):
        ov = registry[rid].base_overrides
        assert ov["deployment_parameters.fast_ladder_freeze_bn"] is True
        assert ov["deployment_parameters.kd_temperature"] == 4.0
        assert "deployment_parameters.kd_ce_alpha" in ov

    # The alpha sweep brackets the CE/KD blend at 0.3 / 0.5 / 0.7.
    assert registry["mixer_lif_genuine_qat"].base_overrides[
        "deployment_parameters.kd_ce_alpha"] == 0.5
    assert registry["mixer_lif_genuine_qat_a03"].base_overrides[
        "deployment_parameters.kd_ce_alpha"] == 0.3
    assert registry["mixer_lif_genuine_qat_a07"].base_overrides[
        "deployment_parameters.kd_ce_alpha"] == 0.7

    # The sync repair relaxes the honest mixer WQ parity residual to 0.15.
    assert registry["mixer_sync_ttfs_faithfulness_repair"].base_overrides[
        "deployment_parameters.nf_scm_parity_max_mismatch_fraction"] == 0.15


# The post-v2 SOLUTION-study wave: a cascaded-collapse revival (the regression is
# conversion_policy silently forcing the controller) and a LIF gradual-adaptation
# study (close the ~1.5-2.5pp gap with non-destructive ramp/calibration, NOT a long
# stabilize). Each recipe is a measured, control-flanked candidate fix.
CASCADED_SOLUTION_RECIPES = {
    "mixer_cascaded_policy_isolate",
    "mixer_cascaded_blend_theta",
    "mixer_cascaded_staircase_ste_theta",
}
LIF_SOLUTION_RECIPES = {
    "mixer_lif_fine_ladder",
    "mixer_lif_q100_distmatch",
    "mixer_lif_controller_gradual",
}
LIF_THETA_RECIPES = {
    "mixer_lif_theta_cotrain",
    "mixer_lif_theta_distmatch",
}
ROUND2_LIF_RECIPES = {
    "mixer_lif_theta_qat",
    "mixer_lif_theta_qat_a03",
}
ROUND2_CASCADED_RECIPES = {
    "mixer_cascaded_blend_fast_finer",
    "mixer_cascaded_blend_fast_kd",
}


def test_cascaded_revival_recipes_isolate_and_repair_the_regression():
    """The cascaded collapse is a regression: conversion_policy vetoes the proven
    fast genuine-blend driver and routes the deep cascade through the controller
    (the controller_baseline failure mode). The revival keeps the fast driver and
    adds the proven per-channel theta / staircase-STE accuracy levers instead."""
    registry = recipe_registry()
    assert CASCADED_SOLUTION_RECIPES <= set(registry)

    # A5 trigger-isolate: the proven blend-fast base + conversion_policy ONLY, so a
    # collapse pins the regression on that single knob (not BN-freeze / kd reweight).
    isolate = registry["mixer_cascaded_policy_isolate"].base_overrides
    assert isolate["deployment_parameters.optimization_driver"] == "fast"
    assert isolate["deployment_parameters.ttfs_genuine_blend_fast"] is True
    assert isolate["deployment_parameters.conversion_policy"] is True

    # Revival via the per-channel trainable theta lever on the surviving blend ramp:
    # keeps the fast driver, never enables conversion_policy.
    blend_theta = registry["mixer_cascaded_blend_theta"].base_overrides
    assert blend_theta["deployment_parameters.optimization_driver"] == "fast"
    assert blend_theta["deployment_parameters.ttfs_genuine_blend_fast"] is True
    assert blend_theta["deployment_parameters.ttfs_theta_cotrain"] is True
    assert "deployment_parameters.conversion_policy" not in blend_theta

    # Revival via the proven near-lossless deep-cascade recipe (staircase-STE fast
    # loop + per-channel theta co-train, split LR + progressive depth).
    ste = registry["mixer_cascaded_staircase_ste_theta"].base_overrides
    assert ste["deployment_parameters.optimization_driver"] == "fast"
    assert ste["deployment_parameters.ttfs_staircase_ste"] is True
    assert ste["deployment_parameters.ttfs_staircase_ste_fast"] is True
    assert ste["deployment_parameters.ttfs_theta_cotrain"] is True
    assert "deployment_parameters.conversion_policy" not in ste


def test_lif_gradual_recipes_use_nondestructive_levers_not_long_stabilize():
    """The LIF gap study closes ~1.5-2.5pp with gradual non-destructive adaptation
    (finer ramp, clip-bias/distmatch, the golden controller ramp) — every recipe
    holds stabilization SHORT (<= 300), never the 600-1200 stabilize of the
    flagged baselines."""
    registry = recipe_registry()
    assert LIF_SOLUTION_RECIPES <= set(registry)

    for rid in LIF_SOLUTION_RECIPES:
        ov = registry[rid].base_overrides
        budget = registry[rid].budget_schedule.to_dict()
        assert budget["stabilization_steps"] <= 300, rid

    fine = registry["mixer_lif_fine_ladder"].base_overrides
    assert fine["deployment_parameters.lif_blend_fast"] is True
    rates = fine["deployment_parameters.lif_blend_fast_rates"]
    assert rates[-1] == 1.0 and len(rates) >= 6  # denser ladder than the 4-rung default
    assert fine["deployment_parameters.lif_blend_fast_stabilize_steps"] <= 300

    q100 = registry["mixer_lif_q100_distmatch"].base_overrides
    assert q100["deployment_parameters.activation_scale_quantile"] == 1.0
    assert q100["deployment_parameters.lif_distmatch"] is True

    grad = registry["mixer_lif_controller_gradual"].base_overrides
    assert grad["deployment_parameters.optimization_driver"] == "controller"
    assert grad["deployment_parameters.tuning_stabilization_bounded"] is True
    assert grad["deployment_parameters.tuning_keepbest_certified"] is True


def test_lif_theta_cotrain_recipes_promote_per_channel_trainable_theta():
    """Round-2: the #1 LIF lever. A single scalar firing threshold cannot serve
    both wide and narrow channels of a perceptron, so rebind activation_scale to a
    per-output-channel TRAINABLE theta the gradual blend ramp co-trains with the
    weights. Rides the SHORT-stabilize fine ladder (never a long polish), and
    composes with the orthogonal DFQ per-neuron bias match."""
    registry = recipe_registry()
    assert LIF_THETA_RECIPES <= set(registry)

    for rid in LIF_THETA_RECIPES:
        ov = registry[rid].base_overrides
        budget = registry[rid].budget_schedule.to_dict()
        assert ov["deployment_parameters.lif_theta_cotrain"] is True, rid
        assert ov["deployment_parameters.lif_blend_fast"] is True, rid
        rates = ov["deployment_parameters.lif_blend_fast_rates"]
        assert rates[-1] == 1.0 and len(rates) >= 6, rid  # gradual non-destructive
        assert ov["deployment_parameters.lif_blend_fast_stabilize_steps"] <= 300, rid
        assert budget["stabilization_steps"] <= 300, rid

    # theta isolated: no distmatch, so a movement is attributable to the scale lever.
    iso = registry["mixer_lif_theta_cotrain"].base_overrides
    assert "deployment_parameters.lif_distmatch" not in iso

    # composed: per-channel theta (scale) + DFQ bias match — the two orthogonal levers.
    comp = registry["mixer_lif_theta_distmatch"].base_overrides
    assert comp["deployment_parameters.lif_distmatch"] is True
    assert comp["deployment_parameters.activation_scale_quantile"] == 1.0


def test_round2_lif_compose_theta_with_genuine_qat():
    """Round-2 LIF gap closer: compose the two MEASURED-faithful levers — per-channel
    theta (decode scale, survives deployment) + genuine-cascade KD+CE QAT — on the
    SHORT-stabilize fine ladder. Round-1 levers each cluster 0.952-0.968; the
    hypothesis is that the faithful scale lever and the conversion-faithful training
    objective compound over 0.97. Never a long polish; alpha sweeps the KD weight."""
    registry = recipe_registry()
    assert ROUND2_LIF_RECIPES <= set(registry)

    for rid in ROUND2_LIF_RECIPES:
        ov = registry[rid].base_overrides
        budget = registry[rid].budget_schedule.to_dict()
        assert ov["deployment_parameters.lif_theta_cotrain"] is True, rid
        assert ov["deployment_parameters.lif_blend_fast"] is True, rid
        assert ov["deployment_parameters.fast_ladder_freeze_bn"] is True, rid
        assert ov["deployment_parameters.cycle_accurate_lif_forward"] is True, rid
        # the genuine-QAT KD+CE objective composed onto the faithful theta lever
        assert 0.0 < ov["deployment_parameters.kd_ce_alpha"] < 1.0, rid
        assert ov["deployment_parameters.kd_temperature"] > 1.0, rid
        rates = ov["deployment_parameters.lif_blend_fast_rates"]
        assert rates[-1] == 1.0 and len(rates) >= 6, rid  # gradual non-destructive
        assert ov["deployment_parameters.lif_blend_fast_stabilize_steps"] <= 300, rid
        assert budget["stabilization_steps"] <= 300, rid

    alphas = {
        registry[rid].base_overrides["deployment_parameters.kd_ce_alpha"]
        for rid in ROUND2_LIF_RECIPES
    }
    assert len(alphas) >= 2, "the two compose recipes must sweep the KD weight"


def test_round2_cascaded_lifts_fast_ladder_without_the_controller():
    """Round-2 cascaded revival: the ONLY surviving faithful cascaded path is the fast
    genuine-blend ladder (genuine_blend_fast: 0.94 @ parity 0.9961); genuine_qat and
    policy_isolate both collapse through the controller. Lift the fast ladder by
    ramping FINER through the rate→1.0 collapse zone with longer per-rung recovery —
    never routing through conversion_policy / the controller. A KD-anchored variant
    holds the teacher tighter across the collapse zone."""
    registry = recipe_registry()
    assert ROUND2_CASCADED_RECIPES <= set(registry)

    default_rates = [0.5, 0.75, 0.9, 0.97, 1.0]
    for rid in ROUND2_CASCADED_RECIPES:
        ov = registry[rid].base_overrides
        assert ov["deployment_parameters.optimization_driver"] == "fast", rid
        assert ov["deployment_parameters.ttfs_genuine_blend_fast"] is True, rid
        assert ov["deployment_parameters.ttfs_genuine_blend_ramp"] is True, rid
        # the controller-collapse trigger must never be set on a revival path
        assert "deployment_parameters.conversion_policy" not in ov, rid
        rates = ov["deployment_parameters.ttfs_blend_fast_rates"]
        assert rates[-1] == 1.0, rid
        assert len(rates) > len(default_rates), rid  # finer than the 5-rung default
        # finer specifically in the collapse zone (more rungs in [0.9, 1.0))
        near = [r for r in rates if 0.9 <= r < 1.0]
        assert len(near) >= 2, rid
        assert ov["deployment_parameters.ttfs_blend_fast_steps_per_rate"] >= 160, rid

    # KD-anchored variant holds the teacher tighter (lower CE weight = more KD) than
    # the finer-only variant's default.
    kd = registry["mixer_cascaded_blend_fast_kd"].base_overrides
    assert kd["deployment_parameters.ttfs_genuine_blend_ce_alpha"] < 0.3


def test_solution_study_recipes_are_wired_into_their_cells():
    manifest = build_mnist_mixer_manifest(seeds=(0, 1))
    cascaded = manifest.cell_by_id("mnist_mmixcore_ttfs_cycle_cascaded")
    lif = manifest.cell_by_id("mnist_mmixcore_lif")
    assert CASCADED_SOLUTION_RECIPES <= set(cascaded.recipe_ids)
    assert LIF_SOLUTION_RECIPES <= set(lif.recipe_ids)
    assert LIF_THETA_RECIPES <= set(lif.recipe_ids)
    assert ROUND2_LIF_RECIPES <= set(lif.recipe_ids)
    assert ROUND2_CASCADED_RECIPES <= set(cascaded.recipe_ids)


def test_recipe_registry_contains_required_presets_and_hard_gates():
    registry = recipe_registry()

    assert REQUIRED_RECIPES <= set(registry)
    for recipe in registry.values():
        assert recipe.acceptance.min_deployed_acc == pytest.approx(ACCEPTANCE_DEPLOYED_ACC)
        assert recipe.acceptance.max_relative_time == pytest.approx(ACCEPTANCE_RELATIVE_TIME)
        assert recipe.base_overrides["deployment_parameters.recipe_id"] == recipe.recipe_id
        assert recipe.budget_schedule.to_dict()["ramp_steps"] >= 0
        assert recipe.budget_schedule.to_dict()["recovery_steps"] >= 0
        assert recipe.budget_schedule.to_dict()["stabilization_steps"] >= 0
        assert recipe.budget_schedule.to_dict()["eval_sample_count"] > 0

    assert registry["mixer_lif_best_known"].base_overrides[
        "deployment_parameters.optimization_driver"
    ] == "controller"
    assert registry["mixer_ttfs_quantized_q100_fast"].base_overrides[
        "deployment_parameters.activation_scale_quantile"
    ] == 1.0
    assert registry["mixer_sync_ttfs_relaxed_parity"].base_overrides[
        "deployment_parameters.nf_scm_parity_max_mismatch_fraction"
    ] == 0.15
    assert registry["mixer_controller_baseline"].base_overrides[
        "deployment_parameters.optimization_driver"
    ] == "controller"
    assert registry["mixer_cascaded_genuine_blend_fast"].base_overrides[
        "deployment_parameters.ttfs_genuine_blend_fast"
    ] is True


def test_acceptance_gate_is_hard_on_accuracy_timing_and_returncode():
    gate = AcceptanceGate()

    assert gate.evaluate(
        {"returncode": 0, "deployed_acc": 0.9701, "relative_time": 0.999}
    ).accepted
    assert not gate.evaluate(
        {"returncode": 0, "deployed_acc": 0.9699, "relative_time": 0.5}
    ).accepted
    assert not gate.evaluate(
        {"returncode": 0, "deployed_acc": 0.98, "relative_time": 1.0}
    ).accepted
    assert not gate.evaluate(
        {"returncode": 1, "deployed_acc": 0.99, "relative_time": 0.5}
    ).accepted


def test_fastest_baseline_and_relative_timing_ignore_failed_rows():
    rows = [
        {"returncode": 0, "deployed_acc": 0.971, "wall_s": 20.0},
        {"returncode": 0, "deployed_acc": 0.974, "result": {"wall_s": 12.0}},
        {"returncode": 1, "deployed_acc": 0.99, "wall_s": 3.0},
        {"returncode": 0, "deployed_acc": None, "wall_s": 4.0},
    ]

    baseline = fastest_successful_baseline_wall_s(rows)
    assert baseline == pytest.approx(12.0)

    annotated = with_relative_timing({"run_wall_s": 9.0}, baseline)
    assert annotated["relative_time"] == pytest.approx(0.75)
    assert annotated["faster_than_baseline"] is True


def test_manifest_has_required_cells_and_planned_ledger_rows():
    manifest = build_mnist_mixer_manifest(seeds=(0, 1))

    roles = {cell.role for cell in manifest.cells}
    assert {"diagnostic", "analytical_control"} <= roles
    assert {cell.cell_id for cell in manifest.cells} >= {
        "mnist_mmixcore_lif",
        "mnist_mmixcore_ttfs_cycle_synchronized",
        "mnist_mmixcore_ttfs_cycle_cascaded",
        "mnist_mmixcore_ttfs_analytical_control",
        "mnist_mmixcore_ttfs_quantized_control",
    }
    for cell in manifest.cells:
        assert cell.acceptance.min_deployed_acc == pytest.approx(0.97)
        assert cell.acceptance.max_relative_time == pytest.approx(1.0)
        assert cell.recipe_ids

    row = planned_mnist_mixer_ledger_row(
        run_id="mnist_mmixcore_lif_s0",
        cell=manifest.cell_by_id("mnist_mmixcore_lif"),
        recipe=recipe_registry()["mixer_lif_best_known"],
        seed=0,
    )
    normalized = normalize_planned_ledger_row(row)

    assert normalized["row_type"] == "planned"
    assert normalized["acceptance"]["min_deployed_acc"] == pytest.approx(0.97)
    assert normalized["acceptance"]["max_relative_time"] == pytest.approx(1.0)
    assert normalized["relative_time"] is None
    assert normalized["axes"]["vehicle"] == "mlp_mixer_core"
    assert normalized["axes"]["dataset"] == "MNIST_DataProvider"
    assert [step["name"] for step in normalized["step_metrics"]] == list(
        MIXER_DIAGNOSTIC_STEP_NAMES
    )
    assert "proxy_genuine" in normalized["probes"]


def test_recipe_promotion_requires_certified_rows():
    manifest = build_mnist_mixer_manifest(seeds=(0, 1))
    cell = manifest.cell_by_id("mnist_mmixcore_ttfs_analytical_control")
    rows = [
        {
            "recipe_id": "mixer_ttfs_analytical_control",
            "seed": 0,
            "returncode": 0,
            "deployed_acc": 0.986,
            "relative_time": 0.8,
        },
        {
            "recipe_id": "mixer_ttfs_analytical_control",
            "seed": 1,
            "returncode": 0,
            "deployed_acc": 0.985,
            "relative_time": 0.9,
        },
    ]
    recipe = recipe_registry()["mixer_ttfs_analytical_control"]
    assert recipe_is_certified_for_promotion(recipe, rows, required_seeds=(0, 1))
    assert default_recipe_for_cell(cell, rows, required_seeds=(0, 1)) == "mixer_ttfs_analytical_control"
