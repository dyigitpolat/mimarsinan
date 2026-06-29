"""Typed research harness contracts for prototype promotion."""

from __future__ import annotations

import pytest

from mimarsinan.research.harness import (
    BudgetSchedule,
    ExperimentContext,
    FixRecipe,
    MechanismResult,
    VehicleSpec,
    normalize_retention,
    promotion_record,
    recipe_preset,
    recipe_presets,
)


def test_vehicle_spec_builds_pipeline_config_overlay():
    vehicle = VehicleSpec(
        name="deep_cnn",
        model_type="deep_cnn",
        dataset="cifar10",
        depth=4,
        input_shape=(3, 32, 32),
        num_classes=10,
        residual=False,
    )

    overlay = vehicle.config_overlay()

    assert overlay["model_type"] == "deep_cnn"
    assert overlay["data_provider_name"] == "CIFAR10_DataProvider"
    assert overlay["model_config"]["depth"] == 4
    assert overlay["model_config"]["input_shape"] == [3, 32, 32]
    assert overlay["model_config"]["num_classes"] == 10


def test_fix_recipe_composes_default_off_config_flags():
    qat = FixRecipe(
        name="sync_qat_fast_bn",
        mechanism="synchronized_qat",
        config_flags={
            "ttfs_sync_genuine_qat": True,
            "ttfs_blend_fast": True,
            "fast_ladder_freeze_bn": True,
        },
        owner="conversion",
    )

    assert qat.config_overlay()["ttfs_sync_genuine_qat"] is True
    assert qat.owner == "conversion"


def test_recipe_presets_are_named_production_overlays():
    presets = recipe_presets()

    assert "sync_qat_fast_bn" in presets
    assert "lif_qat_fast_bn" in presets
    assert recipe_preset("sync_qat_fast_bn").config_overlay()["ttfs_sync_genuine_qat"] is True
    assert recipe_preset("lif_qat_fast_bn").config_overlay()["cycle_accurate_lif_forward"] is True

    with pytest.raises(KeyError):
        recipe_preset("unknown_recipe")


def test_budget_schedule_encodes_timing_recipe():
    budget = BudgetSchedule(
        name="fast_20m",
        max_tuning_wall_s=20 * 60,
        max_ft_pass_wall_s=300,
        max_adaptation_steps=4000,
        stabilization_steps=0,
    )

    overlay = budget.config_overlay()

    assert overlay["tuning_budget_max_wall_s"] == 1200
    assert overlay["max_ft_pass_wall_s"] == 300
    assert overlay["tuning_budget_scale_ramp_steps"] is False


def test_experiment_context_merges_vehicle_recipe_and_budget():
    ctx = ExperimentContext(
        vehicle=VehicleSpec("deep_cnn", "deep_cnn", "cifar10", depth=4),
        recipe=FixRecipe("baseline", "none", {"ttfs_sync_genuine_qat": False}),
        budget=BudgetSchedule("controller", max_tuning_wall_s=None),
        seed=3,
        simulation_steps=16,
        platform={"cores": 1024},
    )

    cfg = ctx.config_overlay()

    assert cfg["model_type"] == "deep_cnn"
    assert cfg["seed"] == 3
    assert cfg["simulation_steps"] == 16
    assert cfg["ttfs_sync_genuine_qat"] is False
    assert cfg["platform_constraints"]["cores"] == 1024


def test_mechanism_result_and_promotion_record_capture_gates():
    result = MechanismResult(
        ann_acc=0.80,
        deployed_acc=0.72,
        wall_s=600.0,
        tuning_wall_s=200.0,
        parity_mismatch=0.0,
        validity_tier="VALID",
    )

    assert normalize_retention(result) == pytest.approx(0.9)

    record = promotion_record(
        context=ExperimentContext(
            vehicle=VehicleSpec("deep_cnn", "deep_cnn", "cifar10"),
            recipe=FixRecipe("sync_qat", "synchronized_qat", {}),
            budget=BudgetSchedule("fast", max_tuning_wall_s=1000.0),
            seed=0,
            simulation_steps=4,
        ),
        result=result,
        source="unit",
    )

    assert record["vehicle"] == "deep_cnn"
    assert record["recipe"] == "sync_qat"
    assert record["retention"] == pytest.approx(0.9)
    assert record["passes_parity_gate"] is True
    assert record["passes_timing_gate"] is True
