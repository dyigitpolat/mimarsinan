"""Config/plan/ledger axis encoding for the coverage hypervolume."""

from __future__ import annotations

from mimarsinan.chip_simulation.hypervolume_axis_encoder import (
    AxisCoordinates,
    cell_coordinates_from_row,
    quantization_axis,
    pruning_axis,
    regime_axis,
    syncs_from_row,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def test_quantization_axis_from_boolean_pair():
    assert quantization_axis(weight_quantization=False, activation_quantization=False) == "none"
    assert quantization_axis(weight_quantization=True, activation_quantization=False) == "wq"
    assert quantization_axis(weight_quantization=False, activation_quantization=True) == "aq"
    assert quantization_axis(weight_quantization=True, activation_quantization=True) == "wq_aq"


def test_pruning_axis_prefers_structured_deployment_sparsity():
    assert pruning_axis(prune_sparsity=0.0, pruning=False, pruning_fraction=0.0) == "dense"
    assert pruning_axis(prune_sparsity=0.25, pruning=False, pruning_fraction=0.0) == "pruned"
    assert pruning_axis(prune_sparsity=0.0, pruning=True, pruning_fraction=0.5) == "pruned"


def test_regime_axis_from_weight_source_and_preload_flag():
    assert regime_axis(weight_source=None, preload_weights=False) == "from_scratch"
    assert regime_axis(weight_source=None, preload_weights=True) == "pretrained"
    assert regime_axis(weight_source="torchvision", preload_weights=False) == "pretrained"


def test_row_coordinates_derives_missing_axes_from_structured_fields():
    row = {
        "model": "deep_cnn",
        "dataset": "FashionMNIST",
        "schedule": "synchronized",
        "spiking_mode": "ttfs_cycle_based",
        "deployment_validity": "VALID_on_chip_majority",
        "activation_quantization": True,
        "weight_quantization": True,
        "prune_sparsity": 0.1,
        "preload_weights": True,
        "simulation_steps": 8,
        "depth": 6,
    }

    coords = cell_coordinates_from_row(row, sync="synchronized")

    assert coords == AxisCoordinates(
        firing="ttfs_cycle_based",
        sync="synchronized",
        backend="sanafe",
        vehicle="deep_cnn",
        dataset="fmnist",
        regime="pretrained",
        quantization="wq_aq",
        pruning="pruned",
        mapping_strategy="packed",
        s="8",
        depth="6",
    )


def test_explicit_row_axis_values_win_over_derived_values():
    row = {
        "model": "deep_cnn",
        "dataset": "mnist",
        "schedule": "cascaded",
        "deployment_validity": "VALID_on_chip_majority",
        "activation_quantization": True,
        "weight_quantization": True,
        "quantization": "none",
        "prune_sparsity": 0.2,
        "pruning": "dense",
        "weight_source": "torchvision",
        "regime": "from_scratch",
        "backend": "hcm",
        "mapping_strategy": "identity",
        "S": 16,
    }

    coords = cell_coordinates_from_row(row, sync="cascaded")

    assert coords.quantization == "none"
    assert coords.pruning == "dense"
    assert coords.regime == "from_scratch"
    assert coords.backend == "hcm"
    assert coords.mapping_strategy == "identity"
    assert coords.s == "16"


def test_dual_schedule_row_expands_to_both_syncs():
    row = {
        "model": "deep_cnn",
        "dataset": "mnist",
        "deployment_validity": "VALID_on_chip_majority",
        "cascaded_deployed_mean": 0.70,
        "synchronized_deployed_mean": 0.80,
    }

    assert syncs_from_row(row) == ["cascaded", "synchronized"]


def test_plan_coordinates_use_deployment_plan_axes():
    cfg = {
        "model_type": "deep_cnn",
        "data_provider_name": "CIFAR10_DataProvider",
        "spiking_mode": "ttfs_cycle_based",
        "ttfs_cycle_schedule": "synchronized",
        "simulation_steps": 32,
        "activation_quantization": True,
        "weight_quantization": False,
        "prune_sparsity": 0.2,
        "preload_weights": True,
        "model_config": {"depth": 8},
    }
    plan = DeploymentPlan.resolve(cfg)

    coords = AxisCoordinates.from_plan(plan, cfg)

    assert coords.firing == "ttfs_cycle_based"
    assert coords.sync == "synchronized"
    assert coords.dataset == "cifar10"
    assert coords.vehicle == "deep_cnn"
    assert coords.regime == "pretrained"
    assert coords.quantization == "aq"
    assert coords.pruning == "pruned"
    assert coords.s == "32"
    assert coords.depth == "8"
