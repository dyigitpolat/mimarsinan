"""Normalized science-row contract for the campaign ledger."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.coverage_ledger import (
    CoverageStatus,
    HypervolumeCell,
    row_to_cell,
)
from mimarsinan.chip_simulation.ledger_schema import (
    LedgerSchemaError,
    normalize_ledger_record,
)


def test_normalize_science_row_stamps_explicit_axes_and_cell_key():
    row = {
        "model": "deep_cnn",
        "dataset": "CIFAR10_DataProvider",
        "schedule": "synchronized",
        "spiking_mode": "ttfs_cycle_based",
        "deployment_validity": "VALID_on_chip_majority",
        "deployed_acc": 0.7,
        "ann_acc": 0.8,
        "activation_quantization": True,
        "weight_quantization": False,
        "prune_sparsity": 0.2,
        "preload_weights": True,
        "simulation_steps": 16,
        "depth": 6,
        "run_id": "cifar10_d6_synchronized_s0",
    }

    out = normalize_ledger_record(row)

    assert out["model"] == "deep_cnn"
    assert out["dataset"] == "cifar10"
    assert out["schedule"] == "synchronized"
    assert out["sync"] == "synchronized"
    assert out["spiking_mode"] == "ttfs_cycle_based"
    assert out["quantization"] == "aq"
    assert out["pruning"] == "pruned"
    assert out["regime"] == "pretrained"
    assert out["S"] == "16"
    assert out["depth"] == "6"
    assert out["deployment_validity_tier"] == CoverageStatus.VALID.name
    assert out["hypervolume_cell_key"] == row_to_cell(out).cell_key
    assert HypervolumeCell.from_key(out["hypervolume_cell_key"]) == row_to_cell(out)


def test_normalize_dual_schedule_row_uses_first_cell_key_and_records_syncs():
    row = {
        "model": "deep_cnn",
        "dataset": "mnist",
        "spiking_mode": "ttfs_cycle_based",
        "deployment_validity": "VALID",
        "cascaded_deployed_mean": 0.70,
        "synchronized_deployed_mean": 0.80,
    }

    out = normalize_ledger_record(row)

    assert out["syncs"] == ["cascaded", "synchronized"]
    assert out["sync"] == "cascaded"
    assert "schedule" not in out
    assert row_to_cell(out).sync == "cascaded"


def test_normalize_record_preserves_timing_and_cost_provenance():
    row = {
        "model": "deep_cnn",
        "dataset": "mnist",
        "schedule": "cascaded",
        "deployment_validity": "VALID_FLAGGED_placement",
        "deployed_acc": 0.9,
        "wall_s": 120.0,
        "tuning_wall_s": 45.0,
        "max_ft_pass_wall_s": 12.0,
        "cost_record": {"mj_per_sample": 1.2, "latency_steps": 64},
        "source": "unit",
    }

    out = normalize_ledger_record(row)

    assert out["deployment_validity_tier"] == CoverageStatus.VALID_FLAGGED.name
    assert out["timing"]["wall_s"] == 120.0
    assert out["timing"]["tuning_wall_s"] == 45.0
    assert out["timing"]["max_ft_pass_wall_s"] == 12.0
    assert out["cost_record"]["latency_steps"] == 64
    assert out["cost_provenance"]["kind"] == "measured_cost_record"
    assert out["provenance"]["source"] == "unit"
    assert out["validity"]["tier"] == CoverageStatus.VALID_FLAGGED.name
    assert out["validity"]["raw"] == "VALID_FLAGGED_placement"


def test_normalize_record_labels_proxy_and_missing_cost_provenance():
    proxy = normalize_ledger_record({
        "model": "deep_cnn",
        "dataset": "mnist",
        "schedule": "cascaded",
        "deployment_validity": "VALID",
        "deployed_acc": 0.9,
        "cost_proxy": {"latency_steps": 16},
    })
    missing = normalize_ledger_record({
        "model": "deep_cnn",
        "dataset": "mnist",
        "schedule": "cascaded",
        "deployment_validity": "VALID",
        "deployed_acc": 0.9,
    })

    assert proxy["cost_provenance"]["kind"] == "proxy"
    assert missing["cost_provenance"]["kind"] == "missing"


def test_non_science_row_can_passthrough_without_normalization():
    row = {"run_id": "done", "state": "FINALIZED_rc0"}

    assert normalize_ledger_record(row, require_science=False) == row


def test_science_row_requires_model_and_validity():
    with pytest.raises(LedgerSchemaError):
        normalize_ledger_record({"dataset": "mnist", "deployment_validity": "VALID"})

    with pytest.raises(LedgerSchemaError):
        normalize_ledger_record({"model": "deep_cnn", "dataset": "mnist"})
