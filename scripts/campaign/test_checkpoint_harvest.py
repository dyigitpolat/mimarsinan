"""Checkpoint JSONL → normalized campaign ledger rows."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import checkpoint_harvest as ch  # noqa: E402

from mimarsinan.chip_simulation.coverage_ledger import (  # noqa: E402
    CoverageStatus,
    classify_validity_tier,
    row_to_cell,
)


def test_baseline_record_maps_direct_cell_fields_to_normalized_row():
    rec = {
        "cell": "cifar10_d4_synchronized",
        "dataset": "cifar10",
        "depth": 4,
        "mode": "synchronized",
        "ann": 0.7956,
        "deployed": 0.7004,
        "retention_pp": -9.52,
        "returncode": 0,
        "wall_s": 463.0,
        "last_step": "Core Quantization Verification",
        "config": "cfg.json",
        "experiment": "exp",
    }

    row = ch.checkpoint_record_to_ledger(rec, source_file="baseline.jsonl")

    assert row["model"] == "deep_cnn"
    assert row["dataset"] == "cifar10"
    assert row["schedule"] == "synchronized"
    assert row["spiking_mode"] == "ttfs_cycle_based"
    assert row["S"] == "4"
    assert row["depth"] == "4"
    assert row["deployed_acc"] == pytest.approx(0.7004)
    assert row["ann_acc"] == pytest.approx(0.7956)
    assert row["verdict"] == "MET"
    assert row["deployment_validity_tier"] == CoverageStatus.VALID.name
    assert row["provenance"]["source_file"] == "baseline.jsonl"
    assert row_to_cell(row).dataset == "cifar10"


def test_returncode_one_predeploy_metric_is_flagged_not_clean_valid():
    rec = {
        "label": "ttfs_budget40_ep60",
        "ann": 0.8375,
        "deployed": 0.7107,
        "retention_pp": -12.68,
        "returncode": 1,
        "wall_s": 333.7,
        "gradual_s": 7.38,
        "last_step": "Activation Analysis",
    }

    row = ch.checkpoint_record_to_ledger(rec, source_file="budget.jsonl")

    assert row["dataset"] == "cifar10"
    assert row["depth"] == "4"
    assert row["schedule"] == "synchronized"
    assert row["spiking_mode"] == "ttfs_cycle_based"
    assert row["deployment_validity_tier"] == CoverageStatus.VALID_FLAGGED.name
    assert classify_validity_tier(row["deployment_validity"]) is CoverageStatus.VALID_FLAGGED
    assert row["metric_context"] == "gate_rejected_predeploy_spiking_accuracy"
    assert row["verdict"] == "FAIL"
    assert row["timing"]["gradual_s"] == pytest.approx(7.38)


def test_lif_label_maps_to_lif_firing_without_sync_schedule():
    rec = {
        "label": "lif_a0.3_base",
        "ann": 0.7969,
        "deployed": 0.1409,
        "retention_pp": -65.6,
        "returncode": 1,
        "wall_s": 1361.1,
        "last_step": "Hard Core Mapping",
    }

    row = ch.checkpoint_record_to_ledger(rec, source_file="alpha.jsonl")

    assert row["spiking_mode"] == "lif"
    assert row["sync"] == "none"
    assert "schedule" not in row
    assert row_to_cell(row).firing == "lif"


def test_t_label_sets_temporal_budget_axis():
    rec = {
        "label": "ttfs_T32",
        "ann": 0.7897,
        "deployed": 0.6885,
        "retention_pp": -10.12,
        "returncode": 0,
        "wall_s": 446.4,
        "last_step": "Core Quantization Verification",
    }

    row = ch.checkpoint_record_to_ledger(rec, source_file="T.jsonl")

    assert row["S"] == "32"
    assert row["schedule"] == "synchronized"


def test_harvest_directory_reads_jsonl_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "one.jsonl").write_text(
        '{"label":"ttfs_T8","ann":0.8,"deployed":0.7,'
        '"retention_pp":-10.0,"returncode":0,"wall_s":1.0}\n'
    )

    rows = ch.harvest_checkpoint_dir(str(data))

    assert len(rows) == 1
    assert rows[0]["provenance"]["source_file"].endswith("one.jsonl")
