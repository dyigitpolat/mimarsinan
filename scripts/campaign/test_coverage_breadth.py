"""Coverage-breadth campaign manifest."""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import coverage_breadth as cb  # noqa: E402


def test_manifest_names_required_breadth_frontiers():
    manifest = cb.coverage_breadth_manifest()
    ids = {item["id"] for item in manifest}

    assert "residual_deep_cnn_cifar10_sync_d4" in ids
    assert "svhn_deep_cnn_sync_d4" in ids
    assert "semantic_screen_pruning" in ids
    assert "semantic_screen_regime" in ids
    assert "imagenet_resnet50_adapted_lif" in ids
    assert "cifar100_deep_cnn_sync_d4" in ids
    assert any(item["kind"] == "mixer_diagnostic" for item in manifest)
    assert any(item["kind"] == "workload_diagnostic" for item in manifest)


def test_run_items_carry_config_overlays_and_success_gates():
    item = next(
        i for i in cb.coverage_breadth_manifest()
        if i["id"] == "svhn_deep_cnn_sync_d4"
    )

    assert item["kind"] == "run"
    assert item["config_overlay"]["data_provider_name"] == "SVHN_DataProvider"
    assert item["config_overlay"]["model_config"]["depth"] == 4
    assert item["success_gates"]["ledger_row_required"] is True
    assert item["success_gates"]["nf_scm_parity_required"] is True
    assert item["axis_coordinates"]["dataset"] == "svhn"
    assert item["axis_coordinates"]["sync"] == "synchronized"
    assert item["axis_coordinates"]["S"] == "4"
    assert item["axis_coordinates"]["depth"] == "4"


def test_screen_items_name_axis_and_artifact_path():
    item = next(
        i for i in cb.coverage_breadth_manifest()
        if i["id"] == "semantic_screen_pruning"
    )

    assert item["kind"] == "semantic_screen"
    assert item["axis"] == "pruning"
    assert item["artifact"].endswith("pruning_semantic_screen.json")


def test_planned_run_items_normalize_to_complete_axis_rows():
    required = {
        "model",
        "dataset",
        "spiking_mode",
        "sync",
        "backend",
        "regime",
        "quantization",
        "pruning",
        "mapping_strategy",
        "S",
        "depth",
        "hypervolume_cell_key",
    }

    for item in cb.coverage_breadth_manifest():
        if item["kind"] != "run":
            continue
        row = cb.planned_ledger_row(item)
        assert required <= set(row)
        assert all(row[key] != "any" for key in required - {"hypervolume_cell_key"})
        assert row["deployment_validity_tier"] == "VALID_FLAGGED"
        assert row["provenance"]["source"] == "coverage_breadth_manifest"


def test_queue_manifest_writes_research_loop_run_jobs(tmp_path):
    jobs = cb.coverage_breadth_queue_manifest(config_dir=str(tmp_path / "cfg"))

    assert len(jobs) == 5
    svhn_job = next(j for j in jobs if j["id"] == "svhn_deep_cnn_sync_d4")
    assert svhn_job["cmd"][:3] == ["env/bin/python", "run.py", "--headless"]
    assert svhn_job["cwd"].endswith("mimarsinan")
    assert svhn_job["expect_artifact"].endswith(
        "generated/svhn_deep_cnn_sync_d4_phased_deployment_run/__target_metric.json"
    )
    assert svhn_job["tags"]["batch_id"] == "coverage_breadth"
    assert svhn_job["tags"]["dataset"] == "svhn"
    assert svhn_job["tags"]["sync"] == "synchronized"
    assert svhn_job["tags"]["cost_or_proxy_required"] is True

    config_path = tmp_path / "cfg" / "svhn_deep_cnn_sync_d4.json"
    cfg = json.loads(config_path.read_text())
    assert cfg["experiment_name"] == "svhn_deep_cnn_sync_d4"
    assert cfg["data_provider_name"] == "SVHN_DataProvider"
    assert cfg["deployment_parameters"]["model_type"] == "deep_cnn"
    assert cfg["deployment_parameters"]["model_config"]["depth"] == 4
    assert cfg["deployment_parameters"]["ttfs_sync_genuine_qat"] is True
    assert cfg["deployment_parameters"]["fast_ladder_freeze_bn"] is True
    assert cfg["platform_constraints"]["simulation_steps"] == 4
