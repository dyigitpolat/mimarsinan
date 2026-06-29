"""Tests for the MNIST mixer diagnostic campaign generator."""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import coverage_breadth as cb


def test_mnist_mixer_batches_validate_and_carry_acceptance_rows() -> None:
    batches = cb.gen_mnist_mixer_diagnostic_batches(seeds=(0, 1), enabled=True)

    assert batches
    assert all(batch["enabled"] is True for batch in batches)
    assert any(batch["tags"]["cell_id"] == "mnist_mmixcore_lif" for batch in batches)
    for batch in batches:
        assert batch["acceptance"]["min_deployed_acc"] == 0.97
        assert batch["acceptance"]["max_relative_time"] == 1.0
        assert len(batch["planned_ledger_rows"]) == 2
        row = batch["planned_ledger_rows"][0]
        assert row["row_type"] == "planned"
        assert row["axes"]["vehicle"] == "mlp_mixer_core"
        assert row["relative_time"] is None


def test_emit_mnist_mixer_queue_manifest_writes_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cb, "_REPO", str(tmp_path))
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    for template in {
        "mnist_mmixcore_matrix_1_lif_rate.json",
        "mnist_mmixcore_matrix_7_ttfs_cycle_synchronized.json",
        "mnist_mmixcore_matrix_6_ttfs_cycle_cascaded.json",
        "mnist_mmixcore_matrix_4_ttfs_analytical.json",
        "mnist_mmixcore_matrix_5_ttfs_quantized_offload.json",
    }:
        (template_dir / template).write_text(
            json.dumps({
                "data_provider_name": "MNIST_DataProvider",
                "experiment_name": "base",
                "generated_files_path": "generated",
                "pipeline_mode": "phased",
                "deployment_parameters": {},
                "platform_constraints": {},
                "start_step": None,
            }),
            encoding="utf-8",
        )

    out = tmp_path / "queue.json"
    count = cb.emit_mnist_mixer_queue_manifest(
        str(out),
        seeds=(0,),
        config_dir=str(tmp_path / "configs"),
    )

    jobs = json.loads(out.read_text(encoding="utf-8"))
    assert count == len(jobs)
    assert jobs
    assert all(job["cmd"][:2] == ["env/bin/python", "run.py"] for job in jobs)
    assert all((tmp_path / job["cmd"][-1]).exists() for job in jobs)
