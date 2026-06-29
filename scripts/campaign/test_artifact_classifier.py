"""Tests for generic hypervolume artifact classification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import artifact_classifier as ac  # noqa: E402


def _artifact(tmp_path: Path, *, exitcode: str = "0", metric: float = 0.986, stderr: str = "") -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "exitcode").write_text(exitcode)
    (root / "stdout.log").write_text(
        "\n".join(
            [
                "[PROFILE] step='Pretraining' wall=  10.00s metric=0.9829 Δ=+0.9829 (prev=0.0000)",
                "[PROFILE] step='Hard Core Mapping' wall=  20.00s metric=0.9820 Δ=+0.0000 (prev=0.9820)",
                f"[PROFILE] step='Simulation' wall=  30.00s metric={metric:.4f} Δ=+0.0000 (prev={metric:.4f})",
            ]
        )
    )
    (root / "stderr.log").write_text(stderr)
    return root


def test_reads_target_metric_json(tmp_path):
    root = _artifact(tmp_path, metric=0.9500)
    run_dir = root / "generated" / "demo_phased_deployment_run"
    run_dir.mkdir(parents=True)
    (run_dir / "__target_metric.json").write_text("0.9860")

    record = ac.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.deployed_acc == 0.986
    assert record.status == "VALID_97_FAST"


def test_measured_dead_from_stderr(tmp_path):
    root = _artifact(tmp_path, metric=0.9900, stderr="MEASURED_DEAD: cascaded collapse\n")

    record = ac.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.status == "MEASURED_DEAD"
    assert record.owner == "research"


def test_emits_ledger_rows(tmp_path):
    root = _artifact(tmp_path)
    record = ac.classify_artifact(
        root,
        fastest_baseline_wall_s=100.0,
        axes={"spiking_mode": "ttfs", "sync": "analytical", "backend": "sanafe"},
    )
    rows = ac.classify_to_ledger_rows(
        [record],
        study="TEST",
        cluster="TEST",
        axes_by_name={
            record.name: {
                "vehicle": "mlp_mixer_core",
                "dataset": "MNIST_DataProvider",
                "spiking_mode": "ttfs",
                "sync": "analytical",
                "backend": "sanafe",
            }
        },
    )
    assert rows[0]["study"] == "TEST"
    assert rows[0]["deployed_acc"] == 0.986
