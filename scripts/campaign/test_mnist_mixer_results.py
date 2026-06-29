"""MNIST mixer Slurmech artifact classification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mnist_mixer_results as mmr  # noqa: E402


def _artifact(tmp_path: Path, *, exitcode: str = "0", metric: float = 0.986) -> Path:
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
    (root / "stderr.log").write_text("")
    return root


def test_classifies_valid_97_fast(tmp_path):
    root = _artifact(tmp_path, metric=0.986)

    record = mmr.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.status == "VALID_97_FAST"
    assert record.deployed_acc == 0.986
    assert record.relative_time == 0.6


def test_classifies_below_accuracy_as_flagged(tmp_path):
    root = _artifact(tmp_path, metric=0.9668)

    record = mmr.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.status == "VALID_FLAGGED_WITH_OWNER"
    assert record.failure_reason == "accuracy_below_97"


def test_classifies_nonzero_exit_as_flagged(tmp_path):
    root = _artifact(tmp_path, exitcode="1", metric=0.986)
    (root / "stderr.log").write_text("RuntimeError: Compilation failed\n")

    record = mmr.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.status == "VALID_FLAGGED_WITH_OWNER"
    assert record.failure_reason == "returncode_nonzero"


def test_classifies_measured_dead(tmp_path):
    root = _artifact(tmp_path, metric=0.9900)
    (root / "stderr.log").write_text("MEASURED_DEAD: cascaded retention collapse\n")

    record = mmr.classify_artifact(root, fastest_baseline_wall_s=100.0)

    assert record.status == "MEASURED_DEAD"
    assert record.failure_reason == "measured_dead"


def test_finds_child_artifacts(tmp_path):
    pack = tmp_path / "pack"
    child = pack / "children" / "child_a"
    child.mkdir(parents=True)
    (child / "exitcode").write_text("0")
    (child / "stdout.log").write_text(
        "[PROFILE] step='Hard Core Mapping' wall=  10.00s metric=0.9800 Δ=+0.0000 (prev=0.9800)\n"
    )
    (child / "stderr.log").write_text("")

    found = list(mmr.iter_artifact_dirs(pack))

    assert found == [child]
