"""[ODIN P7b v2] the package's ORCHESTRATION, gated against a stub cluster.

``run_all.sh`` and ``bootstrap_hacc.sh`` are shell, they run on a machine this
repository cannot reach, and their whole job is decision-making: which partition
to submit to, what counts as already done, whether a second instance may start.
Nothing else in the suite would notice them rotting either.

``scripts/hacc/selftest/run_local_gates.sh`` puts stub ``sbatch``/``sinfo``/
``scontrol``/``squeue``/``groups`` on PATH, backed by the 2026-08-25 hacchead
transcripts, and drives the SHIPPED artifacts exactly as the owner would. This
module is the thin pytest face of that harness: one process, one verdict, and
its transcript on failure.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "scripts" / "hacc" / "selftest" / "run_local_gates.sh"
NOTHING_TO_TEST = 77


@pytest.mark.timeout(240)
def test_the_shipped_package_survives_the_field_cluster(tmp_path):
    """Every gate in the harness, green, or the transcript of the one that is not."""
    if shutil.which("unzip") is None:
        pytest.skip("the harness unpacks the shipped zip; no unzip on this host")
    completed = subprocess.run(
        ["bash", str(HARNESS), str(tmp_path / "work")],
        capture_output=True, text=True, cwd=str(REPO))
    if completed.returncode == NOTHING_TO_TEST:
        pytest.skip("no dist/odin_hacc_package.zip; run scripts/hacc/make_package.py")
    assert completed.returncode == 0, (
        f"the local HACC gates failed:\n{completed.stdout[-8000:]}\n"
        f"{completed.stderr[-2000:]}")
    assert "ALL GATES GREEN" in completed.stdout
