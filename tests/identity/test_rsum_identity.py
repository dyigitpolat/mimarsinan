"""The residual-sum branch's standing identity test (PL/rsum/design.md, test 4): every ResNet fixture whose residual
additions already round once on chip has its canonical plan, IR, program and priced record on this branch equal to
the paper version's byte for byte.

Each fixture is planned, built and priced twice by ``identity_digests.py``, once on the export of record (the paper
version, Form R 644b4c3b on base e5f6bb37, read only: its own src and tests in a subprocess) and once on this branch
(its own, in a subprocess), and the four digests must be equal.

Run alone: ``pytest -m rsum_identity tests/identity`` (the group is out of the default suite, which deselects slow)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from identity_digests import FIXTURES

pytestmark = [pytest.mark.rsum_identity, pytest.mark.slow]

BRANCH = Path(__file__).resolve().parents[2]
#: The export of record: the paper version's tree, never written (the subprocess writes no bytecode).
REFERENCE = Path("/home/yigit/scratch/tcad_0dc5568f/ce_r3/export_r4/tlif_export_e5f6bb37_composed_6c8db1e4cd29533c")
SCRIPT = Path(__file__).resolve().parent / "identity_digests.py"
DIGESTS = ("plan", "ir", "program", "priced")


def _start(tree: Path, fixture: str, cwd: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MPLCONFIGDIR=str(cwd / "mpl"))
    return subprocess.Popen([sys.executable, "-s", "-B", str(SCRIPT), "--tree", str(tree), "--fixture", fixture],
                            cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _read(run: subprocess.Popen, tree: Path) -> dict:
    out, err = run.communicate()
    assert run.returncode == 0, f"the run on {tree} failed ({run.returncode}):\n{err[-4000:]}"
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_a_resnet_fixture_plans_builds_and_prices_the_paper_version(fixture, tmp_path):
    assert (REFERENCE / "src" / "mimarsinan").is_dir(), f"the export of record is missing at {REFERENCE}"
    reference = _start(REFERENCE, fixture, tmp_path)
    branch = _start(BRANCH, fixture, tmp_path)
    want, got = _read(reference, REFERENCE), _read(branch, BRANCH)
    assert {k: got[k] for k in DIGESTS} == {k: want[k] for k in DIGESTS}, (
        f"{fixture}: the branch {got['facts']} differs from the paper version {want['facts']}")
