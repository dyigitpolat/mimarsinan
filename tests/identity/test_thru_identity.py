"""The branch's standing identity test (PL/thru/design.md (e), Identity): with the layer copies' keys absent, and with
``layer_replication`` declared false, every fixture's canonical plan, program and priced record on this branch equal
the paper version's byte for byte.

Each fixture is planned, built and priced twice by ``identity_digests.py``, once on the export of record (the paper
version, Form R 644b4c3b on base e5f6bb37, read only: its own src, tests and scripts in a subprocess) and once on this
branch (its own, in a subprocess), and the three digests must be equal.

Run alone: ``pytest -m thru_identity tests/identity`` (the group is out of the default suite, which deselects slow)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from identity_digests import FIXTURES, CANDIDATE_FIXTURES

pytestmark = [pytest.mark.thru_identity, pytest.mark.slow]

BRANCH = Path(__file__).resolve().parents[2]
#: The export of record: the paper version's tree, never written (the subprocess writes no bytecode).
REFERENCE = Path("/home/yigit/scratch/tcad_0dc5568f/ce_r3/export_r4/tlif_export_e5f6bb37_composed_6c8db1e4cd29533c")
SCRIPT = Path(__file__).resolve().parent / "identity_digests.py"
DIGESTS = ("plan", "program", "priced")


def _start(tree: Path, fixture: str, declare: str, compare: str, cwd: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MPLCONFIGDIR=str(cwd / "mpl"))
    return subprocess.Popen([sys.executable, "-s", "-B", str(SCRIPT), "--tree", str(tree), "--fixture", fixture,
                             "--declare", declare, "--compare", compare], cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _read(run: subprocess.Popen, tree: Path) -> dict:
    out, err = run.communicate()
    assert run.returncode == 0, f"the run on {tree} failed ({run.returncode}):\n{err[-4000:]}"
    return json.loads(out.strip().splitlines()[-1])


def _assert_identical(fixture: str, declare: str, tmp_path: Path) -> None:
    assert (REFERENCE / "src" / "mimarsinan").is_dir(), f"the export of record is missing at {REFERENCE}"
    reference = _start(REFERENCE, fixture, "absent", declare, tmp_path)
    branch = _start(BRANCH, fixture, declare, declare, tmp_path)
    want, got = _read(reference, REFERENCE), _read(branch, BRANCH)
    assert {k: got[k] for k in DIGESTS} == {k: want[k] for k in DIGESTS}, (
        f"{fixture} ({declare}): the branch {got['facts']} differs from the paper version {want['facts']}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("fixture", sorted(FIXTURES) + list(CANDIDATE_FIXTURES))
def test_absent_keys_plan_build_and_price_the_paper_version(fixture, tmp_path):
    _assert_identical(fixture, "absent", tmp_path)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("fixture", sorted(FIXTURES) + list(CANDIDATE_FIXTURES))
def test_replication_declared_false_plans_builds_and_prices_the_paper_version(fixture, tmp_path):
    """[thru:k] layer_replication false (declared on every fixture, on C1's pipelined arms, its own echo left out) is
    the absent program."""
    _assert_identical(fixture, "false", tmp_path)
