"""The E297 planner fixture: RA planner calls with their recorded results, each file pinned by sha256.

Exported once from the RA planner (``x9_0_v3.plan_steps`` under its original front) by
``RA/10_wave1/g5_residency_planner/export_residency_fixture.py``: the 31 spread cases and the 3,341 X10-TA keys of
``v42a_plan.py`` section P1 (stdout 15a52434d17b9a45) and the 9 deep cases of ``plan_wall.py``. The E85 pricing
fixture of CL3 (``e85_price``) holds ``x9_0_v3.price`` on its priced member sets, chip grids, K and energy rows.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "residency_front"

#: sha256 of each fixture's decompressed canonical JSON (sort_keys, compact separators).
FIXTURE_SHA256 = {
    "spread_31": "5ad5ea14c4ae9cd1b9c717fb6159b01bb98d66ff8f89eba7d09ccd36ba633f30",
    "x10ta_3341": "34149bd025d99ec4ccf5ba1e1242e055023fbb380ac66f92641db84590ca38d5",
    "deep_9": "83cbc1a0e76058a992cb926d0ed527d1f9026545cff77fe2971ddd18c0501e04",
    #: CL3's E85 pricing fixture (``RA/11_wave2/explore/cl3/scripts/export_e85_price_fixture.py``).
    "e85_price": "56a2a6944b98c07ce297529f342853081a3ffcb4888a60fa0c8a1d247d84e453",
}


class FixtureDigestError(AssertionError):
    """A fixture's content does not hash to its recorded digest."""


@dataclass(frozen=True)
class PlannerCall:
    case: str
    tiles: Tuple[int, ...]
    live: Tuple[int, ...]
    C: int
    expected: Optional[dict]


def fixture_bytes(name: str) -> bytes:
    return gzip.decompress((FIXTURE_DIR / f"{name}.json.gz").read_bytes())


def check_digest(name: str, data: bytes) -> None:
    got = hashlib.sha256(data).hexdigest()
    if got != FIXTURE_SHA256[name]:
        raise FixtureDigestError(f"fixture {name}: sha256 {got} != recorded {FIXTURE_SHA256[name]}")


def canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _expected(raw: Optional[dict]) -> Optional[dict]:
    if raw is None:
        return None
    return dict(steps=[tuple(s) for s in raw["steps"]], stateless=tuple(raw["stateless"]),
                q_fit=raw["q_fit"], F=raw["F"], fluid_r=raw["fluid_r"])


def load_calls(name: str) -> List[PlannerCall]:
    data = fixture_bytes(name)
    check_digest(name, data)
    return [
        PlannerCall(c["case"], tuple(c["tiles"]), tuple(c["live"]), int(c["C"]), _expected(c["expected"]))
        for c in json.loads(data)["calls"]
    ]


def load_document(name: str) -> dict:
    """A fixture's whole JSON document, its digest checked."""
    data = fixture_bytes(name)
    check_digest(name, data)
    return json.loads(data)


def planner_calls() -> List[PlannerCall]:
    """The 3,372 calls of the P1 check: 31 spread cases, then 3,341 X10-TA keys."""
    return load_calls("spread_31") + load_calls("x10ta_3341")
