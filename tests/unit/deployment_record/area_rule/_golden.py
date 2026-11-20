"""[U-AREA] The expected values of the area rule's tests, read from data/ (never typed in a test).

- data/golden_tests_r3.json: a byte copy of the design's golden file (revision 3, FT/uarea/r3/out, sha256 in
  data/SOURCES.json);
- data/golden_build_moves.json: the values the build's corrections move (N-12, N-13, G), each with its old value,
  its new value and its reason, recomputed by the design's reference functions plus that one correction;
- data/regime_rows.json: the exploration's 2,003 configurations with their expected prices (N-1);
- data/declarations_build.json: the 11 descriptions' area_rule blocks (W4 places them in the profiles).
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

DATA = Path(__file__).resolve().parent / "data"

#: The relative tolerance of the golden comparisons (design section 6: T-6 to 1e-12).
REL = 1e-12


@lru_cache(maxsize=None)
def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text())


def golden() -> Dict[str, Any]:
    return _load("golden_tests_r3.json")


def moves() -> Dict[str, Any]:
    return _load("golden_build_moves.json")


def regime() -> Dict[str, Any]:
    return _load("regime_rows.json")


def declarations() -> Dict[str, Any]:
    return _load("declarations_build.json")


def sources() -> Dict[str, Any]:
    return _load("SOURCES.json")


def data_digest(name: str) -> str:
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def close(a: float, b: float, rel: float = REL) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)


def all_close(pairs: Iterable, rel: float = REL) -> list:
    """The pairs (label, got, want) that differ beyond ``rel``."""
    return [(label, got, want) for label, got, want in pairs if not close(got, want, rel)]
