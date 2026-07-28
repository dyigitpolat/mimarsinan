"""torch.quantile has a hard 2^24 input cap; every call goes through the guard.

Measured twice at deployment scale: a ViT patch-embed input (77M elements)
killed the mvm boundary calibration, and accumulated activation samples kill
tier-1 t1_04 in activation analysis. ``mapping/support/tensor_stats`` is the
one place allowed to call it — everything else routes through safe_quantile,
which is exact below the cap and subsamples above it.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

_RAW_QUANTILE = re.compile(r"torch\.quantile\s*\(")
_OWNER = "mapping/support/tensor_stats.py"


def _offenders() -> list[str]:
    hits = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel == _OWNER:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if _RAW_QUANTILE.search(line):
                hits.append(f"{rel}:{number}")
    return hits


def test_no_raw_torch_quantile_outside_the_guard():
    assert not _offenders(), (
        "raw torch.quantile calls (use mapping.support.tensor_stats."
        "safe_quantile — torch rejects inputs above 2^24 elements): "
        f"{_offenders()}"
    )


def test_the_guard_module_itself_still_calls_torch():
    # Guards against the rule being satisfied by deleting the implementation.
    owner = (SRC / _OWNER).read_text()
    assert _RAW_QUANTILE.search(owner), "the guard must wrap the real torch call"
