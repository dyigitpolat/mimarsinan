"""Byte-identity fingerprint of one elimination analysis, for differential testing.

Any performance change to the elimination analysis must reproduce this EXACTLY.
Floats are captured as ``float.hex()`` rather than repr, so a 1-ulp drift in a
folded constant is a fingerprint mismatch rather than something a tolerance
could hide -- the whole point of the exercise is that the fast path is the same
function, not an approximation of it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping

__all__ = ["analysis_fingerprint", "fingerprint_digest", "first_difference"]


def _sets(mapping: Mapping[int, Any] | None) -> Dict[str, list]:
    """{id: set} -> {str(id): sorted list}, canonical and JSON-safe."""
    return {str(k): sorted(int(v) for v in (vals or ()))
            for k, vals in sorted((mapping or {}).items())}


def _lattice(values: Mapping[Any, float] | None) -> Dict[str, str]:
    """{(op_id, port): value} -> {"op:port": float.hex()} -- bit-exact."""
    out: Dict[str, str] = {}
    for key, val in sorted((values or {}).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out[f"{int(key[0])}:{int(key[1])}"] = float(val).hex()
    return out


def analysis_fingerprint(result, *, depths=None) -> Dict[str, Any]:
    """Everything an optimisation must preserve, in canonical form.

    ``result`` is a GlobalPruningResult; ``depths`` an optional DepthReplay.
    """
    folds = getattr(result, "constant_folds", None)
    lattice = getattr(folds, "lattice", None)
    fp: Dict[str, Any] = {
        "pruned_rows_per_node": _sets(getattr(result, "pruned_rows_per_node", None)),
        "pruned_cols_per_node": _sets(getattr(result, "pruned_cols_per_node", None)),
        "pruned_rows_per_bank": _sets(getattr(result, "pruned_rows_per_bank", None)),
        "pruned_cols_per_bank": _sets(getattr(result, "pruned_cols_per_bank", None)),
        "constant_lattice": _lattice(getattr(lattice, "values", None)),
        "fixpoint_iterations": int(getattr(result, "fixpoint_iterations", -1)),
    }
    if depths is not None:
        fp["row_depths"] = {
            str(n): {str(i): int(d) for i, d in sorted(m.items())}
            for n, m in sorted((getattr(depths, "row_depths", {}) or {}).items())
        }
        fp["col_depths"] = {
            str(n): {str(i): int(d) for i, d in sorted(m.items())}
            for n, m in sorted((getattr(depths, "col_depths", {}) or {}).items())
        }
        fp["waves"] = int(getattr(depths, "waves", -1))
    return fp


def fingerprint_digest(fp: Mapping[str, Any]) -> str:
    """A stable sha256 over the canonical fingerprint."""
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def first_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    """The first differing field, described precisely; None when identical.

    Reporting WHERE it diverges is what makes a failure actionable -- a bare
    digest mismatch tells you nothing about which invariant broke.
    """
    for key in sorted(set(a) | set(b)):
        if key not in a:
            return f"{key}: missing on the left"
        if key not in b:
            return f"{key}: missing on the right"
        av, bv = a[key], b[key]
        if av == bv:
            continue
        if isinstance(av, dict) and isinstance(bv, dict):
            for sub in sorted(set(av) | set(bv)):
                sa, sb = av.get(sub, "<absent>"), bv.get(sub, "<absent>")
                if sa != sb:
                    return f"{key}[{sub}]: {sa!r} != {sb!r}"
        return f"{key}: {av!r} != {bv!r}"
    return None
