""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

FORM_KEY = "form_sha256"
COMMIT_KEY = "commit"
HEAD_LOCK_VAR = "E2E_HEAD_LOCK"
PLAN_LOCK = "plan_lock.json"


def names(head: dict | None, commit: str | None = None) -> dict:
    ""
    if not isinstance(head, dict):
        return {"head": commit, "base_commit": commit}
    base = head.get(COMMIT_KEY)
    return {"head": str(head.get(FORM_KEY) or base), "base_commit": str(base)}


def lock_sets(lock: dict | None) -> list:
    ""
    lock = lock or {}
    sets = list((lock.get("carry") or {}).get("toolchain_sets") or []) or list(lock.get("toolchain_sets") or [])
    return sorted((s for s in sets if isinstance(s, dict)), key=lambda s: -int(s.get("revision") or 0))


def row_set(lock: dict | None, row: str) -> dict | None:
    ""
    return next((s for s in lock_sets(lock) if row in (s.get("rows") or [])), None)


def of_sets(sets: list, what: str = "the sets") -> dict:
    ""
    got = {json.dumps(names(s.get("head")), sort_keys=True) for s in sets}
    if len(got) != 1:
        raise SystemExit(f"REFUSED: HEAD_NAMES_SPLIT: {what} name {len(got)} heads "
                         f"({sorted(json.loads(g)['head'][:16] for g in got)}): a record names one head")
    return json.loads(got.pop())


def lock_of_session(plan: Path | None) -> Path | None:
    ""
    declared = os.environ.get(HEAD_LOCK_VAR)
    if declared:
        return Path(declared)
    if plan is not None and (Path(plan) / PLAN_LOCK).is_file():
        return Path(plan) / PLAN_LOCK
    return None


def session(tc: dict, row: str, plan: Path | None = None) -> dict:
    ""
    path = lock_of_session(plan)
    if path is None:
        return dict(tc)
    if not path.is_file():
        raise SystemExit(f"REFUSED: HEAD_LOCK_ABSENT: the session's lock {path} ({HEAD_LOCK_VAR} or the plan's "
                         f"{PLAN_LOCK}) is absent")
    raw = path.read_bytes()
    s = row_set(json.loads(raw), row)
    if s is None:
        return dict(tc)
    n = names(s.get("head"))
    if tc.get(COMMIT_KEY) != n["base_commit"]:
        raise SystemExit(f"REFUSED: HEAD_BASE_MISMATCH: {row} runs on an export of {tc.get(COMMIT_KEY)}, while its set "
                         f"in {path} is cut on {n['base_commit']}")
    out = dict(tc)
    out.update({COMMIT_KEY: n["head"], "base_commit": n["base_commit"],
                "head_lock": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the head's name of a lock's set (one per line: the head, the base)")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--row", action="append", default=[])
    a = ap.parse_args(argv)
    lock = json.loads(Path(a.lock).read_text())
    if a.row:
        sets = []
        for r in a.row:
            s = row_set(lock, r)
            if s is None:
                raise SystemExit(f"REFUSED: HEAD_NAMES_NO_SET: {r} lies in no toolchain set of {a.lock}")
            sets.append(s)
    else:
        sets = lock_sets(lock)
    n = of_sets(sets, f"the sets of {a.lock}" + (f" for {a.row}" if a.row else ""))
    print(f"{n['head']}\n{n['base_commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
