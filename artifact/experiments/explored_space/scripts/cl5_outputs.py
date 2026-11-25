""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "/work/research"
FORMAT = "tcad-run-record/1"
FAMILY = "cl5_exploration"

TABLES_DECLARED, TABLES_WRITTEN = "tables_r5b.md", "tables.md"


def _rel(path: str, root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if rel == ".." or rel.startswith("../") or os.path.isabs(rel):
        raise SystemExit(f"REFUSED: PLACE_OUTSIDE: {path} lies outside the repository root {root}")
    return rel


def record(out: str, run: str, place_root: str = REPO) -> dict:
    sys.path.insert(0, str(HERE))
    import cl5_head
    names = cl5_head.names()
    set_path, _doc = cl5_head.read_set()
    origin = cl5_head.checked_function(_doc)[1]
    o = _rel(out, place_root)
    stamp = subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip()
    return {"format": FORMAT, "family": FAMILY, "head": names["head"], "base_commit": names["base_commit"],
            "set": {"path": str(set_path), "sha256": hashlib.sha256(Path(set_path).read_bytes()).hexdigest()},
            "origin": origin,
            "written_at": stamp, "writer": {"path": str(Path(__file__).resolve()),
                                            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "outputs": {"out": {"root": o, "files": {TABLES_DECLARED: o + "/" + TABLES_WRITTEN}},
                        "census": {"root": o + "/admission", "files": {}},
                        "run": {"root": _rel(run, place_root), "files": {}}}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the exploration's run record of the paper version")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--place-root", default=REPO)
    a = ap.parse_args(argv)
    path = Path(a.record_dir) / "run.json"
    try:
        doc = record(a.out, a.run, a.place_root)
        if path.exists():
            raise SystemExit(f"REFUSED: OUT_EXISTS: {path} exists; the run record is written once")
    except SystemExit as refused:
        print(refused.code, file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"run record {path} (sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}): head {doc['head']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
