""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PT = HERE.parent
RA = PT.parent.parent
RR = RA / "12_tcad_readiness" / "tiers" / "paper_v6" / "fix5" / "study_build" / "reread"
LOCK = HERE / "lock_r3.json"

FILES = OrderedDict([
    ("r3_nominal/rules_pairs_r3.md", HERE / "rules_pairs_r3.md"),
    ("r3_nominal/rules_pairs_r3.json", HERE / "rules_pairs_r3.json"),
    ("r3_nominal/selections_r3.json", HERE / "selections_r3.json"),
    ("r3_nominal/select_r3.py", HERE / "select_r3.py"),
    ("RR/out/read/P0.json", RR / "out" / "read" / "P0.json"),
    ("RR/out/analysis.json", RR / "out" / "analysis.json"),
    ("pairs_lock.json", PT / "pairs_lock.json"),
    ("pairs_result_record.json", PT / "pairs_result_record.json"),
    ("w06_rows.json", PT / "w06_rows.json"),
    ("pairs_stats.py", PT / "pairs_stats.py"),
    ("rules_pairs.md", PT / "rules_pairs.md"),
    ("rules_pairs.json", PT / "rules_pairs.json"),
])

PINNED = {
    "RR/out/read/P0.json": "134c2343da0146b167abf1a66f9dddc8c53c975e64eec0972a186a32e16d1254",
    "RR/out/analysis.json": "caaba7dee52f931688e1f3c787564c3006045dbdf145b528088e94d81930285e",
    "pairs_lock.json": "65e97a9ff283d01849238c5684c172c70d82d5641d7af536add039a22ec195de",
    "pairs_result_record.json": "c213229dbda4cf986babccab27dcabdf446815ebedd188bc47b1f2a154ddf31a",
    "w06_rows.json": "a39466ed2b6972a098424aebe866b913180e0441776edf92e24e64c8ea22aae8",
    "pairs_stats.py": "ca4e22b3b0b819b72f01ddda2e81e05ab0f1411d24ee8af1f27d27fbeeb98b64",
    "rules_pairs.md": "2acc10bd28ad70e30472596c81d8ba85494d03f07c2b70d7873a659d0d2b1e73",
    "rules_pairs.json": "bf7eac6f9c28c8a593457d2e316b22d68d8b542c69416f8d4a829c9f30c8e13c",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digests() -> OrderedDict:
    return OrderedDict((k, sha256(p)) for k, p in FILES.items())


def problems_of(d: dict) -> list:
    out = []
    for k, pin in PINNED.items():
        if d[k] != pin:
            out.append(f"{k}: sha256 {d[k]}, cited {pin}")
    lock2 = json.loads((PT / "pairs_lock.json").read_text())
    f2 = lock2["digests"]["files_sha256"]
    for k in ("rules_pairs.md", "rules_pairs.json", "pairs_stats.py"):
        if f2.get(k) != d[k]:
            out.append(f"pairs_lock.json pins {k} at {f2.get(k)}, the file is {d[k]}")
    rec = json.loads((PT / "pairs_result_record.json").read_text())
    if rec.get("pairs_lock_sha256") != d["pairs_lock.json"]:
        out.append("pairs_result_record.json does not pin pairs_lock.json")
    if rec.get("w06_rows_sha256") != d["w06_rows.json"]:
        out.append("pairs_result_record.json does not pin w06_rows.json")
    rv = json.loads((HERE / "rules_pairs_r3.json").read_text())
    sel = json.loads((HERE / "selections_r3.json").read_text())
    fc, sf = rv["fourth_contrast"], sel["fourth_contrast"]
    for k in ("J", "J_job", "FO", "FO_job"):
        if fc[k] != sf[k]:
            out.append(f"the revision's fourth contrast {k} {fc[k]!r} differs from selections_r3.json {sf[k]!r}")
    if not sel.get("matches_reread_table") or sel.get("problems"):
        out.append("selections_r3.json does not match the re-read's table")
    if rv["selections"]["sources"]["P0.json"] != d["RR/out/read/P0.json"] or \
            rv["selections"]["sources"]["analysis.json"] != d["RR/out/analysis.json"]:
        out.append("the revision cites other digests of P0.json or analysis.json")
    return out


def date_now() -> OrderedDict:
    iso = subprocess.run(["date", "-Is"], capture_output=True, text=True, check=True).stdout.strip()
    unix = int(subprocess.run(["date", "+%s"], capture_output=True, text=True, check=True).stdout.strip())
    return OrderedDict(locked_at=iso, locked_at_unix=unix)


def verify() -> int:
    if not LOCK.exists():
        print("REFUSED: lock_r3.json does not exist")
        return 2
    lk = json.loads(LOCK.read_text())
    d = digests()
    bad = [k for k, v in lk["files_sha256"].items() if d.get(k) != v]
    if bad or set(lk["files_sha256"]) != set(d):
        print("LOCK DOES NOT VERIFY: " + ", ".join(bad or ["file list"]))
        return 1
    print(f"lock_r3.json verifies ({len(d)} files, locked {lk['locked_at']})")
    return 0


def main(argv: list) -> int:
    if "--verify" in argv:
        return verify()
    if LOCK.exists():
        print("REFUSED: lock_r3.json exists (the lock is written once)")
        return 2
    d = digests()
    pr = problems_of(d)
    if pr:
        print("REFUSED:\n  " + "\n  ".join(pr))
        return 3
    t = date_now()
    out = OrderedDict(
        label="the lock of revision 3 of headline_v46_pairs (rules_pairs_r3.md section 5)",
        files_sha256=d,
        paths=OrderedDict((k, str(p.relative_to(RA))) for k, p in FILES.items()),
        checks=["every input equals the digest the revision cites",
                "pairs_lock.json pins rules_pairs.md, rules_pairs.json and pairs_stats.py at their digests",
                "pairs_result_record.json pins pairs_lock.json and w06_rows.json at their digests",
                "selections_r3.json names the revision's fourth contrast and matches the re-read's table"],
        locked_at=t["locked_at"], locked_at_unix=t["locked_at_unix"],
        by="make_lock_r3.py (date -Is and date +%s, read after every hash)",
        time_of_record="this lock's locked_at is the time of record of rules_pairs_r3.md (revision 3)",
    )
    LOCK.write_text(json.dumps(out, indent=1) + "\n")
    print(f"lock_r3.json written, sha256 {sha256(LOCK)}, locked at {t['locked_at']} ({t['locked_at_unix']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
