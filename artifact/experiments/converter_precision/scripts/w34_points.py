""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
CL6 = HERE.parent
RA = CL6.parents[2]
SCRIPTS = CL6 / "scripts"


NAMES = {
    "W34_IDENTITY_COUNT_DIFFERS": "the identity point's E01 count differs from its row of record's K9 or K14 counts",
    "W34_IDENTITY_IR_DIFFERS": "the B0 identity point's IR function digest differs from its working directory's",
    "W34_IDENTITY_SCHEDULE_DIFFERS": "the B1 identity point's pass_count differs from its row of record's",
    "W34_ROW_OF_RECORD_ABSENT": "a reference result.json is absent or not status complete",
    "W34_PLATFORM_MISMATCH": "identity.platform differs from the point's declaration (cores, allow_scheduling, "
                             "max_schedule_passes, readout_law)",
    "W34_SINGLE_PASS_AT_B1": "the record's pass_count is 1 at B1",
    "W34_BUDGET_SPREAD": "under atol-0 certificates, a member's B0 and B1 counts differ at one adc_bits",
    "W34_NOT_CERTIFIED": "the R and C edges did not both PASS at atol 0 with max |delta| 0",
    "W34_POINT_REFUSED": "the point's status is not ok: a toolchain refusal raised in run_from (a boundary)",
    "W34_POINT_FAILED": "the point's status is not ok for any other cause (harness, door, hook): a defect",
    "W34_POINT_STOPPED": "the point's directory holds no point.json (a cap, a timeout or a stop): never relaunched",
    "NEVRESIM_ON_NON_MNIST": "the point's resolved enable_nevresim_simulation is not False",
    "W34_SRC_DIGEST": "the point or snapshot ran on another src_digest or worktree than the lock's head",
    "W34_SNAPSHOT_MEMBER": "the point ran on a snapshot cut for another member",
    "W34_PRECUT_DIGEST": "the snapshot's kept payloads differ from the member's pre-cut digest",
    "W34_LOCK_SHA256": "the point was run under another lock file (its w34_run.json)",
    "W34_GATE_MIRROR_UNJUSTIFIED": "cl6_point.py's gate_mirror_unjustified is set",
    "W34_E01_SPLIT": "the count is not on the row of record's E01 images (images, index digest)",
    "W34_WQ_SEES_ADC": "the readout-aware arm's check: Readout Law or Weight Quantization metrics, or the WQ model, "
                       "differ across adc_bits at one member and budget",
    "W34_SOURCE_MOVED": "a pinned source (row file, pre-cut payload) differs from the lock",
    "W34_FILE_MOVED": "a pinned file differs from the lock",
    "W34_TIER_PREFIX": "the points are not the declared order with tier 1 its strict prefix",
    "W34_PRECUT_ORDER_UNDECLARED": "the lock declares no precut_order (the order of the pre-cut digest's lines)",
}


DEFECT_STATES = ("W34_IDENTITY_COUNT_DIFFERS", "W34_IDENTITY_IR_DIFFERS", "W34_IDENTITY_SCHEDULE_DIFFERS",
                 "W34_PLATFORM_MISMATCH", "W34_SINGLE_PASS_AT_B1", "W34_BUDGET_SPREAD", "W34_NOT_CERTIFIED",
                 "W34_POINT_FAILED")


class LockError(ValueError):
    ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Tuple[Dict[str, Any], str]:
    ""
    path = Path(path)
    return json.loads(path.read_text()), sha256_file(path)


def ra_path(rel: str) -> Path:
    ""
    p = Path(rel)
    return p if p.is_absolute() else RA / p


def template(text: str) -> str:
    ""
    return str(text).split()[0]


def members(lock: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {m["key"]: m for m in sorted(lock["members"], key=lambda m: int(m["order"]))}


def expected_points(lock: Dict[str, Any]) -> List[Dict[str, Any]]:
    ""
    out, i = [], 0
    identity = int(lock["axes"]["identity_adc_bits"])
    for key, m in members(lock).items():
        for adc in lock["axes"]["adc_bits"]:
            for budget in ("B0", "B1"):
                i += 1
                out.append({"id": i, "member": key, "adc_bits": int(adc), "budget": budget,
                            "cores": m["b0_cores"] if budget == "B0" else m["b1_cores"],
                            "no_scheduling": budget == "B0", "identity_point": int(adc) == identity,
                            "snapshot_member": m["snapshot_member"], "arm": "untrained"})
    return out


def prefix_problems(lock: Dict[str, Any]) -> List[str]:
    ""
    problems = []
    points, want = lock.get("points") or [], expected_points(lock)
    ids = [p.get("id") for p in points]
    if ids != list(range(1, len(want) + 1)):
        problems.append(f"ids {ids[:12]}... are not 1..{len(want)} in order")
    keys = ("id", "member", "adc_bits", "budget", "cores", "no_scheduling", "identity_point", "snapshot_member", "arm")
    for p, w in zip(points, want):
        if any(p.get(k) != w[k] for k in keys):
            problems.append(f"point {p.get('id')} is not the order rule's {w['member']} adc{w['adc_bits']} {w['budget']}")
    tiers = lock.get("tiers") or {}
    t1, t2 = list((tiers.get("tier1") or {}).get("ids") or []), list((tiers.get("tier2") or {}).get("ids") or [])
    if not t1 or t1 != ids[: len(t1)] or t2 != ids[len(t1):] or len(t1) >= len(ids):
        problems.append(f"tier 1 {t1[:3]}...{t1[-1:]} is not a strict prefix with tier 2 the rest")
    problems += [f"point {p['id']} carries tier {p.get('tier')}" for p in points
                 if p.get("tier") != (1 if p["id"] in t1 else 2)]
    for key, m in members(lock).items():
        f = sum(int(c["count"]) for c in m["b0_cores"])
        if [int(c["count"]) for c in m["b1_cores"]] != [math.ceil(int(c["count"]) / 2) for c in m["b0_cores"]]:
            problems.append(f"member {key}: B1 cores {m['b1_cores']} are not ceil(F/2) of F {f}")
    return [f"W34_TIER_PREFIX: {p}" for p in problems]


def tier_points(lock: Dict[str, Any], tier: int) -> List[Dict[str, Any]]:
    ids = set(lock["tiers"][f"tier{int(tier)}"]["ids"])
    return [p for p in lock["points"] if p["id"] in ids]


def rows_root(member: Dict[str, Any], budget: str) -> Path:
    return ra_path(member["b0_rows_root" if budget == "B0" else "b1_rows_root"])


def row_dir(member: Dict[str, Any], budget: str) -> Path:
    return rows_root(member, budget) / member["b0_row" if budget == "B0" else "b1_row"]


def b0_working_directory(member: Dict[str, Any]) -> Path:
    ""
    declared = str(member["b0_working_directory"])
    if not declared.startswith("<"):
        return Path(declared)
    cfg = json.loads((row_dir(member, "B0") / "config.json").read_text())
    return Path(cfg["generated_files_path"]) / f"{cfg['experiment_name']}_phased_deployment_run"


def out_dir(lock: Dict[str, Any], point: Dict[str, Any]) -> Path:
    name = (f"{int(point['id']):02d}_{point['member']}_{point['budget']}_adc{int(point['adc_bits'])}")
    root = template(lock["tiers"]["out_root"]).replace("<id>_<member>_<budget>_adc<N>", name)
    return ra_path(root.rstrip("/"))


def snapshot_dir(lock: Dict[str, Any], member_key: str) -> Path:
    return ra_path(template(lock["tiers"]["snapshot_root"]).replace("<member>", member_key).rstrip("/"))


def point_command(lock: Dict[str, Any], point: Dict[str, Any], python: str) -> List[str]:
    ""
    m = members(lock)[point["member"]]
    return [python, "-B", str(SCRIPTS / "cl6_point.py"), point["snapshot_member"], str(snapshot_dir(lock, m["key"])),
            str(out_dir(lock, point)), "--arm", point["arm"], "--adc", str(int(point["adc_bits"])), "--cores",
            json.dumps(point["cores"]), *(["--no-scheduling"] if point["no_scheduling"] else []), "--rows-root",
            str(rows_root(m, "B0")), "--label", point["label"]]


def cut_command(lock: Dict[str, Any], member_key: str, python: str) -> List[str]:
    m = members(lock)[member_key]
    return [python, "-B", str(SCRIPTS / "cl6_routeb.py"), "cut", m["b0_row"], str(b0_working_directory(m)),
            str(snapshot_dir(lock, member_key)), "--rows-root", str(rows_root(m, "B0"))]


def precut_order(lock: Dict[str, Any]) -> List[str]:
    ""
    order = lock.get("precut_order")
    if not isinstance(order, list) or not order:
        raise LockError("W34_PRECUT_ORDER_UNDECLARED: the lock declares no precut_order, the order of the pre-cut "
                        "digest's lines (draft.md section 5.1): a sorted precut_files_sha256 does not carry it")
    return [str(n) for n in order]


def precut_digest(run_dir: Path, names: List[str]) -> Tuple[str, Dict[str, Optional[str]]]:
    ""
    files = {n: (sha256_file(run_dir / n) if (run_dir / n).is_file() else None) for n in names}
    text = "".join(f"{files[n]}  {n}\n" for n in names)
    return hashlib.sha256(text.encode()).hexdigest(), files


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--tier", type=int, choices=(1, 2), required=True)
    ap.add_argument("--commands", action="store_true")
    ap.add_argument("--python", default="/work/research/mimarsinan/env/bin/python")
    args = ap.parse_args(argv)
    lock, _ = load(args.lock)
    problems = prefix_problems(lock)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    for p in tier_points(lock, args.tier):
        print(" ".join(json.dumps(a) if " " in a else a for a in point_command(lock, p, args.python))
              if args.commands else p["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
