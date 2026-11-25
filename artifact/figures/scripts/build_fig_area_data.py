""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
P = F3.parent
ROOT = "tcad_paper"
NAME = "fig_area"


UNIT = "records/tiers/fidelity/uarea/build_r6/w9_paper_644b4c3bc5d7/k02/fig_area_unit_r2.json"
PIN = "4b28411c00a95c7d5a03fe3f2410713c4b4710cf11a8b5f18fef40622155fa57"


MARK_KINDS = ("measured", "derived", "model")
STATUSES = ("priced", "named refusal")
ROLES = ("upper bound", "bounds nothing", "range low", "range high")

READ_BEFORE = ("parity", "stats", "stacks", "ladder", "dies")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(why: str) -> None:
    sys.exit(f"build_{NAME}_data: {why}")


def check(d: dict) -> None:
    ""
    if d.get("name") != NAME:
        stop(f"the input's name is {d.get('name')!r}")
    for k in READ_BEFORE + ("named_refusals", "stacks_all", "stacks_axis", "displays"):
        if k not in d:
            stop(f"the input holds no field {k!r}")
    for p in d["parity"]:
        if p.get("mark_kind") not in MARK_KINDS:
            stop(f"{p['chip']}: mark_kind {p.get('mark_kind')!r}")
        if p.get("strict_status") not in (None,) + STATUSES:
            stop(f"{p['chip']}: strict_status {p.get('strict_status')!r}")
        if (p.get("strict_status") is None) != (not p["strict_ratio"]):
            stop(f"{p['chip']}: a strict ratio without a status, or a status without a ratio")
        if (p["mark_kind"] == "model") != (not p["silicon"]):
            stop(f"{p['chip']}: mark_kind 'model' and the silicon field disagree")
    head = [p for p in d["parity"] if p["headline"]]
    if len(head) != d["stats"]["n"]:
        stop(f"{len(head)} headline points, the statistics count {d['stats']['n']}")
    priced = {p["chip"] for p in d["parity"]}
    for e in d["named_refusals"]:
        if e.get("status") != "named refusal" or e.get("mark_kind") not in MARK_KINDS or not e.get("source"):
            stop(f"{e.get('chip')}: a named refusal without its status, mark kind or source")
        if e["chip"] in priced:
            stop(f"{e['chip']}: a named refusal that is also a parity point (it would enter a statistic)")
        for r in e["readings"]:
            if r.get("role") not in ROLES:
                stop(f"{e['chip']}: reading role {r.get('role')!r}")
            for f in ("ratio_display", "mm2_display"):
                if not r.get(f, {}).get("text"):
                    stop(f"{e['chip']}: a reading without its {f}")
    focal = d["displays"]["focal_label"]
    parts = focal["parts"]
    if parts["n"]["value"] != d["stats"]["n"] or parts["f"]["value"] != d["stats"]["worst_factor"]:
        stop("the focal label's parts are not the statistics' values")
    if parts["f"]["printed_value"] < d["stats"]["worst_factor"]:
        stop("the focal label prints its bound rounded inward")
    if focal["text"] != focal["template"].format(**{k: v["printed"] for k, v in parts.items()}):
        stop("the focal label's text is not its template filled with its printed parts")

    order = [s["chip"] for s in d["stacks_all"]]
    if [c for c in order if c in {s["chip"] for s in d["stacks"]}] != [s["chip"] for s in d["stacks"]]:
        stop("stacks_all does not keep the order of stacks")
    for s in d["stacks_all"]:
        if s.get("stack_status") not in STATUSES:
            stop(f"{s['chip']}: stack_status {s.get('stack_status')!r}")
        refused = False
        for g, v in s["groups"].items():
            if v.get("status") not in STATUSES:
                stop(f"{s['chip']} {g}: status {v.get('status')!r}")
            if (v["status"] == "named refusal") != (v["estimate_um2"] is None):
                stop(f"{s['chip']} {g}: a refused group with an estimate, or a priced group without one")
            if v["status"] == "named refusal" and not v.get("reason"):
                stop(f"{s['chip']} {g}: a refused group without its reason")
            refused |= v["status"] == "named refusal"
        if refused != (s["stack_status"] == "named refusal"):
            stop(f"{s['chip']}: stack_status and the groups' statuses disagree")
        if refused != (s["estimate_total_um2"] is None):
            stop(f"{s['chip']}: a refused stack with an estimate total, or a priced stack without one")
    ax = d["stacks_axis"]
    longest = max(s["estimate_total_um2"] / s["published_total_um2"] for s in d["stacks_all"]
                  if s["silicon"] and s["estimate_total_um2"] is not None)
    if abs(longest - ax["longest_estimate_ratio"]) > 1e-12 or ax["axis_end_multiple"] < longest:
        stop("stacks_axis: the longest estimate bar or the axis end is not the stacks'")


def main() -> None:
    src = P / UNIT
    got = sha256(src)
    if got != PIN:
        stop(f"{src.name}: sha256 {got[:8]} is not the pinned {PIN[:8]}")
    d = json.loads(src.read_text())
    check(d)

    declared = d["sources"]["fig_area_data.json"]
    side = P / Path(declared["path"]).relative_to(ROOT)
    side_state = "not on disk"
    if side.exists():
        if sha256(side) != declared["sha256"]:
            stop(f"{side.name}: sha256 differs from the digest the input declares")
        side_state = "on disk, digest equal"
    out = dict(d)
    if "provenance" in out:
        stop("the input already holds a field 'provenance'")
    out["provenance"] = dict(
        builder=f"{ROOT}/figures_v3/scripts/{Path(__file__).name}",
        built=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        input=dict(path=f"{ROOT}/{UNIT}", sha256=PIN),
        input_data_file=dict(path=declared["path"], sha256=declared["sha256"], state=side_state),
        rule="every field of the input carried over as it stands (the input's own time is its field 'built'); "
             "this field added; no value computed or typed")
    dst = F3 / "data" / f"{NAME}.json"
    tmp = dst.with_name(f".{NAME}.json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    if {k: v for k, v in json.loads(tmp.read_text()).items() if k != "provenance"} != d:
        tmp.unlink()
        stop("the written file does not hold the input's fields")
    os.replace(tmp, dst)
    refused = [e["chip"] for e in d["named_refusals"]]
    print(f"{dst}: {len(d['parity'])} parity points ({d['stats']['n']} in the statistics), "
          f"{len(refused)} named refusals ({', '.join(refused)}), {len(d['stacks_all'])} stacks, "
          f"{sum(len(m['rows']) for m in d['ladder'])} ladder rows; focal label {d['displays']['focal_label']['text']!r}; "
          f"input {PIN[:8]}, its data file {declared['sha256'][:8]} ({side_state})")


if __name__ == "__main__":
    main()
