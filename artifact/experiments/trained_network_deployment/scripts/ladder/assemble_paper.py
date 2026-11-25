""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = "/work/research"
POINT_FIELDS = ("d", "share", "cores", "verdict", "passes", "most_divided_segment_passes", "segments", "scheduled",
                "sync_count", "max_cores_per_pass", "per_segment_passes_histogram", "residency_composition",
                "refusal", "above_wall", "wall_s", "unscheduled_note")
MOVE_FIELDS = ("cores", "verdict", "passes", "most_divided_segment_passes", "segments", "scheduled")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rel(p):
    p = os.path.realpath(p)
    return p[len(REPO) + 1:] if p.startswith(REPO + "/") else p


def refuse(name, text):
    print(f"REFUSED: {name}: {text}")
    raise SystemExit(2)


def executed(path):
    doc = json.load(open(path))
    dep, sched = doc["deployment"], (doc.get("record") or {}).get("schedule") or {}
    return {"record": rel(path), "record_sha256": sha(path), "status": doc.get("status"), "smoke": doc.get("smoke"),
            "geometry": doc.get("geometry"), "toolchain": ((doc.get("inputs") or {}).get("toolchain") or {}).get("commit"),
            "cores": dep["declared_count"], "passes": sched.get("pass_count"),
            "most_divided_segment_passes": dep.get("max_passes"), "segments_scheduled": dep.get("segments_scheduled"),
            "neural_stages": dep.get("neural_stages")}


def compare(point, ex):
    checks = {"cores": point["cores"] == ex["cores"], "passes": point.get("passes") == ex["passes"]}
    if point.get("scheduled"):
        checks["most_divided_segment_passes"] = point.get("most_divided_segment_passes") == ex[
            "most_divided_segment_passes"]
        checks["segments"] = point.get("segments") == ex["segments_scheduled"]
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    for a in ("calls", "outputs", "snapshot", "declared", "rows-root", "ladder-lock", "ladder-rows", "out"):
        ap.add_argument("--" + a, required=True)
    a = ap.parse_args(argv)
    calls = json.load(open(a.calls))
    snap, decl = json.load(open(a.snapshot)), json.load(open(a.declared))
    outs = {}
    for c in calls["calls"]:
        p = os.path.join(a.outputs, f"plan_{c['name']}.json")
        if not os.path.isfile(p):
            refuse("CALL_MISSING", f"no output {p}")
        outs[c["name"]] = (p, json.load(open(p)))
    heads = {json.dumps({k: o["header"]["toolchain"].get(k) for k in ("head", "base_commit", "export_record_sha256",
                                                                    "src_manifest_sha256")}, sort_keys=True)
             for _p, o in outs.values()}
    if len(heads) != 1:
        refuse("HEAD_SPLIT", f"the calls name {len(heads)} heads: {sorted(heads)}")
    for name, (_p, o) in outs.items():
        tc = o["header"]["toolchain"]
        if not (tc.get("export_verified") and tc.get("export_verified_at_end")):
            refuse("EXPORT_NOT_VERIFIED", f"call {name}: the export is not verified at its start and its end")
        if not (o["header"].get("K0_equals_row") and o["header"].get("F_packed_equals_F")):
            refuse("CALL_CHECKS", f"call {name}: K0_equals_row {o['header'].get('K0_equals_row')}, F_packed_equals_F "
                                  f"{o['header'].get('F_packed_equals_F')}")
    head = json.loads(heads.pop())
    by_call = {c["name"]: c for c in calls["calls"]}
    doc = {
        "format": snap["format"],
        "stamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "what": snap["what"],
        "head": dict(head, name="the paper version (Form R) on the export of record, every call's header"),
        "snapshot": {"file": rel(a.snapshot), "sha256": sha(a.snapshot), "stamp": snap.get("stamp"),
                     "head": "the snapshot's toolchain (its calls' toolchain_head)"},
        "declared_ladder": dict(snap["declared_ladder"]),
        "method": dict(snap.get("method") or {}, **{
            "toolchain": f"the export of record (paper version {head['head']} on {head['base_commit']}, export record "
                         f"{head['export_record_sha256']}), verified against its record at each call's start and end",
            "call": (snap.get("method") or {}).get("call", "") + "; the snapshot's calls, each on the paper "
                    "version's row of the same name (route (b)'s rows of record on the paper version)",
            "script": "code/plan_ladder_paper.py (sha256 in each call's header), its calls code/make_calls.py, this "
                      "record code/assemble_paper.py"}),
        "calls": {}, "members": {}, "gate": [], "gate_other_plan": [],
    }
    if sha(os.path.join(REPO, snap["declared_ladder"]["file"])) != snap["declared_ladder"]["sha256"] \
            if os.path.isfile(os.path.join(REPO, snap["declared_ladder"]["file"])) else False:
        refuse("DECLARED_MOVED", f"{snap['declared_ladder']['file']} is not at the snapshot's sha256")
    for name, (p, o) in outs.items():
        h = o["header"]
        doc["calls"][f"plan_{name}"] = {
            "file": rel(p), "sha256": sha(p), "row": h["row"], "row_config_sha256": h["row_config_sha256"],
            "toolchain_head": h["toolchain"]["head"], "toolchain_base_commit": h["toolchain"]["base_commit"],
            "export_verified": h["toolchain"]["export_verified"],
            "export_verified_at_end": h["toolchain"]["export_verified_at_end"],
            "script_sha256": h["script_sha256"], "driver_unchanged": h.get("driver_unchanged"),
            "K0_equals_row": h["K0_equals_row"], "F_packed": h.get("F_packed"),
            "F_packed_equals_F": h["F_packed_equals_F"], "wall_s": round(float(h.get("wall_s") or 0), 1)}
    plan_of = {}
    for m, e in snap["members"].items():
        for p in list(e.get("points") or []) + list((e.get("other_plan") or {}).get("points") or []):
            plan_of[(p["call"], float(p["d"]))] = p.get("plan")

    def points_of(names):
        pts = []
        for n in names:
            o = outs[n][1]
            for pt in o["points"]:
                row = {k: pt.get(k) for k in POINT_FIELDS if k in pt}
                row.update(template_row=o["header"]["row"], call=n, plan=plan_of.get((n, float(pt["d"]))))
                pts.append(row)
        return sorted(pts, key=lambda r: float(r["d"]))

    moves, gate_rows, gate_other = [], [], []
    for m, se in snap["members"].items():
        drawn = sorted({p["call"] for p in se["points"]}, key=lambda n: [p["call"] for p in se["points"]].index(n))
        other = sorted({p["call"] for p in (se.get("other_plan") or {}).get("points") or []})
        for n in drawn + other:
            if n not in outs:
                refuse("CALL_MISSING", f"{m}: no output of call {n}")
        resident = [c for c in drawn if all(float(d) == 1.0 for d in by_call[c]["ds"])]
        geo = executed(os.path.join(a.rows_root, outs[(resident or drawn)[0]][1]["header"]["row"], "result.json"))["geometry"]
        if se.get("geometry") not in (None, geo):
            refuse("GEOMETRY", f"{m}: the snapshot's geometry {se.get('geometry')} is not the paper row's {geo}")
        pts = points_of(drawn)
        refusals = [p for p in pts if p["verdict"] != "accepted"]
        entry = {"network": (decl["members"].get(m) or {}).get("network", se.get("network")),
                 "F": decl["members"][m]["F"], "plan": se.get("plan"), "geometry": geo,
                 "ladder_complete": not refusals, "refusals": [{"d": p["d"], "verdict": p["verdict"]} for p in refusals],
                 "points": pts}
        for group, spts, ppts in (("drawn", se["points"], pts),
                                  ("other", (se.get("other_plan") or {}).get("points") or [], points_of(other))):
            for sp in spts:
                pp = next((p for p in ppts if float(p["d"]) == float(sp["d"])), None)
                if pp is None:
                    moves.append({"member": m, "plan_group": group, "d": sp["d"], "field": "point", "snapshot": "present",
                                  "paper": "absent"})
                    continue
                for f in MOVE_FIELDS:
                    if sp.get(f) != pp.get(f):
                        moves.append({"member": m, "plan_group": group, "plan": sp.get("plan"), "d": sp["d"],
                                      "field": f, "snapshot": sp.get(f), "paper": pp.get(f)})
        for g in snap.get("gate") or []:
            if g["member"] != m:
                continue
            pt = next(p for p in pts if float(p["d"]) == float(g["d"]))
            ex = executed(os.path.join(a.rows_root, g["executed_row"], "result.json"))
            ch = compare(pt, ex)
            gate_rows.append({"member": m, "plan": g["plan"], "d": g["d"], "executed_row": g["executed_row"],
                              "executed": ex, "planned": {k: pt.get(k) for k in ("cores", "verdict", "passes",
                                                                               "most_divided_segment_passes",
                                                                               "segments", "scheduled")},
                              "checks": ch, "equal": all(ch.values())})
        entry["gate_equal"] = all(g["equal"] for g in gate_rows if g["member"] == m)
        if other:
            opts = points_of(other)
            og = []
            for g in snap.get("gate_other_plan") or []:
                if g["member"] != m:
                    continue
                pt = next(p for p in opts if float(p["d"]) == float(g["d"]))
                ex = executed(os.path.join(a.rows_root, g["executed_row"], "result.json"))
                ch = compare(pt, ex)
                og.append({"member": m, "plan": g["plan"], "d": g["d"], "executed_row": g["executed_row"],
                           "executed": ex, "planned": {k: pt.get(k) for k in ("cores", "verdict", "passes",
                                                                            "most_divided_segment_passes", "segments",
                                                                            "scheduled")},
                           "checks": ch, "equal": all(ch.values())})
            gate_other += og
            why = "; ".join(
                f"at d {g['d']} the planner's program has {g['planned']['segments']} segments ({g['planned']['passes']} "
                f"passes, most divided segment {g['planned']['most_divided_segment_passes']}) where the executed row "
                f"{g['executed_row']} has {g['executed']['segments_scheduled']} scheduled segments "
                f"({g['executed']['passes']} passes, most divided segment {g['executed']['most_divided_segment_passes']})"
                for g in og if not g["equal"])
            entry["other_plan"] = {"plan": (se.get("other_plan") or {}).get("plan"),
                                   "gate_equal": all(g["equal"] for g in og), "drawn": False,
                                   "why_not_drawn": ("the gate does not hold: " + why) if why else
                                   "not drawn: the figure draws the drawn plan's points only",
                                   "points": opts}
        doc["members"][m] = entry
    doc["gate"], doc["gate_other_plan"] = gate_rows, gate_other
    doc["gate_holds"] = {m: e["gate_equal"] for m, e in doc["members"].items()}
    doc["moves"] = moves
    lock = json.load(open(a.ladder_lock))
    ladder = []
    for row in lock["rows"]:
        member = [m for m in doc["members"] if row.startswith(m + ".")]
        if len(member) != 1:
            refuse("LADDER_MEMBER", f"{row} belongs to {len(member)} members of the plan")
        m = member[0]
        ex = executed(os.path.join(a.ladder_rows, row, "result.json"))
        twins = [p for p in doc["members"][m]["points"] if p["cores"] == ex["cores"]]
        e = {"row": row, "member": m, "executed": ex}
        if len(twins) != 1:
            e.update(planned=None, checks={"twin": False}, equal=False,
                     why=f"{len(twins)} planned points of {m} at {ex['cores']} cores, expected 1")
        else:
            ch = compare(twins[0], ex)
            ch["geometry"] = ex["geometry"] == doc["members"][m]["geometry"]
            e.update(planned={k: twins[0].get(k) for k in ("d", "plan", "cores", "passes",
                                                           "most_divided_segment_passes", "segments", "scheduled")},
                     checks=ch, equal=all(ch.values()))
        ladder.append(e)
    doc["executed_ladder"] = ladder
    doc["executed_ladder_match"] = all(e["equal"] for e in ladder)
    with open(a.out, "w") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out} {sha(a.out)}")
    print(f"head {head['head']} base {head['base_commit']}")
    for g in gate_rows + gate_other:
        print(f"gate {g['member']} {g['plan']} d {g['d']} {g['executed_row']}: planned {g['planned']['cores']} "
              f"{g['planned']['passes']} executed {g['executed']['cores']} {g['executed']['passes']} "
              f"{'EQUAL' if g['equal'] else 'DIFFERS ' + json.dumps(g['checks'])}")
    for mv in moves:
        print(f"MOVED {mv['member']} {mv.get('plan')} d {mv['d']} {mv['field']}: {mv['snapshot']} -> {mv['paper']}")
    print(f"moves: {len(moves)}")
    for e in ladder:
        line = (f"ladder {e['row']}: executed cores {e['executed']['cores']} passes {e['executed']['passes']}; planned "
                f"{(e['planned'] or {}).get('cores')} {(e['planned'] or {}).get('passes')}")
        print(("" if e["equal"] else "FINDING: ") + line + (" EQUAL" if e["equal"] else f" {json.dumps(e['checks'])}"))
    print(f"executed ladder match: {doc['executed_ladder_match']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
