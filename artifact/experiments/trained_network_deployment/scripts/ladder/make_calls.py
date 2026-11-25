""
import hashlib
import json
import os
import sys


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def refuse(name, text):
    print(f"REFUSED: {name}: {text}")
    raise SystemExit(2)


def main(argv):
    snap_p, decl_p, rows, out = argv
    snap, decl = json.load(open(snap_p)), json.load(open(decl_p))
    users = {}
    for member, entry in snap["members"].items():
        groups = [("drawn", entry.get("points") or [])]
        if entry.get("other_plan"):
            groups.append(("other", entry["other_plan"].get("points") or []))
        for role, pts in groups:
            for p in pts:
                u = users.setdefault(p["call"], {"member": member, "role": role, "ds": [], "plans": set(),
                                                 "rows": set()})
                if u["member"] != member:
                    refuse("CALL_SHARED", f"call {p['call']} names points of {u['member']} and {member}")
                u["ds"].append(p["d"])
                u["plans"].add(p.get("plan"))
                u["rows"].add(p.get("template_row"))
    calls = []
    for key, c in snap["calls"].items():
        name = key[len("plan_"):] if key.startswith("plan_") else key
        if name not in users:
            refuse("CALL_UNUSED", f"no point of the snapshot names call {name}")
        u = users[name]
        if u["rows"] != {c["row"]}:
            refuse("CALL_ROW", f"call {name}'s points name rows {sorted(u['rows'])}, the call {c['row']}")
        m = (decl.get("members") or {}).get(u["member"])
        if m is None:
            refuse("MEMBER_UNDECLARED", f"{u['member']} is not a member of the declared ladder")
        d = os.path.join(rows, c["row"])
        res = os.path.join(d, "result.json")
        if not (os.path.isfile(res) and os.path.isfile(os.path.join(d, "config.json"))):
            refuse("ROW_MISSING", f"{c['row']} has no result.json and config.json in {rows}")
        r = json.load(open(res))
        if r.get("status") != "complete" or r.get("smoke"):
            refuse("ROW_NOT_COMPLETE", f"{c['row']}: status {r.get('status')!r}, smoke {r.get('smoke')!r}")
        calls.append({"name": name, "key": u["member"], "role": u["role"], "plans": sorted(map(str, u["plans"])),
                      "row": c["row"], "row_dir": d, "F": m["F"], "ds": u["ds"],
                      "row_result_sha256": sha(res), "row_config_sha256": sha(os.path.join(d, "config.json")),
                      "row_toolchain": ((r.get("inputs") or {}).get("toolchain") or {}).get("commit"),
                      "snapshot_row_config_sha256": c.get("row_config_sha256")})
    doc = {"format": "ladder-paper-calls/1",
           "snapshot": {"file": os.path.abspath(snap_p), "sha256": sha(snap_p)},
           "declared": {"file": os.path.abspath(decl_p), "sha256": sha(decl_p)},
           "rows_root": os.path.abspath(rows), "calls": calls}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    for c in calls:
        print(f"call {c['name']}: {c['key']} {c['role']} on {c['row']} F {c['F']}, {len(c['ds'])} budgets, "
              f"row {c['row_result_sha256'][:16]} toolchain {str(c['row_toolchain'])[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
