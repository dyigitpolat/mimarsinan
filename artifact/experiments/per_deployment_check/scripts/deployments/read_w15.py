""
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    d = json.loads(p.read_text()); r = d.get("rerun") or {}; q = d["pipeline"]
    est = bool(r and str(r.get("theorem1")).startswith("established") and str(r.get("proposition2")).startswith("established")
               and not r.get("failures") and not r.get("unknown_forms") and not r.get("unread_forms") and not d.get("error"))
    rows.append({"label": d["label"], "error": d.get("error"), "passes": d.get("passes"), "check_s": d.get("check_s"),
                 "unbounded_ir": d.get("unbounded_ir"), "established_on_R": est,
                 "R_theorem1": str(r.get("theorem1"))[:60], "R_proposition2": str(r.get("proposition2"))[:60],
                 "R_ir_tiling": r.get("ir_tiling"), "R_ir_tiling_violations": r.get("ir_tiling_violations"),
                 "R_unknown_forms": r.get("unknown_forms"), "R_unread_forms": r.get("unread_forms"),
                 "R_failures": r.get("failures"), "R_H9": (r.get("statuses") or {}).get("H9"),
                 "RC4A_pipeline_theorem1": str(q.get("theorem1"))[:40], "fields_differing_vs_RC4A_pipeline": d.get("fields_differing")})
print(json.dumps(rows, indent=1))
print("programs:", len(rows), "established on R:", sum(r["established_on_R"] for r in rows),
      "unknown or unread forms on R:", sum(len(r["R_unknown_forms"] or []) + len(r["R_unread_forms"] or []) for r in rows))
