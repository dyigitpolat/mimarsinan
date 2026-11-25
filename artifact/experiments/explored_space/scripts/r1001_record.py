""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    stdout, guard, out_path, run_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    state = sys.argv[5] if len(sys.argv) > 5 else None
    record = None
    for line in open(stdout, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{") and '"stages"' in line:
            record = json.loads(line)
    if record is None:
        raise SystemExit(f"{stdout} holds no stage record")
    verdict = open(guard + ".exit").read().strip() if os.path.exists(guard + ".exit") else None
    peak = open(guard + ".peak").read().strip() if os.path.exists(guard + ".peak") else None
    stages = record["stages"]
    if "hard_core_mapping" not in stages and verdict is not None and verdict != "0":
        stopped = {"name": "MemoryGuardStop" if verdict.startswith("rss_guard") else "TimeoutStop"
                   if verdict.startswith("timeout") else "Killed" if verdict == "137" else "ExitCode",
                   "reason": f"the caller's guard (guarded.sh) stopped the run: {verdict}",
                   "peak_summed_pss_gib": None if peak is None else float(peak),
                   "class": "MEASURED (guarded.sh: the session's summed proportional set size, sampled every 2 s)"}
        stages["hard_core_mapping"] = {"refusal": stopped,
                                       "stage": "the resident hybrid mapping build (build_hybrid_mapping_for_pipeline, "
                                                "allow_scheduling false, one 256 x 256 core per IR neural core)"}
    record["guard"] = {"log": guard, "verdict": verdict, "peak_summed_pss_gib": None if peak is None else float(peak)}
    record["code_state"] = state
    capacity = run_dir / "capacity" / "resnet1001_preact__value.jsonl"
    points = []
    if capacity.exists():
        lines = capacity.read_text().strip().splitlines()
        complete = bool(lines) and '"trailer": true' in lines[-1]
        for line in lines:
            row = json.loads(line)
            if row.get("network") == "resnet1001_preact" and row.get("x_fluid") is not None:
                points.append({k: row.get(k) for k in ("target", "cores", "softcores", "packed_tiles", "x_fluid",
                                                        "x_packed", "layout_wall_s", "peak_rss_bytes")})
        record["shape_only_points"] = points
        record["shape_only_job"] = {"id": "x:resnet1001_preact:value", "complete": complete, "run_dir": str(run_dir)}
    else:
        record["shape_only_points"] = []
        record["shape_only_job"] = {"id": "x:resnet1001_preact:value", "complete": False, "run_dir": str(run_dir),
                                    "absent": "the capacity job has written no output in this run dir"}
    record["provisional"] = ("provisional: toolchain output of one run; packing and value-path capabilities in "
                             "flight change these numbers until the integrated head reruns them")
    Path(out_path).write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({"stages": list(stages), "hard_core_mapping": stages.get("hard_core_mapping"),
                      "shape_only_points": len(points)}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
