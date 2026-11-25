""
import json
import sys
from pathlib import Path

SCR = Path(sys.argv[1])
E2RUN = Path("/scratch/tcad/thru/e2/run")
SKIP = {"wall_s", "head"}


def diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            out += diffs(a.get(k), b.get(k), f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diffs(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


report = {}
for name in ("w0_c256_B", "e2_c256_b131072_g256_m0_s0", "e2_c576_b131072_g432_m0_s0"):
    a = json.loads((SCR / "run" / f"{name}.json").read_text())
    b = json.loads((E2RUN / f"{name}.json").read_text())
    dd = diffs(a, b)
    report[name] = {"equal": not dd, "differing": dd[:40], "heads": [a.get("head"), b.get("head")]}
report["summary"] = {"equal": sum(1 for v in report.values() if v["equal"]), "compared": len(report)}
print(json.dumps(report, indent=1, sort_keys=True))
