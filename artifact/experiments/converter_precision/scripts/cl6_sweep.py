""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cl6_common as K

PY = sys.executable


CURRENT = K.src_digest()


def done(out: Path) -> bool:
    p = out / "point.json"
    if not p.exists():
        return False
    try:
        point = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return point.get("status") == "ok" and point.get("src_digest") == CURRENT


def run_one(point: dict) -> tuple:
    out = Path(point["out"])
    if (out / "point.json").exists():
        return point["label"], "POINT_PRESENT: kept as recorded, not run", 0.0
    out.mkdir(parents=True, exist_ok=True)
    snap = K.RUNS / "snap" / point["member"]
    cmd = [PY, "-B", str(HERE / "cl6_point.py"), point["member"], str(snap), str(out), "--arm", point["arm"],
           "--adc", str(point["adc"]), "--cores", json.dumps(point["cores"]), "--label", point["label"]]
    if point.get("no_scheduling"):
        cmd.append("--no-scheduling")
    if point.get("keep_run"):
        cmd.append("--keep-run")
    t0 = time.time()
    env = {**os.environ, "CL6_TEST_GUARD_LOG": str(out / "guard.jsonl")}
    with open(out / "driver_stdout.txt", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return point["label"], f"exit {rc}", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    points = json.loads(Path(args.spec).read_text())
    if not os.environ.get("CL6_OUT"):
        sys.exit("REFUSED: ROOT_UNSET: set CL6_OUT to the run's own root")
    root = Path(os.environ["CL6_OUT"]).resolve()
    outside = [p["label"] for p in points if root not in Path(p["out"]).resolve().parents]
    if outside:
        sys.exit(f"REFUSED: OUT_OUTSIDE_ROOT: {len(outside)} point(s) outside {root}: {outside[:6]}")
    print(f"[cl6_sweep] {len(points)} points, {args.workers} workers; src {K.src_digest()}; root {root}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for label, status, wall in pool.map(run_one, points):
            print(f"[cl6_sweep] {label}: {status} ({wall:.0f} s)", flush=True)
    bad = [p["label"] for p in points if not done(Path(p["out"]))]
    print(f"[cl6_sweep] done; not ok: {bad}", flush=True)


if __name__ == "__main__":
    main()
