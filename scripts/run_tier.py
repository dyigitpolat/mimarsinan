"""Run a test_configs tier: each config headlessly, with wall budget and a result table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _working_dir(config: dict) -> Path:
    name = config["experiment_name"]
    mode = config.get("pipeline_mode", "phased")
    return REPO / config["generated_files_path"] / f"{name}_{mode}_deployment_run"


def _run_status(workdir: Path) -> str:
    info = workdir / "_GUI_STATE" / "run_info.json"
    if not info.exists():
        return "no-run-info"
    return json.loads(info.read_text()).get("status", "unknown")


def _final_metrics(workdir: Path) -> dict:
    metrics_path = workdir / "_GUI_STATE" / "live_metrics.jsonl"
    latest: dict = {}
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("name") or row.get("metric")
            if name is not None and "value" in row:
                latest[name] = row["value"]
    return latest


def run_one(config_path: Path, budget_s: float) -> dict:
    config = json.loads(config_path.read_text())
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "run.py", "--headless", str(config_path)],
            cwd=REPO, timeout=budget_s, capture_output=True, text=True,
        )
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, timed_out = -1, True
    wall = time.monotonic() - started
    workdir = _working_dir(config)
    return {
        "name": config["experiment_name"],
        "rc": rc,
        "timed_out": timed_out,
        "wall_s": round(wall, 1),
        "status": _run_status(workdir),
        "metrics": _final_metrics(workdir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", type=int, choices=(0, 1, 2))
    parser.add_argument("--only", nargs="*", help="run-name substrings to filter on")
    parser.add_argument("--budget-scale", type=float, default=1.5,
                        help="wall budget = manifest expected_wall_min × this")
    args = parser.parse_args()

    manifest = json.loads(
        (REPO / "test_configs" / f"tier{args.tier}" / "manifest.json").read_text()
    )
    rows = manifest["runs"]
    if args.only:
        rows = [r for r in rows if any(s in r["name"] for s in args.only)]

    results = []
    for row in rows:
        config_path = REPO / "test_configs" / f"tier{args.tier}" / row["config"]
        budget_s = row["expected_wall_min"] * 60 * args.budget_scale
        print(f"[run_tier] {row['name']} (budget {budget_s / 60:.0f} min)", flush=True)
        results.append(run_one(config_path, budget_s))

    print(f"\n{'run':55s} {'status':10s} {'rc':>3s} {'wall':>8s}  key metrics")
    failed = 0
    for r in results:
        ok = r["status"] == "completed" and r["rc"] == 0
        failed += 0 if ok else 1
        picks = {k: v for k, v in r["metrics"].items()
                 if any(t in str(k).lower() for t in ("accuracy", "parity"))}
        summary = ", ".join(f"{k}={v}" for k, v in sorted(picks.items())[:4])
        flag = "" if ok else "  <-- FAIL"
        print(f"{r['name']:55s} {r['status']:10s} {r['rc']:>3d} {r['wall_s']:>7.1f}s  {summary}{flag}")

    print(f"\n{len(results) - failed}/{len(results)} runs completed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
