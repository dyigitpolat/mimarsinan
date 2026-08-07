"""Run a templates tier: each config headlessly, with wall budget and a result table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mimarsinan.common.lifecycle.child_launcher import run_child  # noqa: E402


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
    result = run_child(
        [sys.executable, "run.py", "--headless", str(config_path)],
        cwd=REPO, timeout_s=budget_s,
    )
    workdir = _working_dir(config)
    return {
        "name": config["experiment_name"],
        "rc": result.returncode,
        "timed_out": result.timed_out,
        "wall_s": round(result.wall_s, 1),
        "status": _run_status(workdir),
        "metrics": _final_metrics(workdir),
        "stderr_tail": result.stderr_tail(),
        "cohort_gone": result.session_gone,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", type=lambda t: t.replace(".", "_"),
                        choices=("0", "0_1", "1", "2"),
                        help="tier directory suffix; 0.1 and 0_1 both address tier0_1")
    parser.add_argument("--only", nargs="*", help="run-name substrings to filter on")
    parser.add_argument("--budget-scale", type=float, default=1.5,
                        help="wall budget = manifest expected_wall_min × this")
    args = parser.parse_args()

    manifest = json.loads(
        (REPO / "templates" / f"tier_{args.tier}" / "manifest.json").read_text()
    )
    rows = manifest["runs"]
    if args.only:
        rows = [r for r in rows if any(s in r["name"] for s in args.only)]

    results = []
    for row in rows:
        config_path = REPO / "templates" / f"tier_{args.tier}" / row["config"]
        budget_s = row["expected_wall_min"] * 60 * args.budget_scale
        print(f"[run_tier] {row['name']} (budget {budget_s / 60:.0f} min)", flush=True)
        results.append(run_one(config_path, budget_s))

    print(f"\n{'run':55s} {'status':10s} {'rc':>3s} {'wall':>8s}  key metrics")
    failed = []
    for r in results:
        ok = r["status"] == "completed" and r["rc"] == 0
        if not ok:
            failed.append(r)
        picks = {k: v for k, v in r["metrics"].items()
                 if any(t in str(k).lower() for t in ("accuracy", "parity"))}
        summary = ", ".join(f"{k}={v}" for k, v in sorted(picks.items())[:4])
        flag = "" if ok else "  <-- FAIL"
        print(f"{r['name']:55s} {r['status']:10s} {r['rc']:>3d} {r['wall_s']:>7.1f}s  {summary}{flag}")

    for r in failed:
        print(f"\n--- {r['name']}: rc={r['rc']} timed_out={r['timed_out']} ---")
        print(r["stderr_tail"] or "(no stderr captured)")
    for r in results:
        if not r["cohort_gone"]:
            print(f"[run_tier] WARNING: {r['name']} left live processes behind")

    print(f"\n{len(results) - len(failed)}/{len(results)} runs completed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
