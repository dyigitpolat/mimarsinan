"""Slurm campaign driver: submit tier configs via slurmech, poll, harvest a results table."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLURMECH = [str(REPO / "env/bin/python"), "-c", "from slurmech.cli import main; main()"]

ACC_GATE = 0.97
WALL_GATE_S = 300.0

_HCM_RE = re.compile(r"Hard-core Spiking Simulation Test: ([0-9.]+)")
_PROFILE_RE = re.compile(r"\[PROFILE\] step='([^']+)' wall=\s*([0-9.]+)s")


def _run(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)


def _slurmech(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    return _run(SLURMECH + list(args), timeout=timeout)


def tier_configs(tier: str, only: list[str] | None = None) -> list[Path]:
    tier_dir = REPO / "test_configs" / f"tier{tier.replace('.', '_')}"
    manifest = json.loads((tier_dir / "manifest.json").read_text())
    rows = manifest["runs"]
    if only:
        rows = [r for r in rows if any(s in r["name"] for s in only)]
    return [tier_dir / row["config"] for row in rows]


def _job_command(config_path: Path) -> str:
    rel = config_path.relative_to(REPO)
    return (
        "export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 "
        "NUMEXPR_NUM_THREADS=20; "
        f"python run.py --headless {rel}"
    )


def submit_wave(configs: list[Path], state_path: Path, time_limit: str) -> dict:
    """Submit one detached job per config; returns {config_name: run_id}."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for config in configs:
        name = config.stem
        if name in state and state[name].get("run_id"):
            print(f"[campaign] {name}: already submitted as {state[name]['run_id']}")
            continue
        proc = _slurmech(
            "run", "--detach", "--time", time_limit, "--", "bash", "-lc",
            _job_command(config),
        )
        run_id = None
        for line in (proc.stdout + proc.stderr).splitlines():
            m = re.search(r"run[_ ]?id[:= ]+\s*([0-9]{8}-[0-9]{6}-[0-9a-f]{8})", line, re.I)
            if m:
                run_id = m.group(1)
        if run_id is None:
            m = re.search(r"([0-9]{8}-[0-9]{6}-[0-9a-f]{8})", proc.stdout + proc.stderr)
            run_id = m.group(1) if m else None
        if run_id is None:
            print(f"[campaign] SUBMIT FAILED for {name}:\n{proc.stdout}\n{proc.stderr}")
            continue
        state[name] = {"run_id": run_id, "config": str(config.relative_to(REPO))}
        state_path.write_text(json.dumps(state, indent=1))
        print(f"[campaign] {name} -> {run_id}")
    return state


def poll_states(state: dict) -> dict[str, str]:
    """{config_name: slurmech state} via `status --json` (falls back to text parse)."""
    proc = _slurmech("status", "--all", "--json")
    by_run: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_run[row.get("run_id", "")] = row.get("state", "UNKNOWN")
    if not by_run:
        for line in proc.stdout.splitlines():
            m = re.match(r"(\S+)\s+job=\S+\s+state=(\S+)", line.strip())
            if m:
                by_run[m.group(1)] = m.group(2)
    return {
        name: by_run.get(entry["run_id"], "UNKNOWN") for name, entry in state.items()
    }


def harvest_one(name: str, entry: dict, out_dir: Path) -> dict:
    """Fetch logs + _GUI_STATE for one finished run and parse the verdict row."""
    run_id = entry["run_id"]
    _slurmech("fetch", run_id)
    _slurmech(
        "fetch", run_id, "--path", "generated/*_deployment_run/_GUI_STATE/*",
    )
    artifacts = Path.home() / ".slurmech/workspaces/mimarsinan-xlog1/runs" / run_id / "artifacts"
    row: dict = {"name": name, "run_id": run_id}

    stdout_log = artifacts / "stdout.log"
    text = stdout_log.read_text(errors="replace") if stdout_log.exists() else ""
    hcm = _HCM_RE.findall(text)
    row["hcm_accuracy"] = float(hcm[-1]) if hcm else None
    row["profile"] = {m.group(1): float(m.group(2)) for m in _PROFILE_RE.finditer(text)}

    infos = list(artifacts.glob("workspace/generated/*_deployment_run/_GUI_STATE/run_info.json"))
    if infos:
        info = json.loads(infos[0].read_text())
        row["status"] = info.get("status")
        started, finished = info.get("started_at"), info.get("finished_at")
        if started and finished:
            row["wall_s"] = round(finished - started, 1)
        row["error"] = info.get("error")
    else:
        row["status"] = "no-run-info"

    exitcode = artifacts / "exitcode"
    row["exitcode"] = exitcode.read_text().strip() if exitcode.exists() else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1))
    return row


def print_table(rows: list[dict]) -> int:
    print(f"\n{'run':52s} {'status':10s} {'acc':>7s} {'wall':>8s}  verdict")
    failed = 0
    for r in sorted(rows, key=lambda r: r["name"]):
        acc = r.get("hcm_accuracy")
        wall = r.get("wall_s")
        acc_ok = acc is not None and acc >= ACC_GATE
        wall_ok = wall is not None and wall <= WALL_GATE_S
        ok = acc_ok and wall_ok and r.get("status") == "completed"
        failed += 0 if ok else 1
        verdict = "PASS" if ok else "FAIL:" + ",".join(
            t for t, bad in (
                ("status", r.get("status") != "completed"),
                ("acc", not acc_ok), ("wall", not wall_ok),
            ) if bad
        )
        acc_s = f"{acc:.4f}" if acc is not None else "-"
        wall_s = f"{wall:.0f}s" if wall is not None else "-"
        print(f"{r['name']:52s} {str(r.get('status')):10s} {acc_s:>7s} {wall_s:>8s}  {verdict}")
    print(f"\n{len(rows) - failed}/{len(rows)} runs pass (gate: acc>={ACC_GATE}, wall<={WALL_GATE_S:.0f}s)")
    return failed


TERMINAL = {"FINISHED", "FAILED", "STALE", "CANCELLED", "TIMEOUT"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "poll", "harvest", "watch"))
    parser.add_argument("--tiers", nargs="*", default=["0", "0.1"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--state", default="generated/_campaign/state.json")
    parser.add_argument("--out", default="generated/_campaign/results")
    parser.add_argument("--time-limit", default="00:30:00")
    parser.add_argument("--wave-size", type=int, default=12)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    args = parser.parse_args()

    state_path = REPO / args.state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / args.out

    configs = [c for tier in args.tiers for c in tier_configs(tier, args.only)]

    if args.action == "submit":
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        pending = [c for c in configs if c.stem not in state]
        wave = pending[: args.wave_size]
        submit_wave(wave, state_path, args.time_limit)
        return 0

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.action == "poll":
        for name, st in sorted(poll_states(state).items()):
            print(f"{name:52s} {st}")
        return 0

    if args.action == "harvest":
        rows = [harvest_one(n, e, out_dir) for n, e in state.items()]
        return 1 if print_table(rows) else 0

    if args.action == "watch":
        while True:
            states = poll_states(state)
            live = {n: s for n, s in states.items() if s not in TERMINAL}
            print(f"[campaign] {len(states) - len(live)}/{len(states)} terminal; live: "
                  + ", ".join(f"{n}={s}" for n, s in sorted(live.items())[:8]))
            if not live:
                break
            time.sleep(args.poll_interval)
        rows = [harvest_one(n, e, out_dir) for n, e in state.items()]
        return 1 if print_table(rows) else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
