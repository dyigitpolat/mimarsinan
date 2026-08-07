"""Purge failed/timed-out campaign runs from state and resubmit them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mimarsinan.common.lifecycle.child_launcher import run_child  # noqa: E402

SLURMECH = [str(REPO / "env/bin/python"), "-c", "from slurmech.cli import main; main()"]
RETRYABLE = {"FAILED", "TIMEOUT", "STALE", "CANCELLED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="generated/_campaign/state.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state_path = REPO / args.state
    state = json.loads(state_path.read_text())

    proc = run_child(
        SLURMECH + ["status", "--all", "--json"], cwd=REPO, timeout_s=180,
    )
    states: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        states[row.get("run_id", "")] = row.get("state", "UNKNOWN")

    purged = []
    for name, entry in sorted(state.items()):
        run_state = states.get(entry["run_id"], "UNKNOWN")
        if run_state in RETRYABLE:
            purged.append((name, entry["run_id"], run_state))

    for name, run_id, run_state in purged:
        print(f"resubmit: {name} (was {run_state}, {run_id})")
        if not args.dry_run:
            entry = state.pop(name)
            history = entry.setdefault("previous_attempts", [])
            history.append(run_id)
            (REPO / "generated/_campaign/retry_history.jsonl").open("a").write(
                json.dumps({"name": name, "run_id": run_id, "state": run_state}) + "\n"
            )

    if not purged:
        print("nothing to resubmit")
        return 0
    if args.dry_run:
        return 0

    state_path.write_text(json.dumps(state, indent=1))
    # --only pins resubmission to the purged runs; otherwise submit refills
    # the state with the first N configs of the whole matrix.
    only = [name for name, _, _ in purged]
    resubmit = run_child(
        [sys.executable, str(REPO / "scripts/slurm_campaign.py"), "submit",
         "--wave-size", str(len(purged)), "--time-limit", "00:45:00",
         "--state", args.state, "--only", *only],
        cwd=REPO, timeout_s=60 * len(purged) + 300,
    )
    sys.stdout.write(resubmit.stdout)
    sys.stderr.write(resubmit.stderr)
    return resubmit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
