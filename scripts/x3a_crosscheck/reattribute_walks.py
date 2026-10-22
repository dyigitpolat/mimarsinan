"""Re-attribution probe (proof check, part 2): code-level counterexamples under the synchronized LIF count
executor, rerun with every stored-latency walk made a dependency order in-process (ChipLatency's ALAP raise
switched off; no file edit). The count executor is timing-free (``executors/sync_counts.py``: a count
staircase per core for any arrival), so the ablation changes only the walk order: a counterexample it
restores is the latency_order_walk class (R6), whatever layer the screen's ablation named.

Usage: ``python reattribute_walks.py OUT.json SCREEN_DIR`` (reads ``on*.jsonl`` of the code-level screen).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mimarsinan.certification.randomized.runner import run_instance
from mimarsinan.mapping.latency.chip import ChipLatency


def dependency_walk(seed: int, arm: str):
    saved = ChipLatency._align_shiftable_cores
    ChipLatency._align_shiftable_cores = lambda self, current_max_latency: current_max_latency
    try:
        return run_instance(seed, arm)
    finally:
        ChipLatency._align_shiftable_cores = saved


def main(argv) -> int:
    out, screen = Path(argv[0]), Path(argv[1])
    rows = []
    for name in ("on", "on_scheduled", "on_values_classes"):
        for line in (screen / f"{name}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec["outcome"] != "counterexample" or rec["semantics"] != "synchronized_lif":
                continue
            again, walked = run_instance(rec["seed"], name), dependency_walk(rec["seed"], name)
            rows.append({"arm": name, "seed": rec["seed"], "screen_classes": rec["classes"],
                         "rerun": [again.outcome, again.mismatches], "dependency_walk": [walked.outcome, walked.mismatches]})
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
