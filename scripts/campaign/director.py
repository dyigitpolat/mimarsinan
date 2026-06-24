"""The persistent research DIRECTOR — keeps the campaign advancing unattended.

Two jobs, both mechanical (no LLM, no scientific judgment):

1. PLAN ROLLOUT — when enabled work runs low, enable the next disabled "plan" batch
   (lowest ``plan_stage`` first). Pre-load a deep plan of disabled batches and the
   director rolls them out over hours/days, so GPUs never starve waiting for a human
   to append experiments. This is the anti-stall guarantee beyond the current backlog.

2. HARVEST TODO — surface finalized-but-unanalyzed cells into ``harvest_todo.json``
   so the consolidation step knows exactly what science is waiting. The director does
   NOT write ledger verdicts: a verdict needs judgment (ANN reference, confound
   flagging) and stays with the harvest workflow. The director only routes work and
   flags the gap.

Layers: scheduler FILLS the queue · runner DRAINS it · director PLANS + FLAGS ·
harvest CONSOLIDATES. Together they run the campaign with no human in the refill loop.

Run:  python scripts/campaign/director.py [--lo 8] [--poll 30]
Stop: touch runs/campaign/q/DIRECTOR_STOP  (or --stop)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "gpu"))
sys.path.insert(0, os.path.join(REPO, "scripts", "campaign"))
from gpu_queue import GpuQueue, campaign_dir  # noqa: E402
import scheduler as sch  # noqa: E402  (reuse instantiate / existing_ids / BACKLOG)

LEDGER = os.path.join(campaign_dir(), "ledger.jsonl")
TODO = os.path.join(campaign_dir(), "harvest_todo.json")
DIRECTOR_STATUS = os.path.join(campaign_dir(), "director_status.json")


def _batch_ids(batch: dict) -> List[str]:
    return [jid for jid, _ in sch.instantiate(batch)]


def enabled_remaining(backlog: List[dict], existing: set) -> int:
    """Un-enqueued job points across ENABLED batches — the fuel left for the scheduler."""
    n = 0
    for b in backlog:
        if not b.get("enabled", True):
            continue
        n += sum(1 for jid in _batch_ids(b) if jid not in existing)
    return n


def _next_plan_batch(backlog: List[dict]) -> Optional[dict]:
    """The lowest-``plan_stage`` disabled batch waiting to be rolled out."""
    waiting = [b for b in backlog if not b.get("enabled", True) and "plan_stage" in b]
    return min(waiting, key=lambda b: b["plan_stage"]) if waiting else None


def _cell_key(tags: dict) -> Optional[str]:
    model, dataset, sched = tags.get("model"), tags.get("dataset"), tags.get("sched")
    if not (model and dataset and sched):
        return None
    parts = [str(model), str(dataset), str(sched)]
    if tags.get("depth") is not None:
        parts.append(f"d{tags['depth']}")
    return "/".join(parts)


def _ledger_cited_run_ids(ledger_path: str) -> set:
    cited = set()
    if not os.path.isfile(ledger_path):
        return cited
    for line in open(ledger_path):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for k, v in row.items():
            if k.endswith("run_ids") and isinstance(v, list):
                cited.update(v)
    return cited


def harvest_todo(q: GpuQueue, ledger_path: str = LEDGER) -> List[dict]:
    """Finalized cells whose runs are NOT yet cited in any ledger row.

    A cell is "covered" if ANY of its finalized runs is already cited (the harvest
    row references that cell), so a consolidated cell is never re-flagged.
    """
    cited = _ledger_cited_run_ids(ledger_path)
    cells: dict = {}
    for state in ("done", "failed"):
        for job in q.list_state(state):
            key = _cell_key(job.get("tags", {}))
            if key is None:
                continue
            cells.setdefault(key, []).append(job["id"])
    todo = []
    for key, run_ids in sorted(cells.items()):
        if any(rid in cited for rid in run_ids):
            continue
        todo.append({"cell": key, "run_ids": sorted(run_ids), "n_finalized": len(run_ids)})
    return todo


class Director:
    def __init__(self, q: GpuQueue, *, lo: int = 8, poll: float = 30.0,
                 backlog_path: str = sch.BACKLOG, ledger_path: str = LEDGER,
                 todo_path: str = TODO, status_path: str = DIRECTOR_STATUS):
        self.q = q
        self.lo = lo
        self.poll = poll
        self.backlog_path = backlog_path
        self.ledger_path = ledger_path
        self.todo_path = todo_path
        self.status_path = status_path
        self.stop_path = os.path.join(q.root, "DIRECTOR_STOP")

    def _load_backlog(self) -> List[dict]:
        try:
            return json.load(open(self.backlog_path))
        except (OSError, ValueError):
            return []

    def _save_backlog(self, backlog: List[dict]) -> None:
        tmp = self.backlog_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(backlog, fh, indent=2)
        os.replace(tmp, self.backlog_path)

    def _write_json(self, path: str, obj) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)

    def tick(self) -> dict:
        backlog = self._load_backlog()
        existing = sch.existing_ids(self.q)
        remaining = enabled_remaining(backlog, existing)
        pending = self.q.counts()["pending"]
        fuel = remaining + pending

        enabled_plan_batch = None
        if fuel < self.lo:
            nxt = _next_plan_batch(backlog)
            if nxt is not None:
                nxt["enabled"] = True
                self._save_backlog(backlog)
                enabled_plan_batch = nxt["id"]

        todo = harvest_todo(self.q, self.ledger_path)
        self._write_json(self.todo_path, todo)
        stages_left = sum(1 for b in backlog if not b.get("enabled", True) and "plan_stage" in b)
        status = {"ts": time.time(), "fuel": fuel, "enabled_remaining": remaining,
                  "pending": pending, "enabled_plan_batch": enabled_plan_batch,
                  "n_harvest_todo": len(todo), "plan_stages_left": stages_left}
        self._write_json(self.status_path, status)
        return status

    def run(self):
        print(f"director: lo={self.lo} backlog={self.backlog_path} (rolls out plan_stage batches, flags harvest)")
        while not os.path.exists(self.stop_path):
            res = self.tick()
            if res["enabled_plan_batch"]:
                print(f"director: rolled out plan batch {res['enabled_plan_batch']} (fuel was {res['fuel']})")
            time.sleep(self.poll)
        print("director: stopped.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lo", type=int, default=8)
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--stop", action="store_true")
    args = p.parse_args(argv)
    q = GpuQueue()
    sp = os.path.join(q.root, "DIRECTOR_STOP")
    if args.stop:
        open(sp, "w").close()
        print("director stop requested.")
        return 0
    if os.path.exists(sp):
        os.remove(sp)
    Director(q, lo=args.lo, poll=args.poll).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
