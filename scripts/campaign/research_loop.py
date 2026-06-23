"""Research-loop primitives so a dynamic workflow can run design→enqueue→wait→analyze
against the shared autonomous GPU runner.

A research cluster (one workflow) repeatedly: designs a small kill-gated batch, ENQUEUEs
it, BLOCKS until those jobs finalize, then ANALYZEs the deployed metrics and decides the
next round. These CLIs are the enqueue/wait/ledger verbs its agents call.

  enqueue   python scripts/campaign/research_loop.py enqueue <manifest.json>
  wait      python scripts/campaign/research_loop.py wait <id> [<id> ...] [--timeout S]
  results   python scripts/campaign/research_loop.py results <id> [<id> ...]
  ledger    python scripts/campaign/research_loop.py ledger-append <record.json>
            python scripts/campaign/research_loop.py ledger-read [--cluster WS3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "gpu"))
from gpu_queue import GpuQueue, campaign_dir  # noqa: E402

LEDGER = os.path.join(campaign_dir(), "ledger.jsonl")


def _deployed_acc(job: dict) -> Optional[float]:
    art = job.get("expect_artifact")
    path = art if (art and os.path.isabs(art)) else os.path.join(job.get("cwd", REPO), art or "")
    if art and os.path.isfile(path):
        try:
            return float(json.load(open(path)))
        except (ValueError, OSError):
            return None
    return None


def _index(q: GpuQueue) -> Dict[str, dict]:
    """id -> {state, returncode, deployed_acc, tags} for everything finalized."""
    out = {}
    for state in ("done", "failed"):
        for job in q.list_state(state):
            r = job.get("result", {})
            out[job["id"]] = {"state": state, "returncode": r.get("returncode"),
                              "deployed_acc": _deployed_acc(job), "tags": job.get("tags", {}),
                              "wall_s": r.get("wall_s")}
    return out


def cmd_enqueue(args) -> int:
    q = GpuQueue()
    jobs = json.load(open(args.manifest))
    ids = [q.enqueue(j) for j in jobs]
    print(json.dumps({"enqueued": len(ids), "ids": ids}))
    return 0


def cmd_wait(args) -> int:
    q = GpuQueue()
    targets = set(args.ids)
    deadline = time.time() + args.timeout
    while True:
        idx = _index(q)
        have = targets & set(idx)
        if have == targets:
            print(json.dumps({"all_done": True, "results": {i: idx[i] for i in targets}}, indent=2))
            return 0
        if time.time() > deadline:
            print(json.dumps({"all_done": False, "timed_out": True,
                              "missing": sorted(targets - have),
                              "results": {i: idx[i] for i in have}}, indent=2))
            return 1
        time.sleep(args.poll)


def cmd_results(args) -> int:
    idx = _index(GpuQueue())
    print(json.dumps({i: idx.get(i, {"state": "pending"}) for i in args.ids}, indent=2))
    return 0


def cmd_ledger_append(args) -> int:
    rec = json.load(open(args.record)) if os.path.isfile(args.record) else json.loads(args.record)
    recs = rec if isinstance(rec, list) else [rec]
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as fh:
        for r in recs:
            r.setdefault("ts", time.time())
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"appended": len(recs)}))
    return 0


def cmd_ledger_read(args) -> int:
    rows = []
    if os.path.isfile(LEDGER):
        for line in open(LEDGER):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if args.cluster and r.get("cluster") != args.cluster:
                continue
            rows.append(r)
    print(json.dumps(rows, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enqueue"); e.add_argument("manifest"); e.set_defaults(fn=cmd_enqueue)
    w = sub.add_parser("wait"); w.add_argument("ids", nargs="+")
    w.add_argument("--timeout", type=float, default=14400); w.add_argument("--poll", type=float, default=20)
    w.set_defaults(fn=cmd_wait)
    r = sub.add_parser("results"); r.add_argument("ids", nargs="+"); r.set_defaults(fn=cmd_results)
    la = sub.add_parser("ledger-append"); la.add_argument("record"); la.set_defaults(fn=cmd_ledger_append)
    lr = sub.add_parser("ledger-read"); lr.add_argument("--cluster", default=None); lr.set_defaults(fn=cmd_ledger_read)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
