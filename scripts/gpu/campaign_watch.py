"""Wake the operator BEFORE the GPUs go idle (or on drain / stall / failure spike).

Runs as a background task that blocks on the runner's ``status.json`` and EXITS (so
the harness notifies the operator) the moment a decision/refill point is reached —
so the human refills the queue proactively at a low watermark instead of discovering
idle GPUs after the fact. Triggers:

  * refill  — pending backlog dropped below ``--low`` (top the queue up NOW)
  * drained — nothing pending and nothing running (batch finished / advance the plan)
  * stall   — pending work exists but nothing has been running for ``--stall`` s
              (the runner can't place anything — e.g. all pending need a free GPU and
              none is free, or a bug — needs a human look)
  * failures— failed count jumped by >= ``--fail-burst`` since the watch started

Run:  python scripts/gpu/campaign_watch.py --low 8 --stall 300 --fail-burst 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from gpu_queue import campaign_dir


def _read(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def watch(status_path: str, *, low: int, stall: float, fail_burst: int,
          poll: float = 10.0, sleep=time.sleep, clock=time.monotonic) -> dict:
    base_failed = None
    stalled_since = None
    while True:
        s = _read(status_path)
        if s is None:
            sleep(poll)
            continue
        if base_failed is None:
            base_failed = s.get("failed", 0)
        pending, running = s.get("pending", 0), s.get("running", 0)
        failed = s.get("failed", 0)

        if pending == 0 and running == 0:
            return {"trigger": "drained", "status": s}
        if failed - base_failed >= fail_burst:
            return {"trigger": "failures", "delta": failed - base_failed, "status": s}
        if pending > 0 and running == 0:
            stalled_since = stalled_since or clock()
            if clock() - stalled_since >= stall:
                return {"trigger": "stall", "status": s}
        else:
            stalled_since = None
        if pending < low and not (pending == 0 and running == 0):
            return {"trigger": "refill", "pending": pending, "status": s}
        sleep(poll)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--low", type=int, default=8, help="refill watermark (pending)")
    p.add_argument("--stall", type=float, default=300.0)
    p.add_argument("--fail-burst", type=int, default=5)
    p.add_argument("--poll", type=float, default=10.0)
    args = p.parse_args(argv)
    status_path = os.path.join(campaign_dir(), "status.json")
    ev = watch(status_path, low=args.low, stall=args.stall, fail_burst=args.fail_burst,
               poll=args.poll)
    s = ev["status"]
    print(f"WATCH WAKE [{ev['trigger']}]: pending={s.get('pending')} running={s.get('running')} "
          f"done={s.get('done')} failed={s.get('failed')}")
    print(json.dumps(ev, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
