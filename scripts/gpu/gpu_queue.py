"""A persistent, crash-safe filesystem work queue for the autonomous GPU runner.

The v1 dispatcher drained a FIXED manifest and then EXITED — so the GPUs went idle
the moment the batch finished and a human had to notice. This queue decouples
*producing* work (any number of producers drop job specs in) from *running* it (the
long-lived `campaign_runner` drains it forever), so producers can keep the backlog
full and the runner never starves.

Layout (under ``$MIM_CAMPAIGN_DIR`` or ``runs/campaign``):
  q/pending/  q/running/  q/done/  q/failed/   — a job moves pending→running→done|failed
  q/STOP                                        — sentinel: runner drains running, then exits
Each job is one JSON file; state transitions are atomic ``os.rename`` (the rename IS
the claim, so two runners never double-run a job). Jobs are ordered by
``priority`` then enqueue time, so cheap-diagnostic rungs run before expensive ones.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


def campaign_dir() -> str:
    d = os.environ.get("MIM_CAMPAIGN_DIR")
    if not d:
        d = os.path.join(os.getcwd(), "runs", "campaign")
    return d


class GpuQueue:
    def __init__(self, root: Optional[str] = None):
        self.root = root or os.path.join(campaign_dir(), "q")
        for sub in (PENDING, RUNNING, DONE, FAILED):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    # -- paths --
    def _dir(self, state: str) -> str:
        return os.path.join(self.root, state)

    @property
    def stop_path(self) -> str:
        return os.path.join(self.root, "STOP")

    def stop_requested(self) -> bool:
        return os.path.exists(self.stop_path)

    @property
    def pause_path(self) -> str:
        return os.path.join(self.root, "PAUSE")

    def pause_requested(self) -> bool:
        # When present, the runner keeps reaping finished jobs but launches NO new
        # ones — so the operator can do git integration on the runner's checkout
        # without a half-written/conflict-marked .py corrupting an in-flight job.
        return os.path.exists(self.pause_path)

    def request_stop(self) -> None:
        open(self.stop_path, "w").close()

    # -- producer side --
    def enqueue(self, spec: Dict[str, Any]) -> str:
        """Add a job. spec: {cmd:[...]|str, mode:'free'|'fit', need_mb, timeout_s,
        priority, cwd, env, tags}. Returns the job id."""
        jid = spec.get("id") or f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        job = dict(spec)
        job.setdefault("id", jid)
        job.setdefault("mode", "fit")
        job.setdefault("need_mb", 8000)
        job.setdefault("priority", 50)
        job["enqueued_at"] = time.time()
        name = f"{int(job['priority']):03d}_{int(job['enqueued_at']*1000):015d}_{jid}.json"
        path = os.path.join(self._dir(PENDING), name)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(job, fh)
        os.replace(tmp, path)
        return jid

    # -- consumer side (runner) --
    def pending_names(self) -> List[str]:
        return sorted(n for n in os.listdir(self._dir(PENDING)) if n.endswith(".json"))

    def claim_next(self) -> Optional[tuple]:
        """Atomically move the highest-priority pending job to running/.
        Returns (job_dict, running_path) or None if nothing claimable."""
        for name in self.pending_names():
            src = os.path.join(self._dir(PENDING), name)
            dst = os.path.join(self._dir(RUNNING), name)
            try:
                os.rename(src, dst)           # atomic claim; loser raises and we skip
            except OSError:
                continue
            try:
                with open(dst) as fh:
                    return json.load(fh), dst
            except (OSError, ValueError):
                os.rename(dst, os.path.join(self._dir(FAILED), name))
        return None

    def finish(self, running_path: str, result: Dict[str, Any], success: bool) -> None:
        name = os.path.basename(running_path)
        try:
            with open(running_path) as fh:
                job = json.load(fh)
        except (OSError, ValueError):
            job = {"id": name}
        job["result"] = result
        dest_dir = self._dir(DONE if success else FAILED)
        dst = os.path.join(dest_dir, name)
        with open(dst, "w") as fh:
            json.dump(job, fh, indent=2)
        try:
            os.remove(running_path)
        except OSError:
            pass

    def counts(self) -> Dict[str, int]:
        return {s: len([n for n in os.listdir(self._dir(s)) if n.endswith(".json")])
                for s in (PENDING, RUNNING, DONE, FAILED)}

    def list_state(self, state: str) -> List[Dict[str, Any]]:
        out = []
        for n in sorted(os.listdir(self._dir(state))):
            if not n.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._dir(state), n)) as fh:
                    out.append(json.load(fh))
            except (OSError, ValueError):
                pass
        return out

    def requeue_running(self) -> int:
        """On runner startup, move orphaned running/ jobs (from a crashed runner)
        back to pending/ so they re-run. Returns how many."""
        n = 0
        for name in os.listdir(self._dir(RUNNING)):
            if name.endswith(".json"):
                os.rename(os.path.join(self._dir(RUNNING), name),
                          os.path.join(self._dir(PENDING), name))
                n += 1
        return n
