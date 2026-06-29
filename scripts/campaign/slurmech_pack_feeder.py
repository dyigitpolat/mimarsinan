"""Drive a multi-node slurmech campaign by chunking a queue manifest into packs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SLURMECH_BIN = "env/bin/slurmech"
PYTHONPATH_PREFIX = "export PYTHONPATH=src:spikingjelly:sana_fe:${PYTHONPATH:-};"
DEFAULT_PACKS_DIR = os.path.join(_REPO, "runs", "campaign", "slurmech_packs_v2")
DEFAULT_NODES = 6
DEFAULT_MAX_PARALLELISM = 8
DEFAULT_GPU_MEM_MB = 81000
DEFAULT_PER_JOB_MB = 8000
DEFAULT_POLL_INTERVAL_S = 30.0
# Matches .slurmech.toml [slurm] cpus_per_gpu — the per-node core budget the
# co-located pack jobs share.
DEFAULT_CPUS_PER_NODE = 40
# The BLAS/OpenMP pools each headless job would otherwise size to the whole node;
# capping them keeps N concurrent pack jobs off each other's cores.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_ACTIVE_STATES = ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING")


def threads_per_job(cpus_per_node: int, parallelism: int) -> int:
    """Even split of a node's cores across its concurrent pack jobs (floor, min 1)."""
    if parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism}")
    return max(1, cpus_per_node // parallelism)


def thread_cap_prefix(thread_cap: Optional[int]) -> str:
    """``export OMP_NUM_THREADS=N ...;`` capping the math-library pools, or ``""``.

    ``None`` leaves thread pools at their library defaults (no prefix); a value
    pins every pool in :data:`THREAD_ENV_VARS` so co-located jobs don't
    oversubscribe a shared node.
    """
    if thread_cap is None:
        return ""
    if thread_cap < 1:
        raise ValueError(f"thread_cap must be >= 1, got {thread_cap}")
    assignments = " ".join(f"{var}={thread_cap}" for var in THREAD_ENV_VARS)
    return f"export {assignments};"


def config_path_for_job(job: dict) -> str:
    """The job's headless config path: honor an explicit manifest cmd, else flat default.

    Manifest jobs may carry their own ``cmd`` (e.g. produced with a non-default
    ``--config-dir`` subdirectory). The path after ``--headless`` is authoritative;
    only when absent do we reconstruct the canonical flat ``experiments/campaign/<id>.json``.
    """
    cmd = job.get("cmd")
    if cmd:
        tokens = list(cmd)
        if "--headless" in tokens:
            idx = tokens.index("--headless")
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
    return f"experiments/campaign/{job['id']}.json"


def normalize_job_cmd(job: dict, thread_cap: Optional[int] = None) -> str:
    """Reconstruct the canonical headless run command for a manifest job."""
    config = config_path_for_job(job)
    lead = thread_cap_prefix(thread_cap)
    lead = f"{lead} " if lead else ""
    return f"{lead}{PYTHONPATH_PREFIX} python run.py --headless {config}"


def chunk_manifest(jobs: Sequence[dict], n_packs: int) -> list[list[dict]]:
    """Split jobs into at most n_packs contiguous, size-balanced, non-empty chunks."""
    if n_packs < 1:
        raise ValueError(f"n_packs must be >= 1, got {n_packs}")
    n = len(jobs)
    if n == 0:
        return []
    k = min(n_packs, n)
    base, rem = divmod(n, k)
    chunks: list[list[dict]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append(list(jobs[start:start + size]))
        start += size
    return chunks


def default_parallelism(
    n_jobs_in_pack: int,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    per_job_mb: int = DEFAULT_PER_JOB_MB,
    cap: int = DEFAULT_MAX_PARALLELISM,
) -> int:
    """Memory-safe concurrency: min(cap, floor(gpu_mem/per_job), n_jobs), at least 1."""
    if per_job_mb <= 0:
        raise ValueError(f"per_job_mb must be > 0, got {per_job_mb}")
    mem_limit = gpu_mem_mb // per_job_mb
    return max(1, min(cap, mem_limit, n_jobs_in_pack))


def build_pack_yaml(
    jobs_chunk: Sequence[dict],
    parallelism: int,
    thread_cap: Optional[int] = None,
) -> dict[str, Any]:
    """Return the slurmech pack YAML dict for one chunk of manifest jobs."""
    return {
        "parallelism": int(parallelism),
        "fail_fast": False,
        "kill_on_failure": False,
        "jobs": [
            {"name": j["id"], "cmd": normalize_job_cmd(j, thread_cap)}
            for j in jobs_chunk
        ],
    }


def write_pack_yaml(pack_dict: dict[str, Any], path: str | os.PathLike) -> str:
    """Write a pack YAML dict to path, creating parent dirs. Returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as fh:
        yaml.safe_dump(pack_dict, fh, sort_keys=False, default_flow_style=False)
    return str(target)


def pack_command(pack_path: str, parallelism: int, slurmech_bin: str = SLURMECH_BIN) -> list[str]:
    """The exact slurmech invocation that submits one detached pack."""
    return [slurmech_bin, "pack", str(pack_path), "--parallelism", str(parallelism), "--detach"]


def pack_prefix_from_manifest(manifest_path: str) -> str:
    """Derive a pack-name prefix from a manifest filename (drops queue/manifest suffixes)."""
    stem = Path(manifest_path).stem
    for suffix in ("_queue_manifest", "_manifest", "_queue"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "campaign"


@dataclass
class PackPlan:
    """One pack: its name, on-disk YAML path, jobs, parallelism and thread cap."""

    index: int
    name: str
    path: str
    jobs: list[dict]
    parallelism: int
    thread_cap: Optional[int] = None

    @property
    def command(self) -> list[str]:
        return pack_command(self.path, self.parallelism)


def plan_packs(
    jobs: Sequence[dict],
    n_packs: int,
    packs_dir: str,
    *,
    pack_prefix: str = "campaign",
    max_parallelism: int = DEFAULT_MAX_PARALLELISM,
    gpu_mem_mb: int = DEFAULT_GPU_MEM_MB,
    per_job_mb: int = DEFAULT_PER_JOB_MB,
    cpus_per_node: Optional[int] = DEFAULT_CPUS_PER_NODE,
) -> list[PackPlan]:
    """Chunk jobs and build a PackPlan (path + parallelism + thread cap) per chunk.

    ``cpus_per_node`` (``None`` to leave thread pools at library defaults) is
    split evenly across each pack's concurrent jobs so co-located runs don't
    oversubscribe the node's cores.
    """
    plans: list[PackPlan] = []
    for i, chunk in enumerate(chunk_manifest(jobs, n_packs), start=1):
        parallelism = default_parallelism(len(chunk), gpu_mem_mb, per_job_mb, cap=max_parallelism)
        thread_cap = (
            None if cpus_per_node is None
            else threads_per_job(cpus_per_node, parallelism)
        )
        name = f"{pack_prefix}_pack_{i}"
        path = os.path.join(packs_dir, f"{name}.yaml")
        plans.append(PackPlan(
            index=i, name=name, path=path, jobs=list(chunk),
            parallelism=parallelism, thread_cap=thread_cap,
        ))
    return plans


def parse_active_count(status_text: str) -> int:
    """Count active slurmech allocations from `slurmech status` output (state-line heuristic)."""
    count = 0
    for line in status_text.splitlines():
        upper = line.upper()
        if any(state in upper for state in _ACTIVE_STATES):
            count += 1
    return count


def default_runner(cmd: Sequence[str]) -> None:
    """Submit a pack for real via subprocess (used only off the dry-run path)."""
    subprocess.run(list(cmd), check=True, cwd=_REPO)


def default_status_fn() -> str:
    """Fetch `slurmech status` output for allocation accounting."""
    result = subprocess.run(
        [SLURMECH_BIN, "status"], cwd=_REPO, capture_output=True, text=True, check=False
    )
    return result.stdout + result.stderr


def _submit_plan(
    plan: PackPlan,
    *,
    runner: Optional[Callable[[Sequence[str]], None]],
    dry_run: bool,
    writer: Callable[[dict, str], str],
    logger: Callable[[str], None],
) -> list[str]:
    """Write a plan's YAML then submit it (or only print, under dry_run). Returns the command."""
    writer(build_pack_yaml(plan.jobs, plan.parallelism, plan.thread_cap), plan.path)
    cmd = plan.command
    if dry_run:
        logger("[dry-run] would submit: " + " ".join(cmd))
    else:
        logger("[submit] " + " ".join(cmd))
        if runner is None:
            raise ValueError("runner is required when dry_run is False")
        runner(cmd)
    return cmd


def submit_packs(
    plans: Sequence[PackPlan],
    *,
    runner: Optional[Callable[[Sequence[str]], None]] = None,
    dry_run: bool = False,
    writer: Callable[[dict, str], str] = write_pack_yaml,
    logger: Callable[[str], None] = print,
) -> list[PackPlan]:
    """Write every plan's YAML and submit it once (no concurrency throttle)."""
    for plan in plans:
        _submit_plan(plan, runner=runner, dry_run=dry_run, writer=writer, logger=logger)
    return list(plans)


def feed_packs(
    plans: Sequence[PackPlan],
    *,
    n_nodes: int = DEFAULT_NODES,
    runner: Optional[Callable[[Sequence[str]], None]] = None,
    status_fn: Callable[[], str] = default_status_fn,
    active_count_fn: Optional[Callable[[str], int]] = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    dry_run: bool = False,
    writer: Callable[[dict, str], str] = write_pack_yaml,
    logger: Callable[[str], None] = print,
) -> list[PackPlan]:
    """Submit up to n_nodes packs, then feed the rest as allocations free below n_nodes.

    Under dry_run no status is polled: every YAML is written and its command printed.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    count_active = active_count_fn if active_count_fn is not None else parse_active_count
    pending = list(plans)
    submitted: list[PackPlan] = []
    if dry_run:
        return submit_packs(pending, runner=runner, dry_run=True, writer=writer, logger=logger)
    while pending:
        active = count_active(status_fn())
        free = n_nodes - active
        while free > 0 and pending:
            plan = pending.pop(0)
            _submit_plan(plan, runner=runner, dry_run=False, writer=writer, logger=logger)
            submitted.append(plan)
            free -= 1
        if pending:
            logger(f"[feed] {len(submitted)} submitted, {len(pending)} pending, "
                   f"{active} active; waiting {poll_interval}s")
            sleep_fn(poll_interval)
    return submitted


def load_manifest(path: str) -> list[dict]:
    """Load a queue manifest (a JSON list of job dicts)."""
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"manifest {path} must be a JSON list, got {type(data).__name__}")
    return data


def summarize_plans(plans: Sequence[PackPlan]) -> dict[str, Any]:
    """A JSON-able summary: pack count and per-pack job count / parallelism / command."""
    return {
        "n_packs": len(plans),
        "total_jobs": sum(len(p.jobs) for p in plans),
        "packs": [
            {
                "name": p.name,
                "path": p.path,
                "n_jobs": len(p.jobs),
                "parallelism": p.parallelism,
                "thread_cap": p.thread_cap,
                "command": " ".join(p.command),
            }
            for p in plans
        ],
    }


def _print_summary(plans: Sequence[PackPlan], *, nodes: int, dry_run: bool) -> None:
    summary = summarize_plans(plans)
    header = "DRY-RUN" if dry_run else "PLAN"
    print(f"[{header}] {summary['n_packs']} packs, {summary['total_jobs']} jobs, "
          f"up to {nodes} nodes concurrent")
    for pack in summary["packs"]:
        print(f"  {pack['name']}: {pack['n_jobs']} jobs, "
              f"parallelism={pack['parallelism']}, threads/job={pack['thread_cap']}")
        print(f"    $ {pack['command']}")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Optional[Callable[[Sequence[str]], None]] = None,
    status_fn: Optional[Callable[[], str]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="queue manifest JSON (a list of job dicts)")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES,
                        help="max concurrent slurmech allocations (one node each)")
    parser.add_argument("--packs", type=int, default=None,
                        help="number of packs to chunk into (default: --nodes)")
    parser.add_argument("--max-parallelism", type=int, default=DEFAULT_MAX_PARALLELISM,
                        help="cap on per-pack concurrent child processes")
    parser.add_argument("--packs-dir", default=DEFAULT_PACKS_DIR)
    parser.add_argument("--per-job-mb", type=int, default=DEFAULT_PER_JOB_MB)
    parser.add_argument("--gpu-mem-mb", type=int, default=DEFAULT_GPU_MEM_MB)
    parser.add_argument("--cpus-per-node", type=int, default=DEFAULT_CPUS_PER_NODE,
                        help="node cores split across each pack's jobs as a per-job "
                             "thread cap (0 disables capping)")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--keep-feeding", action="store_true",
                        help="feed packs as allocations free below --nodes")
    parser.add_argument("--dry-run", action="store_true",
                        help="write YAMLs and print commands without submitting")
    args = parser.parse_args(argv)

    jobs = load_manifest(args.manifest)
    n_packs = args.packs if args.packs is not None else args.nodes
    plans = plan_packs(
        jobs,
        n_packs,
        args.packs_dir,
        pack_prefix=pack_prefix_from_manifest(args.manifest),
        max_parallelism=args.max_parallelism,
        gpu_mem_mb=args.gpu_mem_mb,
        per_job_mb=args.per_job_mb,
        cpus_per_node=args.cpus_per_node if args.cpus_per_node > 0 else None,
    )

    _print_summary(plans, nodes=args.nodes, dry_run=args.dry_run)

    if args.dry_run:
        for plan in plans:
            write_pack_yaml(
                build_pack_yaml(plan.jobs, plan.parallelism, plan.thread_cap),
                plan.path,
            )
        return 0

    run = runner if runner is not None else default_runner
    status = status_fn if status_fn is not None else default_status_fn
    if args.keep_feeding:
        feed_packs(
            plans,
            n_nodes=args.nodes,
            runner=run,
            status_fn=status,
            poll_interval=args.poll_interval,
            sleep_fn=sleep_fn,
        )
    else:
        submit_packs(plans, runner=run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
