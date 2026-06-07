"""Run the 9-config verification matrix and tabulate per-step metrics.

Usage:
    python scripts/run_verification_matrix.py [--parallel N] [--filter SUBSTR]

Launches each template config headlessly (run.py --headless), waits, then
prints a table of step metrics with the >=0.97 simulation gate and the
fine-tuning gates highlighted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TEMPLATES = [
    "templates/mnist_mmixcore_lif_60.json",
    "templates/mnist_mmixcore_lif_60_offload.json",
    "templates/mnist_mmixcore_ttfs_60.json",
    "templates/mnist_mmixcore_ttfs_60_offload.json",
    "templates/mnist_mmixcore_ttfs_q_30.json",
    "templates/mnist_mmixcore_ttfs_q_30_offload.json",
    "templates/mnist_mmixcore_ttfs_cycle_60.json",
    "templates/mnist_mmixcore_ttfs_cycle_60_offload.json",
    "templates/regression.json",
]

SIM_GATE = 0.97


def launch(template: str, log_dir: str) -> subprocess.Popen:
    name = os.path.splitext(os.path.basename(template))[0]
    log = open(os.path.join(log_dir, f"{name}.log"), "w")
    return subprocess.Popen(
        [sys.executable, "run.py", "--headless", template],
        stdout=log, stderr=subprocess.STDOUT,
    )


def _run_started_at(run_dir: str) -> float:
    try:
        info = json.load(open(os.path.join(run_dir, "_GUI_STATE", "run_info.json")))
        return float(info.get("started_at", 0.0))
    except Exception:
        return 0.0


def newest_run_dir(experiment_name: str) -> str | None:
    cands = [
        os.path.join("generated", d)
        for d in os.listdir("generated")
        if d.startswith(experiment_name + "_phased_deployment_run")
    ]
    return max(cands, key=_run_started_at) if cands else None


def experiment_name(template: str) -> str:
    return json.load(open(template))["experiment_name"]


def collect(template: str) -> dict:
    run_dir = newest_run_dir(experiment_name(template))
    if run_dir is None:
        return {}
    steps_path = os.path.join(run_dir, "_GUI_STATE", "steps.json")
    if not os.path.exists(steps_path):
        return {}
    steps = json.load(open(steps_path))["steps"]
    return {name: st.get("target_metric") for name, st in steps.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--filter", default="")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip launching; tabulate the newest existing runs.")
    args = ap.parse_args()

    templates = [t for t in TEMPLATES if args.filter in t]
    log_dir = "/tmp/verification_matrix_logs"
    os.makedirs(log_dir, exist_ok=True)

    if not args.collect_only:
        pending = list(templates)
        running: list[tuple[str, subprocess.Popen]] = []
        while pending or running:
            while pending and len(running) < args.parallel:
                t = pending.pop(0)
                print(f"[matrix] launching {t}", flush=True)
                running.append((t, launch(t, log_dir)))
            time.sleep(15)
            still = []
            for t, p in running:
                if p.poll() is None:
                    still.append((t, p))
                else:
                    print(f"[matrix] finished {t} (rc={p.returncode})", flush=True)
            running = still

    failures = 0
    for t in templates:
        metrics = collect(t)
        if not metrics:
            print(f"{experiment_name(t):45s}  NO RUN FOUND")
            failures += 1
            continue
        sim = metrics.get("SANA-FE Simulation") or metrics.get("Simulation") \
            or metrics.get("Hard Core Mapping")
        ok = sim is not None and float(sim) >= SIM_GATE
        failures += 0 if ok else 1
        flag = "PASS" if ok else "FAIL"
        interesting = {
            k: v for k, v in metrics.items()
            if k in ("Pretraining", "LIF Adaptation", "TTFS Cycle Fine-Tuning",
                     "Clamp Adaptation", "Activation Quantization",
                     "Weight Quantization", "Normalization Fusion",
                     "Soft Core Mapping", "Hard Core Mapping", "Simulation",
                     "SANA-FE Simulation")
        }
        detail = "  ".join(f"{k}={v}" for k, v in interesting.items())
        print(f"{experiment_name(t):45s}  [{flag}] sim={sim}\n    {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
