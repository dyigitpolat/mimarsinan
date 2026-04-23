"""Fine-grained memory-usage profiler for the HCM simulation path.

Loads a cached IR graph + HCM mapping from a previous run and replays
the HCM simulation on a small sample, logging CPU RSS and CUDA
allocated / reserved memory at every meaningful point:

  * after IR/HCM unpickle
  * after SpikingHybridCoreFlow construction
  * after .to(device)
  * before every batch
  * before the forward's per-stage loop (entry)
  * after each stage (neural or compute)
  * after the forward
  * after each batch's ``del`` cleanup

Run:

    python scripts/profile_hcm_sim.py <run_dir> [--batches N] [--device cpu|cuda]

Reports two things at exit:

  1. A per-phase table of (rss, cuda_alloc, cuda_reserved).
  2. A ``delta`` column showing growth batch-over-batch (if any) —
     a non-zero trend indicates a leak.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _mem_snapshot(device: str) -> Tuple[int, int, int]:
    """Return (rss_bytes, cuda_allocated_bytes, cuda_reserved_bytes)."""
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        rss = 0
    cuda_alloc = 0
    cuda_reserved = 0
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            cuda_alloc = torch.cuda.memory_allocated()
            cuda_reserved = torch.cuda.memory_reserved()
    return rss, cuda_alloc, cuda_reserved


def _fmt_mb(n: int) -> str:
    return f"{n / 1e6:8.2f} MB"


class MemLog:
    """Record (phase_tag, rss, cuda_alloc, cuda_reserved) tuples."""

    def __init__(self, device: str):
        self.device = device
        self.rows: List[Tuple[str, int, int, int]] = []

    def mark(self, tag: str) -> None:
        # Let Python release unreachable refs before sampling so the
        # sample reflects live memory rather than GC-delay noise.
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
        rss, ca, cr = _mem_snapshot(self.device)
        self.rows.append((tag, rss, ca, cr))

    def print_table(self) -> None:
        print()
        print(f"{'phase':<52} {'RSS':>12} {'CUDA alloc':>12} {'CUDA rsv':>12}")
        print('-' * 92)
        prev = None
        for tag, rss, ca, cr in self.rows:
            suffix = ''
            if prev is not None:
                d_rss = rss - prev[0]
                d_ca = ca - prev[1]
                d_cr = cr - prev[2]
                if abs(d_rss) >= 5 * 1024 * 1024 or abs(d_ca) >= 5 * 1024 * 1024:
                    suffix = f"   ΔRSS={d_rss/1e6:+.1f} ΔalcCUDA={d_ca/1e6:+.1f} ΔrsvCUDA={d_cr/1e6:+.1f}"
            print(f"{tag:<52} {_fmt_mb(rss)} {_fmt_mb(ca)} {_fmt_mb(cr)}{suffix}")
            prev = (rss, ca, cr)

    def summarise_batches(self) -> None:
        """Isolate batch-boundary rows and report growth trend."""
        boundary_rows = [
            (tag, rss, ca, cr)
            for tag, rss, ca, cr in self.rows
            if tag.startswith("after_batch_")
        ]
        if len(boundary_rows) < 2:
            return
        print()
        print("-- Batch-boundary trend --")
        print(f"{'batch':<12} {'RSS':>12} {'CUDA alloc':>12} {'CUDA rsv':>12}")
        for tag, rss, ca, cr in boundary_rows:
            print(f"{tag:<12} {_fmt_mb(rss)} {_fmt_mb(ca)} {_fmt_mb(cr)}")
        first = boundary_rows[0]
        last = boundary_rows[-1]
        d_rss = last[1] - first[1]
        d_ca = last[2] - first[2]
        d_cr = last[3] - first[3]
        print()
        print(f"Total drift across {len(boundary_rows)} batches:")
        print(f"  RSS:        {d_rss/1e6:+.2f} MB  ({d_rss / max(1, len(boundary_rows)-1) / 1e6:+.2f} MB/batch)")
        print(f"  CUDA alloc: {d_ca/1e6:+.2f} MB  ({d_ca / max(1, len(boundary_rows)-1) / 1e6:+.2f} MB/batch)")
        print(f"  CUDA rsv:   {d_cr/1e6:+.2f} MB  ({d_cr / max(1, len(boundary_rows)-1) / 1e6:+.2f} MB/batch)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    # src/ layout import path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    import numpy as np
    import torch

    device = torch.device(args.device)
    log = MemLog(args.device)
    log.mark("start")

    ir_path = args.run_dir / "Soft Core Mapping.ir_graph.pickle"
    hcm_path = args.run_dir / "Hard Core Mapping.hard_core_mapping.pickle"
    if not hcm_path.exists():
        print(f"Missing {hcm_path}; this run did not reach HCM step.")
        sys.exit(1)

    with open(ir_path, "rb") as f:
        ir_graph = pickle.load(f)
    log.mark("after_unpickle_ir_graph")

    with open(hcm_path, "rb") as f:
        hcm = pickle.load(f)
    log.mark("after_unpickle_hcm")

    from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow

    # Pull sim parameters from the run directory.
    import json
    with open(args.run_dir / "_RUN_CONFIG" / "config.json") as f:
        run_config = json.load(f)
    dp = run_config.get("deployment_parameters", {})
    firing = dp.get("firing_mode", "TTFS")
    spike = dp.get("spike_generation_mode", "TTFS")
    thresh = dp.get("thresholding_mode", "<=")
    spiking = dp.get("spiking_mode", "ttfs_quantized")
    try:
        with open(args.run_dir / "Model Configuration.scaled_simulation_length.json") as f:
            sim_len = int(json.load(f))
    except Exception:
        sim_len = int(run_config.get("platform_constraints", {}).get("simulation_steps", 32))
    # MNIST default; override via env if needed.
    input_shape = tuple(int(x) for x in os.environ.get("INPUT_SHAPE", "1,28,28").split(","))
    print(f"Sim config: T={sim_len}, firing={firing}, spike={spike}, "
          f"thresh={thresh}, mode={spiking}, input={input_shape}")

    flow = SpikingHybridCoreFlow(
        input_shape=input_shape,
        hybrid_mapping=hcm,
        simulation_length=sim_len,
        firing_mode=firing,
        spike_mode=spike,
        thresholding_mode=thresh,
        spiking_mode=spiking,
    )
    log.mark("after_flow_construct")

    flow = flow.to(device)
    log.mark("after_flow_to_device")

    # Install per-stage hooks on the forward loop by monkey-patching
    # _run_neural_segment_* and add_compute_op path.  Instead of patching
    # methods we wrap forward and log before+after each stage ourselves
    # by hand — simpler: wrap _get_segment_tensors to log entry/exit.
    orig_get_seg = flow._get_segment_tensors
    stage_counter = [0]

    def _wrapped_get_seg(stage, dev):
        idx = stage_counter[0]
        log.mark(f"b?_stage{idx}_get_tensors_pre")
        out = orig_get_seg(stage, dev)
        log.mark(f"b?_stage{idx}_get_tensors_post")
        stage_counter[0] += 1
        return out

    flow._get_segment_tensors = _wrapped_get_seg  # type: ignore[assignment]

    # Synthetic inputs — the exact distribution does not matter for a
    # memory profile; we just need consistent shapes and batch sizes.
    batch_size = args.batch_size
    num_batches = args.batches
    log.mark("before_first_batch")

    n_stages = len(hcm.stages)
    n_neural = sum(1 for s in hcm.stages if s.kind == "neural")
    print(f"HybridHardCoreMapping: {n_stages} stages "
          f"({n_neural} neural), simulation_length={sim_len}")

    for b in range(num_batches):
        stage_counter[0] = 0
        # Rename pending b?_stage tags for this batch:
        # drop and re-mark on the fly by editing the last n rows.
        batch_start_idx = len(log.rows)
        log.mark(f"batch_{b:02d}_pre_forward")
        x = torch.randn(batch_size, *input_shape, device=device)
        with torch.no_grad():
            y = flow(x)
        log.mark(f"batch_{b:02d}_post_forward")
        # Rewrite any wildcard "b?" tags emitted during this forward.
        for i in range(batch_start_idx, len(log.rows)):
            tag, rss, ca, cr = log.rows[i]
            if tag.startswith("b?"):
                log.rows[i] = (tag.replace("b?", f"b{b:02d}", 1), rss, ca, cr)
        del x, y
        log.mark(f"after_batch_{b:02d}")

    log.print_table()
    log.summarise_batches()


if __name__ == "__main__":
    main()
