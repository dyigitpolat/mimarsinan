"""Replay the HCM simulation on the latest cifar_vit run with fine-grained
memory probes injected at every meaningful transition.

Loads the cached HCM pickle, constructs a SpikingHybridCoreFlow, and
runs ``max_simulation_samples`` batches exactly the way the pipeline's
HardCoreMappingStep does — logging CPU RSS and CUDA allocated/reserved:

  * after unpickle IR + HCM
  * after flow construct / .to(device)
  * before each batch
  * *inside* every ``_get_segment_tensors`` call (pre/post upload)
  * *inside* every ``_run_neural_segment_*`` call (pre/post matmul loop)
  * inside the compute-op stage branch (pre/post ``gather + execute``)
  * after each batch / post-forward empty_cache

Usage:
    python scripts/profile_cifar_vit_hcm.py <run_dir> \
        [--batches N] [--batch-size N] [--device cuda|cpu]

Dumps a per-phase row every 3rd stage to stay legible, plus a full
batch-boundary trend table.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _mem(device: str) -> tuple[int, int, int, int]:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        rss = 0
    ca = cr = peak = 0
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            ca = torch.cuda.memory_allocated()
            cr = torch.cuda.memory_reserved()
            peak = torch.cuda.max_memory_allocated()
    return rss, ca, cr, peak


class Log:
    def __init__(self, device: str, verbose_stride: int = 1):
        self.device = device
        self.rows: list[tuple[str, int, int, int, int]] = []
        self.stride = verbose_stride

    def mark(self, tag: str, force: bool = False) -> None:
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()
        rss, ca, cr, peak = _mem(self.device)
        if force or (len(self.rows) % self.stride == 0) or tag.startswith(("batch", "after_batch", "before_", "after_unpickle", "after_flow")):
            self.rows.append((tag, rss, ca, cr, peak))

    def print_all(self, batch_only: bool = False) -> None:
        print()
        hdr = f"{'phase':<64} {'RSS':>10} {'alcCUDA':>10} {'rsvCUDA':>10} {'peakCUDA':>10}"
        print(hdr)
        print('-' * len(hdr))
        prev = None
        for tag, rss, ca, cr, peak in self.rows:
            if batch_only and not (tag.startswith("batch_") or tag.startswith("after_batch_") or tag in ("start", "after_unpickle_ir", "after_unpickle_hcm", "after_flow_construct", "after_flow_to_device")):
                continue
            extra = ""
            if prev is not None:
                d_rss = rss - prev[0]
                d_ca = ca - prev[1]
                d_cr = cr - prev[2]
                if abs(d_rss) >= 50e6 or abs(d_ca) >= 50e6 or abs(d_cr) >= 50e6:
                    extra = f"   Δ RSS={d_rss/1e6:+.0f} alc={d_ca/1e6:+.0f} rsv={d_cr/1e6:+.0f}"
            print(f"{tag:<64} {rss/1e6:8.1f} M {ca/1e6:8.1f} M {cr/1e6:8.1f} M {peak/1e6:8.1f} M{extra}")
            prev = (rss, ca, cr, peak)

    def summary(self) -> None:
        boundaries = [r for r in self.rows if r[0].startswith("after_batch_")]
        if len(boundaries) < 2:
            return
        print()
        print("-- Batch-boundary trend --")
        print(f"{'batch':<14} {'RSS':>10} {'alcCUDA':>10} {'rsvCUDA':>10} {'peakCUDA':>10}")
        for tag, rss, ca, cr, peak in boundaries:
            print(f"{tag:<14} {rss/1e6:8.1f} M {ca/1e6:8.1f} M {cr/1e6:8.1f} M {peak/1e6:8.1f} M")
        first, last = boundaries[0], boundaries[-1]
        n = len(boundaries) - 1
        print()
        print(f"Drift over {len(boundaries)} batches:")
        print(f"  RSS   : {(last[1]-first[1])/1e6:+8.2f} MB total   ({(last[1]-first[1])/n/1e6:+.2f} MB/batch)")
        print(f"  alc   : {(last[2]-first[2])/1e6:+8.2f} MB total   ({(last[2]-first[2])/n/1e6:+.2f} MB/batch)")
        print(f"  rsv   : {(last[3]-first[3])/1e6:+8.2f} MB total   ({(last[3]-first[3])/n/1e6:+.2f} MB/batch)")
        print(f"  peak  : {(last[4]-first[4])/1e6:+8.2f} MB total   ({(last[4]-first[4])/n/1e6:+.2f} MB/batch)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true",
                    help="Print the full (very verbose) per-stage log.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    import torch

    device = torch.device(args.device)
    log = Log(args.device, verbose_stride=1)
    log.mark("start", force=True)

    print(f"Loading {args.run_dir}")
    # The cached pickles store torch tensors on whatever CUDA device
    # produced them.  When CUDA_VISIBLE_DEVICES remaps devices (or we
    # run on CPU), remap tensor storages to CPU during unpickle.
    import io
    import torch.serialization as _torch_ser

    def _cpu_map_unpickle(path: str):
        with open(path, "rb") as f:
            raw = f.read()

        def _map_fn(storage, location):
            # Force everything to CPU; tensors are later moved via flow.to(device).
            return storage

        # Monkey-patch default_restore_location so legacy-format torch
        # tensor persistent loads land on CPU.
        orig = _torch_ser.default_restore_location

        def _patched(storage, location):
            try:
                return orig(storage, location)
            except RuntimeError:
                return storage
        _torch_ser.default_restore_location = _patched
        try:
            return pickle.loads(raw)
        finally:
            _torch_ser.default_restore_location = orig

    print("  unpickling IR graph...")
    t0 = time.time()
    ir_graph = _cpu_map_unpickle(str(args.run_dir / "Soft Core Mapping.ir_graph.pickle"))
    print(f"    took {time.time()-t0:.1f}s")
    log.mark("after_unpickle_ir", force=True)

    print("  unpickling HCM...")
    t0 = time.time()
    hcm = _cpu_map_unpickle(str(args.run_dir / "Hard Core Mapping.hard_core_mapping.pickle"))
    print(f"    took {time.time()-t0:.1f}s")
    log.mark("after_unpickle_hcm", force=True)
    # Release IR graph — HCM sim does not need it.
    del ir_graph
    gc.collect()
    log.mark("after_free_ir", force=True)

    with open(args.run_dir / "_RUN_CONFIG" / "config.json") as f:
        cfg = json.load(f)
    dp = cfg.get("deployment_parameters", {})
    firing = dp.get("firing_mode", "TTFS")
    spike = dp.get("spike_generation_mode", "TTFS")
    thresh = dp.get("thresholding_mode", "<=")
    spiking = dp.get("spiking_mode", "ttfs_quantized")
    try:
        with open(args.run_dir / "Model Configuration.scaled_simulation_length.json") as f:
            sim_len = int(json.load(f))
    except Exception:
        sim_len = int(cfg.get("platform_constraints", {}).get("simulation_steps", 32))
    input_shape = (3, 224, 224)
    print(f"Sim config: T={sim_len}, firing={firing}, spike={spike}, thresh={thresh}, mode={spiking}, input={input_shape}")
    print(f"HybridHardCoreMapping: {len(hcm.stages)} stages")

    from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
    flow = SpikingHybridCoreFlow(
        input_shape=input_shape,
        hybrid_mapping=hcm,
        simulation_length=sim_len,
        firing_mode=firing,
        spike_mode=spike,
        thresholding_mode=thresh,
        spiking_mode=spiking,
    )
    log.mark("after_flow_construct", force=True)
    flow = flow.to(device)
    log.mark("after_flow_to_device", force=True)

    # Wrap _get_segment_tensors to log only when the next NEURAL segment
    # begins — compute ops never hit this codepath, and probing every
    # stage (2442) with gc.collect + cuda.synchronize blows past 30s
    # per batch just in probe overhead.
    orig_get = flow._get_segment_tensors
    neural_stage_ptr = [0]
    batch_ptr = [0]

    def _probe_get(stage, dev):
        nidx = neural_stage_ptr[0]
        log.mark(f"b{batch_ptr[0]:02d}_n{nidx:02d}_get_pre", force=True)
        out = orig_get(stage, dev)
        log.mark(f"b{batch_ptr[0]:02d}_n{nidx:02d}_get_post", force=True)
        neural_stage_ptr[0] += 1
        return out

    flow._get_segment_tensors = _probe_get  # type: ignore[assignment]

    # Synthetic input (HCM sim test cares about memory pattern, not data).
    batch_size = args.batch_size
    num_batches = args.batches

    log.mark("before_first_batch", force=True)

    for b in range(num_batches):
        batch_ptr[0] = b
        neural_stage_ptr[0] = 0
        log.mark(f"batch_{b:02d}_pre_forward", force=True)
        x = torch.randn(batch_size, *input_shape, device=device)
        with torch.no_grad():
            y = flow(x)
        log.mark(f"batch_{b:02d}_post_forward", force=True)
        del x, y
        gc.collect()
        if args.device == "cuda":
            torch.cuda.synchronize()
        log.mark(f"after_batch_{b:02d}", force=True)

    if args.full:
        log.print_all(batch_only=False)
    else:
        log.print_all(batch_only=True)
    log.summary()

    # Final: print CUDA memory_stats for fragmentation overview.
    if args.device == "cuda":
        st = torch.cuda.memory_stats()
        print()
        print("-- CUDA memory_stats highlights --")
        for k in ("allocation.all.current", "allocation.all.peak",
                  "reserved_bytes.all.current", "reserved_bytes.all.peak",
                  "active_bytes.all.current", "active_bytes.all.peak",
                  "num_alloc_retries", "num_ooms"):
            v = st.get(k, "n/a")
            if isinstance(v, int) and v > 1024*1024:
                print(f"  {k:<36} {v/1e6:10.2f} MB")
            else:
                print(f"  {k:<36} {v}")


if __name__ == "__main__":
    main()
