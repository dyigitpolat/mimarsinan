#!/usr/bin/env python
"""Run ONE command pinned to a leased GPU (blocks until one is free for the mode).

  python scripts/gpu/with_gpu.py free  --need-mb 8000 -- python run.py --headless cfg.json
  python scripts/gpu/with_gpu.py fit   --need-mb 4000 -- pytest tests/...

`free` waits for an exclusively-free GPU (profiling); `fit` takes any GPU that fits
(correctness). On exit the lease is released so a queued job grabs the GPU at once.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gpu_lease as gl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["free", "fit"])
    p.add_argument("--need-mb", type=int, default=gl.DEFAULT_FIT_MB)
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        p.error("no command given (use -- before the command)")
    lease = gl.acquire_blocking(a.mode, a.need_mb, timeout=a.timeout, cmd=" ".join(cmd))
    if lease is None:
        sys.stderr.write(f"with_gpu: no {a.mode} GPU within {a.timeout}s\n")
        return 75
    sys.stderr.write(f"with_gpu: leased GPU {lease.gpu} ({a.mode})\n")
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lease.gpu))
        return subprocess.run(cmd, env=env).returncode
    finally:
        gl.release(lease)


if __name__ == "__main__":
    raise SystemExit(main())
