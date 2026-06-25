"""Run the fast ResNet-50 ImageNet training on GPUs 0,1 while RESERVING them from
the campaign runner via ``free`` GPU leases (so the runner stays up but only uses
2,3). This process holds the leases for the whole run (its PID keeps them live);
they are released on exit. The campaign keeps draining on the other GPUs.
"""
import json
import os
import subprocess
import sys
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import gpu_lease as gl  # noqa: E402

GPUS = [int(g) for g in os.environ.get("IMAGENET_GPUS", "0,1").split(",")]
LEASE_DIR = gl.lease_dir()


def _write_free_leases(pid: int):
    paths = []
    for g in GPUS:
        path = os.path.join(LEASE_DIR, f"{pid}_{uuid.uuid4().hex}.lease")
        payload = {"gpu": g, "pid": pid, "mode": "free", "mb": 90000,
                   "ts": time.time(), "cmd": "imagenet-reserve"}
        with open(path + ".tmp", "w") as fh:
            json.dump(payload, fh)
        os.replace(path + ".tmp", path)
        paths.append(path)
    return paths


def main() -> int:
    pid = os.getpid()
    lease_paths = _write_free_leases(pid)
    print(f"[lease] reserved GPUs {GPUS} (free leases, pid {pid}) -> runner pinned to 2,3", flush=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPUS)
    env["PYTHONPATH"] = f"{_REPO}/src:{_REPO}/spikingjelly"
    cmd = [
        f"{_REPO}/env/bin/python", "-m", "torch.distributed.run",
        f"--nproc_per_node={len(GPUS)}", "--master_port=29577",
        os.path.join(_HERE, "train_imagenet_fast.py"),
        "--batch-size", os.environ.get("IMAGENET_BATCH", "512"),
        "--workers", os.environ.get("IMAGENET_WORKERS", "16"),
        "--epochs", os.environ.get("IMAGENET_EPOCHS", "16"),
        "--eval-every", "4", "--log-every", "50",
        "--out", f"{_REPO}/runs/imagenet/resnet50.pt",
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, cwd=_REPO)
    try:
        rc = proc.wait()
    finally:
        for p in lease_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        print(f"[lease] released GPUs {GPUS} after {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[train] exited rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
