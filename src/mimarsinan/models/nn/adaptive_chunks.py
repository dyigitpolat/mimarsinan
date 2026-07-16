"""[C4'] adaptive batch chunking on CUDA OOM for pure-eval forwards."""

from __future__ import annotations

import torch


def _is_cuda_oom(exc):
    """True when ``exc`` or anything on its ``__cause__`` chain is a CUDA OOM
    (fail-loud executors wrap forward failures with node context)."""
    while exc is not None:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        exc = exc.__cause__
    return False


def forward_adaptive_chunks(forward, x):
    """Full-batch forward, halving into chunks on CUDA OOM: cycle-accurate
    spiking forwards materialize S x batch x features, so a val batch sized
    for the fp32 loaders can exceed VRAM at deploy-eval time (measured
    36.94 GiB = 32 cycles x 512 batch on a ViT). Value-identical for
    per-sample-independent eval (recomputed per retry); an OOM at chunk 1
    fails loud. Callers must be grad-free (chunking a training forward would
    change gradient accumulation)."""
    chunk = int(x.size(0))
    while True:
        try:
            if chunk >= x.size(0):
                return forward(x)
            return torch.cat([
                forward(part) for part in torch.split(x, chunk)
            ])
        except RuntimeError as exc:  # torch.OutOfMemoryError subclasses this
            if not _is_cuda_oom(exc) or chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
