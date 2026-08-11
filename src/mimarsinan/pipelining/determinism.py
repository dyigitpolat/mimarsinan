"""A run's randomness: the ONE place it is pinned, and the ONE place it is isolated.

``apply_determinism`` seeds every RNG family a run draws from;
``isolated_rng_stream`` is its counterpart — the sanctioned way for a side
computation to consume, or even RE-SEED, randomness without moving the stream
the deployment itself draws from. The two enumerate the SAME families, which is
why they live in one module: a family added to one and forgotten in the other is
a side computation that silently re-rolls the weights a run deploys.
"""

from __future__ import annotations

import contextlib
import random
from typing import Iterator, List

import numpy as np
import torch


def apply_determinism(seed: int) -> None:
    """Sole owner of the registry's ``PipelineSession/determinism`` contract:
    seed every RNG family and pin deterministic fp32 math, once per session and
    before any step runs (docs/research/findings/numerical_boundary_consistency.md §5a/§5c)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # TF32 rounds matmul/conv inputs to 10-bit mantissas and flips staircase boundaries; fp32 evaluation is flip-free on the probed stacks.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _cuda_devices() -> List[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


@contextlib.contextmanager
def isolated_rng_stream() -> Iterator[None]:
    """Run a block without letting it move the run's RNG stream.

    Every family :func:`apply_determinism` seeds is put back on exit, RE-SEEDING
    included — a block that calls ``torch.manual_seed`` is undone, not merely
    rewound past what it consumed. That is the case this exists for: candidate
    scoring seeds the world to its own scoring seed (deliberately: it is what
    makes a candidate's score reproducible), so without isolation the weights a
    run deploys depend on how many candidates a search happened to look at, and
    the same config trains differently with the search ON than with the winning
    chip declared by hand.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=_cuda_devices(), enabled=True):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
