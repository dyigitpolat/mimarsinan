"""[calculus §17] the spike-count certificate: the faithfulness observable.

The certification observable is per-neuron, per-window SPIKE COUNTS —
integers. Post-WQ chip arithmetic is integer weights x integer counts, so
count equality across backends is exact BY CONSTRUCTION under the
integer-exact accumulation contract; the float host ops are one shared torch
implementation in every backend. accuracy(oracle) + counts(oracle == backend)
=> accuracy(backend), derived rather than re-measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

# Per-backend exactness classes. `exact`: we control the executor and demand
# bit equality. `counts-export`: hardware quantizes internally; a documented
# +-1-count tolerance applies (Loihi exports counts only).
BACKEND_CLASSES: dict[str, str] = {
    "hcm": "exact",
    "scm": "exact",
    "nevresim": "exact",
    "sanafe": "exact",
    "loihi": "counts-export",
}

_CLASS_ATOL: dict[str, float] = {"exact": 0.0, "counts-export": 1.0}

_MAX_DIVERGENT_REPORTED = 16


@dataclass
class SpikeCountCertificate:
    """Typed verdict of one reference<->backend spike-count comparison."""

    backend: str
    backend_class: str
    samples: int
    neuron_windows_compared: int
    exact_match_fraction: float
    max_abs_delta: float
    divergent: list = field(default_factory=list)
    passed: bool = False

    def summary(self) -> str:
        return (
            f"spike-count certificate [{self.backend}/{self.backend_class}]: "
            f"{'PASS' if self.passed else 'FAIL'} "
            f"exact={self.exact_match_fraction:.6f} "
            f"max|dcount|={self.max_abs_delta:g} "
            f"over {self.neuron_windows_compared} neuron-windows, "
            f"{self.samples} sample(s)"
        )


def certify_spike_counts(
    reference_counts: Callable[[torch.Tensor], dict],
    backend_counts: Callable[[torch.Tensor], dict],
    sample_batches: Sequence[torch.Tensor],
    *,
    backend: str,
) -> SpikeCountCertificate:
    """Compare per-neuron window counts key-by-key over the sample batches.

    Both callables map a batch to ``{key: (B, n) count tensor}`` with matching
    keys (segment/core identifiers). Fail-loud on unknown backends and on
    missing keys; the pass rule is the backend class's count tolerance.
    """
    if backend not in BACKEND_CLASSES:
        raise KeyError(
            f"unknown certification backend {backend!r}; classify it in "
            f"BACKEND_CLASSES (exact | counts-export) first"
        )
    backend_class = BACKEND_CLASSES[backend]
    atol = _CLASS_ATOL[backend_class]

    compared = 0
    matched = 0
    max_delta = 0.0
    divergent: list = []
    n_samples = 0
    for batch in sample_batches:
        n_samples += int(batch.shape[0]) if hasattr(batch, "shape") else 1
        ref = reference_counts(batch)
        got = backend_counts(batch)
        missing = set(ref) - set(got)
        if missing:
            raise KeyError(f"backend {backend!r} missing count keys: {sorted(missing)!r}")
        for key, r in ref.items():
            g = got[key].to(r.dtype)
            delta = (g - r).abs()
            compared += int(delta.numel())
            matched += int((delta == 0).sum())
            d_max = float(delta.max()) if delta.numel() else 0.0
            max_delta = max(max_delta, d_max)
            if d_max > 0 and len(divergent) < _MAX_DIVERGENT_REPORTED:
                idx = torch.nonzero(delta > 0, as_tuple=False)[0].tolist()
                divergent.append((key, idx, d_max))

    return SpikeCountCertificate(
        backend=backend,
        backend_class=backend_class,
        samples=n_samples,
        neuron_windows_compared=compared,
        exact_match_fraction=(matched / compared) if compared else 1.0,
        max_abs_delta=max_delta,
        divergent=divergent,
        passed=max_delta <= atol,
    )
