"""Per-fine-tuning-PASS wall log (AC5): monotonic-clock walls + the worst pass.

AC5 = "no fine-tuning step exceeds 5 min" is judged PER fine-tuning PASS, not on
the end-to-end pipeline wall (which is dominated by non-FT steps — Soft Core
Mapping, Weight Quantization, Simulation). Each adaptation PASS (the
recover_to / ramp / stabilize passes inside the SmoothAdaptation cycle) is timed
here with a MONOTONIC clock (immune to wall-clock / NTP jumps), and the MAX single
pass is surfaced as ``max_ft_pass_wall_s`` (the exact field A4's AC5 verdict reads).

Timing only — recording a pass changes no numerics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List


class FtPassWallLog:
    """Ordered log of fine-tuning-pass walls, with the worst-pass max.

    ``record(label, wall_s)`` appends a ``{"label", "wall_s"}`` entry; ``time_pass``
    is the context-manager form that times a block on ``time.monotonic`` (the wall
    is recorded even when the block raises). ``max_wall_s`` is the worst single pass
    (0.0 when empty — AC5 reads a float, never ``None``)."""

    def __init__(self) -> None:
        self._passes: List[Dict[str, object]] = []

    def record(self, label: str, wall_s: float) -> None:
        wall = float(wall_s)
        if wall < 0.0:
            raise ValueError(
                f"a fine-tuning pass wall must be non-negative; got {wall} for "
                f"{label!r}"
            )
        self._passes.append({"label": str(label), "wall_s": wall})

    @contextmanager
    def time_pass(self, label: str) -> Iterator[None]:
        """Time a block on the monotonic clock and record it under ``label``."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(label, time.monotonic() - t0)

    @property
    def passes(self) -> List[Dict[str, object]]:
        """The per-pass breakdown, in record order (a shallow copy)."""
        return [dict(p) for p in self._passes]

    @property
    def max_wall_s(self) -> float:
        """The worst single fine-tuning pass wall (0.0 when no pass ran)."""
        if not self._passes:
            return 0.0
        return max(float(p["wall_s"]) for p in self._passes)
