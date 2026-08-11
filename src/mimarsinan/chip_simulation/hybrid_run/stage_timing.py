"""Opt-in host-op wall timing for hybrid stage runs (W4 stage 3).

``StageTimer`` accumulates the measured host wall of every ComputeOp stage
invocation (``deployment_record_schema.md`` §2.1 ``ComputeOpRecord.wall_s_total``).
Neural segments are never timed here — the chip simulator measures those.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Tuple


class StageTimer:
    """Per-compute-stage wall accumulation on a monotonic clock.

    Walls are keyed by ``(stage_index, stage_name)``; repeated invocations of
    the same key accumulate (a stage re-run adds to its total). Export order
    is first-observation order.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._walls: Dict[Tuple[int, str], float] = {}
        self._invocations: Dict[Tuple[int, str], int] = {}

    @contextmanager
    def time_compute_stage(self, stage_index: int, name: str) -> Iterator[None]:
        """Time one ComputeOp invocation; accumulates even if the op raises
        (the partial wall is real measurement, and the run fails loudly)."""
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            key = (int(stage_index), str(name))
            self._walls[key] = self._walls.get(key, 0.0) + float(elapsed)
            self._invocations[key] = self._invocations.get(key, 0) + 1

    def compute_stage_walls(self) -> List[Dict[str, Any]]:
        """JSON-safe export: one row per timed compute stage, in
        first-observation order, ``wall_s_total`` matching the
        ``ComputeOpRecord`` field name."""
        return [
            {
                "stage_index": stage_index,
                "name": name,
                "wall_s_total": self._walls[(stage_index, name)],
                "invocations": self._invocations[(stage_index, name)],
            }
            for (stage_index, name) in self._walls
        ]
