"""Opt-in host-op wall timing for hybrid stage runs (W4 stage 3), and [U-HOST] the one per-inference host wall.

``StageTimer`` accumulates the measured host wall of every ComputeOp stage
invocation (``deployment_record_schema.md`` §2.1 ``ComputeOpRecord.wall_s_total``).
Neural segments are never timed here — the chip simulator measures those.

[U-HOST] The timer records, per stage, the wall, the calls and the samples each call carried; the runner that drives
the traversal states that count (``hybrid_stage_runner.run_hybrid_stages``'s ``timed_samples``), never a tensor shape.
``host_wall`` is the one function every consumer reads the host wall through (the record's timing fields, its
quantities, the objective, the measured cost plane, the ODIN FPGA step), under the one definition
``PER_INFERENCE_DEFINITION``. The wall is the development machine's: a diagnostic, never a priced host term (a priced
host is an identified host description's rows, ``platform_physics.hosts.admission``).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

#: [U-HOST] The one definition of the per-inference host wall.
PER_INFERENCE_DEFINITION = (
    "amortized per inference: each host ComputeOp's measured wall over the inferences its timed calls carried "
    "(calls x samples per call; a stage runs once per inference's traversal), summed over the operators; a batched "
    "call's per-sample share, not a one-sample latency, measured on the development machine (a diagnostic, never a "
    "priced host term)")
#: [U-HOST] The label every raw (whole-run) host wall carries where it is printed.
RAW_TOTAL_LABEL = "diagnostic: the raw measured host wall over the whole run (every call, every sample), never priced"
#: [U-HOST2] The one name of a raw (whole-run) host wall field, in the record (timing.latency) and in a fragment: named
#: apart from the per-inference ``host_ops_s``; written beside its label, read by no consumer.
RAW_TOTAL_FIELD = "host_ops_s_raw_total"


class HostWallCountError(ToolchainRefusal, ValueError):
    """[U-HOST] A host wall without the count of the samples its calls carried: it cannot be put per inference.
    [U-HOST3] Also a wall or a count that is not the timer's (``count_refusal``, ``wall_refusal``)."""


def count_refusal(value: Any) -> Optional[str]:
    """[U-HOST3] Why ``value`` is not a count the timer writes (its calls, the samples each call carried: a positive
    int, never a bool), or None. Every reader of a sealed count applies it (``host_wall``, the fold)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"{value!r} is not a positive int"
    return None


def wall_refusal(value: Any) -> Optional[str]:
    """[U-HOST3] Why ``value`` is not a wall the timer writes (a finite number of seconds at or above zero, never a
    bool), or None. Every reader of a sealed wall applies it (``host_wall``, the fold)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return f"{value!r} is not a finite number of seconds at or above zero"
    return None


def timed_sample_count(samples: Any, where: str) -> int:
    if count_refusal(samples) is not None:
        raise HostWallCountError(f"{where}: a timed host call needs the samples it carries (a positive int stated by "
                                 f"the runner that drives the traversal), got {samples!r}")
    return samples


class StageTimer:
    """Per-compute-stage wall accumulation on a monotonic clock.

    Walls are keyed by ``(stage_index, stage_name)``; repeated invocations of
    the same key accumulate (a stage re-run adds to its total). Export order
    is first-observation order. [U-HOST] Each call states the samples it
    carries; one key's calls carry one count (a timer times one traversal size).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._walls: Dict[Tuple[int, str], float] = {}
        self._invocations: Dict[Tuple[int, str], int] = {}
        self._samples: Dict[Tuple[int, str], int] = {}

    @contextmanager
    def time_compute_stage(self, stage_index: int, name: str, samples: int) -> Iterator[None]:
        """Time one ComputeOp invocation of ``samples`` samples; accumulates even if the op raises
        (the partial wall is real measurement, and the run fails loudly)."""
        key = (int(stage_index), str(name))
        count = timed_sample_count(samples, f"host stage {name!r} (stage {stage_index})")
        if self._samples.setdefault(key, count) != count:
            raise HostWallCountError(f"host stage {name!r} (stage {stage_index}) is timed at {self._samples[key]} and "
                                     f"{count} samples per call: one timer times one traversal size")
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            self._walls[key] = self._walls.get(key, 0.0) + float(elapsed)
            self._invocations[key] = self._invocations.get(key, 0) + 1

    def compute_stage_walls(self) -> List[Dict[str, Any]]:
        """JSON-safe export: one row per timed compute stage, in
        first-observation order, ``wall_s_total`` matching the
        ``ComputeOpRecord`` field name, with its calls and samples per call."""
        return [
            {
                "stage_index": stage_index,
                "name": name,
                "wall_s_total": self._walls[(stage_index, name)],
                "invocations": self._invocations[(stage_index, name)],
                "samples_per_call": self._samples[(stage_index, name)],
            }
            for (stage_index, name) in self._walls
        ]


@dataclass(frozen=True)
class HostWall:
    """[U-HOST] The host wall of a set of timed host operators under ``PER_INFERENCE_DEFINITION``: the
    per-inference wall (None when nothing was timed or the count is missing), the raw total (``RAW_TOTAL_LABEL``)
    and, where the per-inference wall cannot be stated, why (naming the operators)."""

    per_inference_s: Optional[float]
    raw_total_s: Optional[float]
    refusal: Optional[str] = None
    definition: str = PER_INFERENCE_DEFINITION

    @property
    def timed(self) -> bool:
        return self.raw_total_s is not None

    def required(self) -> Optional[float]:
        """The per-inference wall, a refusal raised by name where the count is missing (None when untimed)."""
        if self.refusal is not None:
            raise HostWallCountError(self.refusal)
        return self.per_inference_s


def _field(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)


def host_wall(entries: Iterable[Any]) -> HostWall:
    """[U-HOST] The one function: the host wall of ``entries`` (the timer's rows, a run's rows over several runs, or a
    sealed schedule's stages; an entry without a ``wall_s_total`` is untimed), grouped by operator name: each
    operator's summed wall over the inferences its calls carried (calls x samples per call), summed over the
    operators. A timed operator without its calls or samples per call refuses the per-inference wall by name.
    [U-HOST3] The writers' rule is applied to every sealed wall and count it reads: a count that is not a positive int
    withholds the per-inference wall by name, and a wall that is not a finite non-negative number withholds the raw
    total as well, by name (a hand-edited or foreign record never prints a wall)."""
    walls: Dict[str, float] = {}
    inferences: Dict[str, int] = {}
    uncounted: List[str] = []
    unread: List[str] = []
    for entry in entries:
        wall = _field(entry, "wall_s_total")
        if wall is None:
            continue
        name = str(_field(entry, "name"))
        if (why := wall_refusal(wall)) is not None:
            unread.append(f"{name} (wall_s_total {why})")
            continue
        calls, per_call = _field(entry, "invocations"), _field(entry, "samples_per_call")
        walls[name] = walls.get(name, 0.0) + float(wall)
        bad = [f"{field} {why}" for field, count in (("invocations", calls), ("samples_per_call", per_call))
               if (why := count_refusal(count)) is not None]
        if bad:
            uncounted.append(f"{name} ({'; '.join(bad)})")
            continue
        inferences[name] = inferences.get(name, 0) + calls * per_call
    if unread:
        return HostWall(per_inference_s=None, raw_total_s=None, refusal=(
            f"the sealed host walls of {sorted(set(unread))} are not walls the timer writes: the wall and its raw "
            "total are withheld"))
    if not walls:
        return HostWall(per_inference_s=None, raw_total_s=None)
    raw = float(sum(walls.values()))
    if uncounted:
        return HostWall(per_inference_s=None, raw_total_s=raw, refusal=(
            f"the host walls of {sorted(set(uncounted))} carry no calls or samples per call the timer writes (a wall "
            "measured before the timer recorded the samples each call carried, by a runner that stated none, or a "
            "count edited after it was sealed): the wall cannot be put per inference, and a guessed count would be a "
            "proxy presented as a measurement"))
    return HostWall(per_inference_s=float(sum(walls[name] / inferences[name] for name in walls)), raw_total_s=raw)
