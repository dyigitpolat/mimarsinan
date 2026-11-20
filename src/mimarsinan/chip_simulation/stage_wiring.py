"""[U-LIF2 r6, m-2] An execution stage's wiring as the gated executor reads it, for every reader of it.

One leveling (``dependency_levels``): a core's dependency level in its stage is 0 when it reads no stepped core of the
stage, else one more than its deepest stepped producer, walked in the one R6 walk order (``dependency_order``, which
refuses a cycle of reads by name); a core never stepped has none and is no producer. The gate levels its stage tensors
by it (``gated_window.stage_dependency_levels``) and the program census levels a built pass by it
(``program_census.pass_dependency_depth``), so the depth a record seals and the depth the gate runs are one reading.
The producer windows a stage's neurons integrate before their own windows (``stage_timesteps.stage_window_arrivals``)
are read from the same levels and reads, over the stages a built pass executes (its re-timed level stages, else
itself): the record seals them and the cost rule prices them."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.stage_timesteps import stage_window_arrivals
from mimarsinan.chip_simulation.value_run.value_execution import dependency_order
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources


def dependency_levels(reads: Sequence[Sequence[int]], latencies: Sequence[int],
                      stepped: Optional[Sequence[bool]] = None) -> Tuple[Optional[int], ...]:
    """Each core's dependency level over its in-stage ``reads`` (``stepped`` None: every core is stepped)."""
    stepped = [True] * len(reads) if stepped is None else list(stepped)
    levels: List[Optional[int]] = [None] * len(reads)
    for i in dependency_order(reads, latencies):
        if stepped[i]:
            levels[i] = 1 + max((lv for lv in (levels[p] for p in reads[i]) if lv is not None), default=-1)
    return tuple(levels)


def pass_wiring(segment: Any) -> Tuple[List[List[int]], Tuple[Optional[int], ...]]:
    """A built pass's (hard core mapping's) in-pass core reads and each core's level, from its cores' axon sources."""
    reads = [sorted({int(span.src_core) for span in compress_spike_sources(core.axon_sources) if span.kind == "core"})
             for core in segment.cores]
    return reads, dependency_levels(reads, [int(getattr(core, "latency", 0) or 0) for core in segment.cores])


def executed_window_arrivals(stage: Any) -> int:
    """The producer windows a built neural stage's neurons integrate before their own windows under integrate then
    emit, over the stages it executes (its re-timed level stages, else itself), each core weighted by the neurons it
    drives."""
    total = 0
    for unit in getattr(stage, "retimed_level_stages", None) or [stage]:
        segment = unit.hard_core_mapping
        reads, levels = pass_wiring(segment)
        neurons = [CoreOccupancy.from_hard_core(core).neurons_used for core in segment.cores]
        total += stage_window_arrivals(neurons, reads, levels)
    return total
