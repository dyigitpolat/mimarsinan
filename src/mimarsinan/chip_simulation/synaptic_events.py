"""[H1] Synaptic-event census: measured arrivals ⊗ occupied consumer columns."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy


class EventCensusRefusal(ToolchainRefusal, ValueError):
    """A run record that cannot answer the event census of its program: a stage or a core it lacks, named."""


def census_by_unit(record: Any, mapping: Any) -> Dict[int, int]:
    """[U-S] Measured synaptic events of one executed sample per neural execution unit, keyed as the record is.

    ``record`` is a ``RunRecord`` (the Hard Core Mapping recorder's, or any backend's projection to it): per core,
    the per-axon input counts the core integrated inside its window. A core's events are its recorded arrivals
    times its used columns (the estimand of ``synaptic_event_census``, read on the consumer side). The record's
    index is the executor's (``enumerate_execution_stages``: level stages included); a unit or a used core the
    record lacks, or counts over other axons than the core uses, refuse by name rather than undercount.
    """
    out: Dict[int, int] = {}
    for index, _stage, unit in enumerate_execution_stages(mapping.stages):
        if unit.kind != "neural":
            continue
        where = f"neural stage {index} ({getattr(unit, 'name', '?')!r})"
        segment = record.segments.get(index)
        occupancies = [CoreOccupancy.from_hard_core(core) for core in unit.hard_core_mapping.cores]
        if segment is None and any(o.neurons_used > 0 for o in occupancies):
            raise EventCensusRefusal(
                f"{where} has no segment in the run record (sample {record.sample_index}); the census refuses "
                f"rather than undercount")
        recorded = {} if segment is None else {int(c.core_index): c for c in segment.cores}
        events = 0
        for core_index, occupancy in enumerate(occupancies):
            if occupancy.neurons_used <= 0:
                continue
            counts = recorded.get(core_index)
            if counts is None:
                raise EventCensusRefusal(
                    f"{where}: core {core_index} has no recorded input counts in the run record (sample "
                    f"{record.sample_index}); the census refuses rather than undercount")
            arrivals = np.asarray(counts.input_spike_count, dtype=np.int64)
            if arrivals.size != max(occupancy.axons_used, 1):
                raise EventCensusRefusal(
                    f"{where}: core {core_index} records {arrivals.size} axons but uses {occupancy.axons_used}; "
                    f"the record and the program's index books disagree")
            events += int(arrivals.sum()) * int(occupancy.neurons_used)
        out[index] = events
    return out


def census_from_run_record(record: Any, mapping: Any) -> int:
    """[U-S] Measured synaptic events of one executed sample: the sum of ``census_by_unit``."""
    return int(sum(census_by_unit(record, mapping).values()))


def census_from_trace_groups(
    cores: Iterable[Any],
    *,
    group_spike_counts,
    core_to_group,
    boundary_arrivals,
) -> Optional[int]:
    """The census over a runner's trace-group tallies — the ONE join site.

    ``core_to_group`` values are GROUP OBJECTS; the name-keyed counts are
    joined through the group's own name (joining on the object itself
    refused every stage whose cores read other cores — the fused
    multi-latency shape, first hit by LeNet5's classifier [H5]).
    """
    def _emissions(core_index: int):
        group = core_to_group.get(core_index)
        if group is None:
            return None
        name = group.get_name() if hasattr(group, "get_name") else group.name
        return group_spike_counts.get(name)

    return synaptic_event_census(
        cores,
        emissions_of=_emissions,
        boundary_arrivals_of=lambda c: boundary_arrivals.get(c, 0),
    )


def synaptic_event_census(
    cores: Iterable[Any],
    *,
    emissions_of: Callable[[int], Optional[Sequence[int]]],
    boundary_arrivals_of: Callable[[int], int],
) -> Optional[int]:
    """Measured synaptic events of one executed stage, or ``None``.

    One event = one spike arriving at one OCCUPIED cell: a spike entering a
    consumer's axon row drives that core's used columns. Inter-core arrivals
    come from the producer's measured per-neuron emissions through the HCM's
    own axon-source spans; input and always-on arrivals are already tallied
    per consumer by the trace groups (``boundary_arrivals_of``). ``None``
    from ``emissions_of`` for a producer some span reads means the trace was
    not parsed — the census refuses rather than undercounting.
    """
    total = 0
    for consumer_index, core in enumerate(cores):
        columns = CoreOccupancy.from_hard_core(core).neurons_used
        if columns <= 0:
            continue
        arrivals = int(boundary_arrivals_of(consumer_index))
        for span in core.get_axon_source_spans():
            if span.kind != "core":
                continue  # input/on arrivals are measured per consumer above
            emissions = emissions_of(int(span.src_core))
            if emissions is None:
                return None
            lo, hi = int(span.src_start), int(span.src_start) + int(span.length)
            if hi > len(emissions):
                raise ValueError(
                    f"axon-source span [{lo}, {hi}) of consumer core "
                    f"{consumer_index} reads past producer core "
                    f"{span.src_core}'s {len(emissions)} traced neurons — "
                    f"the index books disagree"
                )
            arrivals += int(sum(int(count) for count in emissions[lo:hi]))
        total += arrivals * int(columns)
    return total
