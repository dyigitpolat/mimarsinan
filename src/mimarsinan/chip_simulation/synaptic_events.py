"""[H1] Synaptic-event census: measured arrivals ⊗ occupied consumer columns."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from mimarsinan.mapping.crossbar_utilization import CoreOccupancy


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
