"""Distinct-wire census of the layout walk — the NoC estimator's traffic basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_source_view_ops import total_size

#: Per-node walk record: ({real producer node id -> distinct cells},
#: network-input cells, always-on cells incl. the bias axon).
NodeCellCensus = Tuple[Dict[int, int], int, int]


@dataclass(frozen=True)
class LayoutWireCensus:
    """Distinct wires between softcores, counted at the layout walk.

    ``pair_wires[(p, c)]`` is the number of DISTINCT neurons of softcore ``p``
    feeding softcore ``c`` — the message-generation basis (one message per
    firing source neuron per destination core). ``input_wires[c]`` counts
    network-input and host-boundary cells (they re-enter as segment input,
    never as mesh traffic); ``on_wires[c]`` counts always-on cells plus the
    bias axon (they fire every active cycle).
    """

    pair_wires: Mapping[Tuple[int, int], int]
    input_wires: Tuple[int, ...]
    on_wires: Tuple[int, ...]


def source_cell_census(input_sources) -> NodeCellCensus:
    """Distinct-cell census of one consumer's inputs.

    Duplicate ``(node, index)`` cells count once (a source neuron wired to two
    axons of one consumer core still sends one message per spike); ``off``
    cells carry no traffic and count nowhere.
    """
    if input_sources is None:
        return {}, 0, 0
    arr = np.asarray(input_sources, dtype=object).ravel()
    seen: set = set()
    pairs: Dict[int, int] = {}
    input_cells = 0
    on_cells = 0
    for src in arr:
        if not isinstance(src, IRSource):
            continue
        key = (src.node_id, src.index)
        if key in seen:
            continue
        seen.add(key)
        if src.node_id >= 0:
            pairs[src.node_id] = pairs.get(src.node_id, 0) + 1
        elif src.is_input():
            input_cells += 1
        elif src.is_always_on():
            on_cells += 1
    return pairs, input_cells, on_cells


def census_of_walk(mapping) -> "LayoutWireCensus":
    """The walked layout mapping's census; refuses when collection was off."""
    if not getattr(mapping, "collect_wire_census", False):
        raise ValueError(
            "wire census was not collected; construct the layout mapping with "
            "collect_wire_census=True before the walk")
    return build_wire_census(
        mapping._node_wire_census, mapping._node_is_neural,
        mapping._node_id_to_softcore_idx, len(mapping.layout_softcores))


def record_emission_census(
    store: Dict[int, NodeCellCensus], node_id: int, input_sources, input_count,
) -> None:
    """Record one softcore emission's census; the axon count beyond the view
    is the bias axon, which fires every cycle like an always-on cell."""
    pairs, input_cells, on_cells = source_cell_census(input_sources)
    bias_axons = max(0, int(input_count) - total_size(input_sources))
    store[int(node_id)] = (pairs, input_cells, on_cells + bias_axons)


def build_wire_census(
    node_census: Mapping[int, NodeCellCensus],
    node_is_neural: Mapping[int, bool],
    node_to_softcore: Mapping[int, int],
    softcore_count: int,
) -> LayoutWireCensus:
    """Assemble the walk records into softcore-indexed wires.

    A non-neural producer (a host compute op) breaks the on-chip pair: its
    cells arrive at the consumer as SEGMENT INPUT, so they fold into
    ``input_wires`` rather than into a mesh pair.
    """
    pair_wires: Dict[Tuple[int, int], int] = {}
    input_wires = [0] * int(softcore_count)
    on_wires = [0] * int(softcore_count)
    for node_id, (pairs, input_cells, on_cells) in node_census.items():
        consumer = node_to_softcore.get(node_id)
        if consumer is None:
            continue
        input_total = int(input_cells)
        for producer_node, count in pairs.items():
            if node_is_neural.get(producer_node, False):
                producer = node_to_softcore[producer_node]
                key = (producer, consumer)
                pair_wires[key] = pair_wires.get(key, 0) + int(count)
            else:
                input_total += int(count)
        input_wires[consumer] = input_total
        on_wires[consumer] = int(on_cells)
    return LayoutWireCensus(
        pair_wires=pair_wires,
        input_wires=tuple(input_wires),
        on_wires=tuple(on_wires),
    )
