"""[E6] The ``ir-tiling`` keys of the IR neurons with no perceptron identity the integrated units build.

A pool lane's reduction core (WI-5e, ``lanes/ir/pool_cores``) replaces its pool op: its column ``j`` is the pool's
output ``lane_outputs[j]``, keyed as the host op it replaces would be (its name and output index); a producer-side
reduction (D11, ``lanes/ir/reduce_cores``) writes its pool's outputs from one perceptron's accumulators and is keyed by
that perceptron and the output index. Neither depends on how the IR's per-core limits group the outputs into cores.

A depth-balancing relay (C5, ``latency/depth_balancing``) copies one neuron per column one cycle later; it is keyed by
the neuron its column relays (a relay of a relay by that relay's key), so a chain at the declared geometry and the
chain at unbounded geometry relay the same neurons under the same keys. A relay column reading no single source is not
keyed (``unkeyed_neuron``)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS, LANE_SOURCE_NAME
from mimarsinan.mapping.latency.depth_balancing import RELAY_MARKER

LANE, RELAY = "lane", "relay"
#: A unit-keyed node: its kind, one owner per column, and the channel each column holds under its owner.
UnitKey = Tuple[str, List[Tuple[Any, ...]], np.ndarray]


def unit_key(node: Any, graph: Any, sources: Sequence[Any], rows: np.ndarray,
             source_codes: Callable[[list], np.ndarray]) -> Optional[UnitKey]:
    """The key of a lane core or a relay (``rows``: its kept rows, ``source_codes``: the keyed IR's codes of a list
    of sources), None for any other node."""
    outputs = getattr(node, LANE_OUTPUTS, None)
    if outputs is not None:
        name = getattr(node, LANE_SOURCE_NAME, None)
        owner = ("lane", str(name)) if name is not None else ("lane-reduce", node.perceptron_index)
        channels = np.asarray(outputs, dtype=np.int64)
        return LANE, [owner] * len(channels), channels
    if not getattr(node, RELAY_MARKER, False):
        return None
    matrix = np.asarray(node.get_core_matrix(graph))[rows]
    owners: List[Tuple[Any, ...]] = []
    for column in range(matrix.shape[1]):
        read = np.flatnonzero(matrix[:, column] != 0)
        if len(read) != 1:
            return None
        owners.append(("relay", int(source_codes([sources[int(rows[read[0]])]])[0])))
    return RELAY, owners, np.zeros(len(owners), dtype=np.int64)
