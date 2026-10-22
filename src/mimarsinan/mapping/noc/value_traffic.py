"""[CH-D2] Value-domain NoC traffic: every inter-core wire of an execution stage carries one value.

A value-domain (MVM) core emits each output once per inference, so the traffic on the mesh is a
fact of the placement and the wire census, never of a spiking activity factor: one value of
``activation_bits / 8`` bytes per distinct wire, over the XY hops between the endpoint tiles. The
geometry is the SANA-FE SSOT the spike estimator and the trace analysis share.

Boundary (stated): a placement lists one (softcore, hardcore) pair per fragment and the wire census
counts wires per softcore pair, so a neuron-split softcore's wires are counted at the core of its
last-listed fragment (the approximation ``chip_simulation/sanafe/noc_estimate.py`` makes). The
census holds no per-fragment neuron range to distribute them by, so the byte-hops of a program with
split softcores are an estimate at that placement, not a per-fragment count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from mimarsinan.chip_simulation.sanafe.noc_geometry import (
    tile_and_local_of_core,
    xy_of_tile,
    xy_route_hops,
)
from mimarsinan.common.refusal import ToolchainRefusal


class ValueNocCensusError(ToolchainRefusal, ValueError):
    """A value NoC census asked for at a value grid that does not exist."""


def value_noc_wire_hops(
    *,
    pair_wires: Any,
    stage_placements: Sequence[Sequence[Tuple[int, int]]],
    cores_per_tile: int,
    mesh_height: int,
) -> int:
    """Value-hops per inference: Σ over the wires whose producer and consumer run in the same
    execution stage of ``wires x XY hops`` (one value per wire, whatever its width).

    A wire between stages (another pass, another segment) crosses the chip boundary, which the carry
    and host terms price; a wire inside one tile travels no inter-tile link.
    """
    into: Dict[int, List[Tuple[int, int]]] = {}
    for (producer, consumer), wires in pair_wires.items():
        into.setdefault(int(consumer), []).append((int(producer), int(wires)))
    value_hops = 0
    for placements in stage_placements:
        core_of: Dict[int, int] = {int(softcore): int(hardcore) for softcore, hardcore in placements}
        for consumer, dst in core_of.items():
            for producer, wires in into.get(consumer, ()):
                src = core_of.get(producer)
                if src is None:
                    continue
                value_hops += wires * xy_route_hops(
                    xy_of_tile(tile_and_local_of_core(src, cores_per_tile)[0], mesh_height),
                    xy_of_tile(tile_and_local_of_core(dst, cores_per_tile)[0], mesh_height),
                )
    return value_hops


def value_noc_bytes(
    *,
    pair_wires: Any,
    stage_placements: Sequence[Sequence[Tuple[int, int]]],
    cores_per_tile: int,
    mesh_height: int,
    activation_bits: int,
) -> float:
    """Byte-hops per inference: the value-hops of :func:`value_noc_wire_hops` x ``activation_bits / 8``."""
    bits = int(activation_bits)
    if bits <= 0:
        raise ValueNocCensusError(f"value_noc_bytes needs activation_bits > 0, got {bits}")
    return value_noc_wire_hops(pair_wires=pair_wires, stage_placements=stage_placements,
                               cores_per_tile=cores_per_tile, mesh_height=mesh_height) * bits / 8.0
