"""Candidate-time NoC estimate over shape-only fragments — the wireload model.

Mirrors the record's trace conventions exactly (the geometry SSOT is shared):
one message per (firing source neuron, destination core); XY routing hops =
Manhattan distance between the endpoint tiles; input and always-on somas live
ON their consumer core, so their messages are intra-tile input-path traffic;
on-wires (bias axons, always-on cells) fire every active cycle, everything
else at the DECLARED activity factor. Cross-pass producer/consumer pairs ride
the carry (DMA), never the mesh, and are excluded by pass membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from mimarsinan.chip_simulation.sanafe.noc_geometry import (
    tile_and_local_of_core,
    xy_of_tile,
    xy_route_hops,
)


@dataclass(frozen=True)
class NocEstimate:
    """Modeled NoC census of one candidate program (all passes summed)."""

    total_packets: float
    inter_tile_packets: float
    intra_tile_packets: float
    input_path_packets: float
    total_hops: float


def estimate_noc(
    *,
    fragments: Any,
    cores_per_tile: int,
    mesh_height: int,
    activity_factor: float,
    timesteps: int,
) -> NocEstimate:
    """Price the fragments' traffic on the resolved floorplan.

    ``fragments`` duck-types ``LayoutNocFragments`` (``pass_placements`` +
    ``census``). Refuses an undeclared activity factor or window: a partial
    estimate would read as a full one.
    """
    activity = float(activity_factor)
    steps = int(timesteps)
    if activity <= 0.0:
        raise ValueError(
            "estimate_noc needs a declared activity_factor > 0; spike-dependent "
            "traffic may not rest on an assumption nobody stated"
        )
    if steps <= 0:
        raise ValueError("estimate_noc needs timesteps > 0")

    census = fragments.census
    inter = intra = input_path = hops = 0.0

    for placements in fragments.pass_placements:
        core_of: Dict[int, int] = {
            int(softcore): int(hardcore) for softcore, hardcore in placements
        }
        tile_xy: Dict[int, Any] = {}

        def _xy(hardcore: int):
            found = tile_xy.get(hardcore)
            if found is None:
                tile, _local = tile_and_local_of_core(hardcore, cores_per_tile)
                found = xy_of_tile(tile, mesh_height)
                tile_xy[hardcore] = found
            return found

        for (producer, consumer), wires in census.pair_wires.items():
            src = core_of.get(int(producer))
            dst = core_of.get(int(consumer))
            if dst is None:
                continue  # this pass does not consume the wire at all
            if src is None:
                # [E5] Cross-pass: the producer ran in an earlier pass, so this
                # wire comes back over the host boundary and enters at its
                # consumer's own core — input traffic of THIS pass, never a
                # mesh crossing (the DMA terms price the boundary itself).
                carried = float(wires) * activity * steps
                intra += carried
                input_path += carried
                continue
            messages = float(wires) * activity * steps
            distance = xy_route_hops(_xy(src), _xy(dst))
            if distance == 0:
                intra += messages
            else:
                inter += messages
                hops += messages * distance
        for softcore in core_of:
            local_input = float(census.input_wires[softcore]) * activity * steps
            local_on = float(census.on_wires[softcore]) * steps
            intra += local_input + local_on
            input_path += local_input + local_on

    return NocEstimate(
        total_packets=inter + intra,
        inter_tile_packets=inter,
        intra_tile_packets=intra,
        input_path_packets=input_path,
        total_hops=hops,
    )
