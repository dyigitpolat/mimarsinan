"""SANA-FE trace, energy, and connectivity analysis helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeArchGeometry,
    SanafeNocLink,
    SanafeNocLinkLoad,
)


def _flatten_message_trace(message_trace: Any) -> Optional[List[dict]]:
    """Flatten per-cycle message lists; drop placeholder entries."""
    if not message_trace:
        return None
    flat: List[dict] = []
    for events in message_trace:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("placeholder"):
                    continue
                flat.append({k: (float(v) if isinstance(v, float) else v)
                             for k, v in ev.items()})
    return flat or None


def _compute_tile_packets_per_cycle(
    message_trace: Any,
) -> List[Dict[int, int]]:
    """Per-cycle packet count per destination ``tile_id``."""
    if not message_trace:
        return []
    out: List[Dict[int, int]] = []
    for events in message_trace:
        bins: Dict[int, int] = {}
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            dt = int(ev.get("dest_tile_id", -1))
            if dt < 0:
                continue
            bins[dt] = bins.get(dt, 0) + 1
        out.append(bins)
    return out

def _aggregate_noc_links(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLink]:
    """Aggregate cross-tile message trace into directed NoC links."""
    if not message_trace:
        return []
    bins: Dict[Tuple[int, int], Dict[str, int]] = {}
    src_coord: Dict[int, Tuple[int, int]] = {}
    dst_coord: Dict[int, Tuple[int, int]] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            src_t = int(ev.get("src_tile_id", -1))
            dst_t = int(ev.get("dest_tile_id", -1))
            if src_t < 0 or dst_t < 0 or src_t == dst_t:
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            src_coord[src_t] = (sx, sy)
            dst_coord[dst_t] = (dx, dy)
            slot = bins.setdefault(
                (src_t, dst_t),
                {"packets": 0, "spikes": 0, "hops": 0},
            )
            slot["packets"] += 1
            slot["spikes"] += int(ev.get("spikes", 0) or 0)
            slot["hops"] += int(ev.get("hops", 0) or 0)
    out: List[SanafeNocLink] = []
    for (src_t, dst_t), b in sorted(bins.items()):
        sx, sy = src_coord.get(src_t, (-1, -1))
        dx, dy = dst_coord.get(dst_t, (-1, -1))
        if geom is not None:
            if (sx < 0 or sy < 0) and 0 <= src_t < len(geom.tiles_xy):
                sx, sy = geom.tiles_xy[src_t]
            if (dx < 0 or dy < 0) and 0 <= dst_t < len(geom.tiles_xy):
                dx, dy = geom.tiles_xy[dst_t]
        out.append(SanafeNocLink(
            src_tile=src_t, dst_tile=dst_t,
            src_x=int(sx), src_y=int(sy),
            dst_x=int(dx), dst_y=int(dy),
            packet_count=int(b["packets"]),
            spike_count=int(b["spikes"]),
            total_hops=int(b["hops"]),
        ))
    return out


def _aggregate_noc_link_load(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLinkLoad]:
    """Per-mesh-edge packet load via XY routing."""
    if not message_trace:
        return []
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            cx, cy = sx, sy
            step_x = 1 if dx > sx else -1 if dx < sx else 0
            step_y = 1 if dy > sy else -1 if dy < sy else 0
            while cx != dx:
                nx = cx + step_x
                k = (cx, cy, nx, cy)
                counts[k] = counts.get(k, 0) + 1
                cx = nx
            while cy != dy:
                ny = cy + step_y
                k = (cx, cy, cx, ny)
                counts[k] = counts.get(k, 0) + 1
                cy = ny
    out: List[SanafeNocLinkLoad] = []
    for (fx, fy, tx, ty), n in sorted(counts.items()):
        out.append(SanafeNocLinkLoad(
            from_x=fx, from_y=fy, to_x=tx, to_y=ty, packet_count=int(n),
        ))
    return out

def _compute_noc_traffic_per_cycle(message_trace: Any) -> List[List[List[int]]]:
    """Per-cycle ``[src_x, src_y, dst_x, dst_y, count]`` quintuples."""
    if not message_trace:
        return []
    out: List[List[List[int]]] = []
    for events in message_trace:
        bins: Dict[Tuple[int, int, int, int], int] = {}
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            if sx == dx and sy == dy:
                continue
            k = (sx, sy, dx, dy)
            bins[k] = bins.get(k, 0) + 1
        out.append([[fx, fy, tx, ty, n] for (fx, fy, tx, ty), n in bins.items()])
    return out
