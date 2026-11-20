"""[U-TGAUGE3, U-TGAUGE4] The timing walk's region reads by the placement rule: each element of a segment's region node
read as the deployed program reads it, from its producer.

A timing segment's region node (a layout between perceptrons: a concatenation, a flatten) holds every element as its
producer emitted it. The deployed program gathers each output element from its producer's buffer: a core's value
complete in its own window at its own gauge (``chip_simulation/ttfs/ttfs_segment.py``), an outside producer's (a host
op's output, the raw input, an earlier segment's output) in that producer's own buffer. So the region's reading is its
own element movement applied to its producers' values, each producer read by itself: a perceptron of the segment as its
own decoded value (its own window, its own gauge along its channel axis), an outside producer as its value. Before
U-TGAUGE3 the walk decoded a region node at its first producing source's scale; before U-TGAUGE4 it accumulated the
region over the region's own window (from its deepest producer's depth), so a shallower producer's elements were read
late by the depth gap, and a spike before that window was lost.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from mimarsinan.mapping.support.gauge_placement import GaugePlacementError
from mimarsinan.spiking.segment_partition import perceptron_of

__all__ = ["region_read", "region_producers"]


def region_producers(driver: Any, node: Any) -> list:
    """The perceptrons of ``node``'s segment whose elements ``node`` holds (through the layouts of the segment between
    them), the producers ``region_read`` reads by themselves."""
    segment = driver._seg_of.get(node)
    found: list = []
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen or driver._seg_of.get(n) is not segment:
            continue
        seen.add(id(n))
        if perceptron_of(n) is not None:
            found.append(n)
        else:
            stack.extend(driver._deps.get(n, []))
    return found


def region_read(driver: Any, node: Any, producer_value: Callable[[Any], torch.Tensor]) -> torch.Tensor:
    """``node``'s values (batched), each element as its producer holds it: ``producer_value(n)`` for a perceptron of
    the segment (its decoded value) and for a node outside the segment (its value), moved by the layouts between."""
    segment = driver._seg_of.get(node)
    memo: dict = {}

    def read(n: Any) -> torch.Tensor:
        if n in memo:
            return memo[n]
        if driver._seg_of.get(n) is not segment or perceptron_of(n) is not None:
            held = producer_value(n)
        elif getattr(n, "moves_elements_only", False):
            parts = [read(d) for d in driver._deps.get(n, [])]
            held = n.forward(parts[0] if len(parts) == 1 else tuple(parts))
        else:
            raise GaugePlacementError(
                f"{type(n).__name__} {getattr(n, 'name', None)!r} computes inside a timing segment without being a "
                f"perceptron or a layout, so its elements hold no producer's value")
        memo[n] = held
        return held

    return read(node)
