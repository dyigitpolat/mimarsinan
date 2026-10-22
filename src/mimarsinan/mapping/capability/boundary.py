"""Boundary elements of a placed IR: distinct activation elements crossing between chip and host per inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.capability.ladder import CHIP, HOST, INPUT_NODE, TRANSPARENT, CensusNode

_SIDE_CODE = {CHIP: 1, HOST: 0}
_SHIFT = np.int64(32)


@dataclass(frozen=True)
class BoundaryElements:
    """Image entry, host-op boundaries in each direction, and the readout of on-chip outputs."""

    entry: int
    carry_in: int
    carry_out: int
    readout: int

    @property
    def total(self) -> int:
        return self.entry + self.carry_in + self.carry_out + self.readout


def _keys(ids: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return ((ids.astype(np.int64) + 4) << _SHIFT) | indices.astype(np.int64)


def _key_nodes(keys: np.ndarray) -> np.ndarray:
    return (keys >> _SHIFT) - 4


class _Resolver:
    """Element keys behind each source, wiring passed through: a transparent node stands for all of its inputs."""

    def __init__(self, nodes: Sequence[CensusNode]):
        self.side_code: Dict[int, int] = {INPUT_NODE: _SIDE_CODE[HOST]}
        self.through: Dict[int, np.ndarray] = {}
        for node in nodes:
            if node.side == TRANSPARENT:
                self.through[node.id] = np.unique(self.resolve(node.source_ids, node.source_indices))
            else:
                self.side_code[node.id] = _SIDE_CODE[node.side]

    def resolve(self, ids: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if not self.through:
            return _keys(ids, indices)
        passing = np.isin(ids, np.fromiter(self.through, dtype=np.int64))
        parts = [_keys(ids[~passing], indices[~passing])]
        parts.extend(self.through[int(t)] for t in np.unique(ids[passing]).tolist())
        return np.concatenate(parts)

    def producer_on_chip(self, keys: np.ndarray) -> np.ndarray:
        nodes = _key_nodes(keys)
        unique, inverse = np.unique(nodes, return_inverse=True)
        codes = np.array([self.side_code[int(n)] for n in unique.tolist()], dtype=np.int8)
        return codes[inverse] == 1


def boundary_elements(
    nodes: Sequence[CensusNode], output_sources: Tuple[np.ndarray, np.ndarray],
) -> BoundaryElements:
    """Distinct (producer, element) crossings: input read on chip, host op to chip, chip to host op, chip to output."""
    resolver = _Resolver(nodes)
    entry: List[np.ndarray] = []
    carry_in: List[np.ndarray] = []
    carry_out: List[np.ndarray] = []
    for node in nodes:
        if node.side == TRANSPARENT or node.source_ids.size == 0:
            continue
        keys = resolver.resolve(node.source_ids, node.source_indices)
        on_chip = resolver.producer_on_chip(keys)
        if node.side == CHIP:
            crossing = keys[~on_chip]
            from_input = _key_nodes(crossing) == INPUT_NODE
            entry.append(crossing[from_input])
            carry_in.append(crossing[~from_input])
        else:
            carry_out.append(keys[on_chip])
    out_keys = resolver.resolve(*output_sources)
    readout = np.setdiff1d(out_keys[resolver.producer_on_chip(out_keys)], _union(carry_out))
    return BoundaryElements(
        entry=int(_union(entry).size), carry_in=int(_union(carry_in).size),
        carry_out=int(_union(carry_out).size), readout=int(np.unique(readout).size),
    )


def _union(parts: List[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))
