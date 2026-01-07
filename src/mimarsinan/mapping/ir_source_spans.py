from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from mimarsinan.mapping.ir import IRSource


IRSourceKind = Literal["off", "input", "on", "node"]


@dataclass(frozen=True)
class IRSourceSpan:
    """
    A contiguous span of IRSource objects in destination order (axon idx / flat idx),
    where the source indices are also contiguous (stride=1).

    This enables efficient gather via slicing from:
      - input spike train (node_id == -2)
      - another node's spike train cache (node_id >= 0)
    """

    kind: IRSourceKind
    src_node_id: int
    src_start: int
    length: int
    dst_start: int

    @property
    def dst_end(self) -> int:
        return int(self.dst_start + self.length)

    @property
    def src_end(self) -> int:
        return int(self.src_start + self.length)


def _classify(src: IRSource) -> tuple[IRSourceKind, int, int]:
    if src.is_off():
        return ("off", -1, 0)
    if src.is_input():
        return ("input", -2, int(src.index))
    if src.is_always_on():
        return ("on", -3, 0)
    return ("node", int(src.node_id), int(src.index))


def compress_ir_sources(sources: Sequence[IRSource] | Iterable[IRSource]) -> list[IRSourceSpan]:
    """
    Compress a list of per-index IRSource objects into contiguous spans where possible.
    """
    if not isinstance(sources, Sequence):
        sources = list(sources)

    spans: list[IRSourceSpan] = []
    i = 0
    n = len(sources)
    while i < n:
        kind, src_node_id, src_start = _classify(sources[i])
        dst_start = i

        length = 1
        prev_src = src_start
        while (i + length) < n:
            nk, nnid, nstart = _classify(sources[i + length])
            if nk != kind:
                break
            if kind in ("node", "input"):
                if nnid != src_node_id:
                    break
                if nstart != (prev_src + 1):
                    break
                prev_src = nstart
            # "on"/"off" can always extend
            length += 1

        spans.append(
            IRSourceSpan(
                kind=kind,
                src_node_id=int(src_node_id),
                src_start=int(src_start),
                length=int(length),
                dst_start=int(dst_start),
            )
        )
        i += length

    return spans


