from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from mimarsinan.code_generation.cpp_chip_model import SpikeSource


SpikeSourceKind = Literal["off", "input", "on", "core"]


@dataclass(frozen=True)
class SpikeSourceSpan:
    """Contiguous span of SpikeSources with stride-1 source indices in destination order."""

    kind: SpikeSourceKind
    src_core: int
    src_start: int
    length: int
    dst_start: int

    @property
    def dst_end(self) -> int:
        return int(self.dst_start + self.length)

    @property
    def src_end(self) -> int:
        return int(self.src_start + self.length)


def _classify(s: SpikeSource) -> tuple[SpikeSourceKind, int, int]:
    if s.is_off_:
        return ("off", -1, 0)
    if s.is_input_:
        return ("input", -2, int(s.neuron_))
    if s.is_always_on_:
        return ("on", -3, 0)
    return ("core", int(s.core_), int(s.neuron_))


def compress_spike_sources(sources: Sequence[SpikeSource] | Iterable[SpikeSource]) -> list[SpikeSourceSpan]:
    """
    Compress a list of per-index SpikeSource objects into contiguous spans where possible.
    """
    if not isinstance(sources, Sequence):
        sources = list(sources)

    spans: list[SpikeSourceSpan] = []
    i = 0
    n = len(sources)
    while i < n:
        kind, src_core, src_start = _classify(sources[i])
        dst_start = i

        length = 1
        prev_src = src_start
        while (i + length) < n:
            nk, ncore, nstart = _classify(sources[i + length])
            if nk != kind:
                break
            if kind in ("core", "input"):
                if ncore != src_core:
                    break
                if nstart != (prev_src + 1):
                    break
                prev_src = nstart
            length += 1

        spans.append(
            SpikeSourceSpan(
                kind=kind,
                src_core=int(src_core),
                src_start=int(src_start),
                length=int(length),
                dst_start=int(dst_start),
            )
        )
        i += length

    return spans


def expand_spike_source_spans(spans: Sequence[SpikeSourceSpan]) -> list[SpikeSource]:
    """Expand spans back into a list of SpikeSource objects (compatibility path; simulation prefers spans)."""
    out: list[SpikeSource] = []
    for sp in spans:
        if sp.kind == "off":
            for _ in range(sp.length):
                out.append(SpikeSource(-1, 0, is_input=False, is_off=True))
        elif sp.kind == "on":
            for _ in range(sp.length):
                out.append(SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True))
        elif sp.kind == "input":
            for k in range(sp.length):
                out.append(SpikeSource(-2, int(sp.src_start + k), is_input=True, is_off=False))
        elif sp.kind == "core":
            for k in range(sp.length):
                out.append(SpikeSource(int(sp.src_core), int(sp.src_start + k), is_input=False, is_off=False))
        else:
            raise ValueError(f"Unknown span kind: {sp.kind}")
    return out


