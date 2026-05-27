from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from mimarsinan.code_generation.cpp_chip_model import SpikeSource


SpikeSourceKind = Literal["off", "input", "on", "core"]


@dataclass(frozen=True)
class SpikeSourceSpan:
    """
    A contiguous span of SpikeSources in *destination* index order (axon idx / output idx),
    where the *source* indices are also contiguous (stride=1).

    This enables efficient gather via slicing:
      - input:   input[:, src_start:src_start+length]
      - core:    buffers[src_core][:, src_start:src_start+length]
      - on/off:  fill 1/0
    """

    kind: SpikeSourceKind
    # For kind=="core", the producing core id; otherwise a sentinel (-1/-2/-3).
    src_core: int
    # For kind in {"input","core"}: starting neuron/index in the source.
    # For kind in {"off","on"}: always 0.
    src_start: int
    # Number of elements in this span.
    length: int
    # Starting destination index in the flattened sources list.
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

        # Extend while:
        # - same kind and (for core/input) same src_core
        # - next src index increments by 1 for core/input
        # - for on/off: identical kind (src index irrelevant)
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
            # "on"/"off" can always extend as long as kind matches.
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
    """
    Expand spans back into a list of SpikeSource objects.

    NOTE: This is primarily for compatibility (e.g., codegen). Simulation should prefer spans.
    """
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


