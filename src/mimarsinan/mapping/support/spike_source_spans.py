from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

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


SPIKE_SOURCES_DENSE_TAG = "spike-sources-dense-v1"
SPIKE_SOURCES_SPANS_TAG = "spike-sources-spans-v1"

# Pickled cost is ~35 B/span vs ~9 B/source: spans win only when runs average >= 4.
_SPANS_MIN_AVG_RUN_LENGTH = 4


def span_round_trip_exact(s: SpikeSource) -> bool:
    """Whether compress→expand reproduces ``s`` field-for-field (``_classify`` normalizes everything else)."""
    if s.is_off_:
        return (
            int(s.core_) == -1 and int(s.neuron_) == 0
            and not s.is_input_ and not s.is_always_on_
        )
    if s.is_input_:
        return int(s.core_) == -2 and not s.is_always_on_
    if s.is_always_on_:
        return int(s.core_) == -3 and int(s.neuron_) == 0
    return True


def _encode_spike_sources_dense(sources: Sequence[SpikeSource]) -> tuple:
    n = len(sources)
    cores = np.fromiter((int(s.core_) for s in sources), dtype=np.int32, count=n)
    neurons = np.fromiter((int(s.neuron_) for s in sources), dtype=np.int32, count=n)
    flags = np.fromiter(
        (
            (1 if s.is_input_ else 0)
            | (2 if s.is_off_ else 0)
            | (4 if s.is_always_on_ else 0)
            for s in sources
        ),
        dtype=np.uint8,
        count=n,
    )
    return (SPIKE_SOURCES_DENSE_TAG, cores, neurons, flags)


def _decode_spike_sources_dense(cores, neurons, flags) -> list[SpikeSource]:
    return [
        SpikeSource(
            core, neuron,
            is_input=bool(flag & 1), is_off=bool(flag & 2), is_always_on=bool(flag & 4),
        )
        for core, neuron, flag in zip(cores.tolist(), neurons.tolist(), flags.tolist())
    ]


def encode_spike_sources_packed(sources: Sequence[SpikeSource]) -> tuple:
    """Tagged pickle payload: spans when lossless AND compressive, dense (always lossless) otherwise."""
    if not isinstance(sources, Sequence):
        sources = list(sources)
    if all(span_round_trip_exact(s) for s in sources):
        spans = compress_spike_sources(sources)
        if len(spans) * _SPANS_MIN_AVG_RUN_LENGTH <= len(sources):
            return (SPIKE_SOURCES_SPANS_TAG, tuple(spans))
    return _encode_spike_sources_dense(sources)


def decode_spike_sources_packed(payload: tuple) -> list[SpikeSource]:
    """Rebuild the exact SpikeSource list from a packed payload; unknown tags fail loud."""
    tag = payload[0] if payload else None
    if tag == SPIKE_SOURCES_SPANS_TAG:
        return expand_spike_source_spans(payload[1])
    if tag == SPIKE_SOURCES_DENSE_TAG:
        _, cores, neurons, flags = payload
        return _decode_spike_sources_dense(cores, neurons, flags)
    raise ValueError(f"Unknown spike-source pickle payload tag: {tag!r}")

