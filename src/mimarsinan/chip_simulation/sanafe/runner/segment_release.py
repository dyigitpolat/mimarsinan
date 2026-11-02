"""[U-SFR] What a SANA-FE run in flat memory does per neural segment: its spike trace taken as text, the run's totals
over the segments it keeps, and the allocator's freed pages handed back once a segment's chip, network, traces and
record are gone.

A segment's working set (its chip, its network, SANA-FE's spike and message traces) is freed when the segment ends,
but glibc keeps freed heap pages resident, charged to the run's memory class, until ``malloc_trim`` returns them
(measured on a 1,024-core segment: most of the freed working set stayed resident until then). Where the C library
has no ``malloc_trim`` the release is a no-op; nothing simulated reads it.

SANA-FE's memory spike trace holds every spike twice at the end of ``chip.sim`` (a C++ address and a Python address
object). Given a file-like object instead, SANA-FE writes the same spikes in the same order, one
``<group>.<offset>,<timestep>`` line each and one ``write`` per timestep; :class:`CompactSpikeTrace` keeps each
timestep's ``<group>.<offset>`` lines as one string, and its :meth:`CompactSpikeTrace.trace` reads them back per
timestep as the strings ``str()`` gives for SANA-FE's own address objects.
"""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.sanafe.records import SanafeEnergyBreakdown


class SpikeTraceView:
    """A spike trace's timesteps, each read as a list of ``<group>.<offset>`` strings when reached (the last read one
    kept, so a reader that reads one timestep per core splits it once)."""

    def __init__(self, steps: List[str]):
        self._steps = steps
        self._last: Tuple[int, List[str]] = (-1, [])

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[List[str]]:
        for t in range(len(self._steps)):
            yield self[t]

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return SpikeTraceView(self._steps[index])
        t = range(len(self._steps))[index]
        if self._last[0] != t:
            text = self._steps[t]
            self._last = (t, text.split("\n")[:-1] if text else [])
        return self._last[1]


class CompactSpikeTrace:
    """The file-like object SANA-FE writes its spike trace into: one string per timestep, its timestep column cut."""

    def __init__(self) -> None:
        self._steps: List[str] = []

    def write(self, text: str) -> int:
        if text:
            cut = text.find(",")
            suffix = text[cut:text.find("\n", cut) + 1]
            names = text.replace(suffix, "\n")
            if "," in names or names.count("\n") != text.count("\n") or not text.endswith("\n"):
                raise ValueError(f"a spike trace write holds more than one timestep or an unknown line form: "
                                 f"{text[:200]!r}")
            self._steps.append(names)
        else:
            self._steps.append("")
        return len(text)

    def flush(self) -> None:
        return None

    def trace(self) -> SpikeTraceView:
        return SpikeTraceView(self._steps)

_TRIM: list[Optional[Callable[[int], int]]] = []


def _malloc_trim() -> Optional[Callable[[int], int]]:
    if not _TRIM:
        try:
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)  # the process's own C library
        except OSError:
            trim = None
        _TRIM.append(trim)
    return _TRIM[0]


def release_freed_heap() -> None:
    """Return the C allocator's free pages to the OS (glibc ``malloc_trim(0)``; elsewhere nothing)."""
    trim = _malloc_trim()
    if trim is not None:
        trim(0)


def segment_totals(segments: Mapping[int, Any]) -> Tuple[SanafeEnergyBreakdown, float, int, int]:
    """The run's energy, longest simulated time, spikes and packets over its segments (full or persisted)."""
    agg_e = SanafeEnergyBreakdown.zero()
    max_sim_time = 0.0
    total_spikes = 0
    total_packets = 0
    for seg in segments.values():
        agg_e = agg_e.add(seg.energy)
        if seg.sim_time_s > max_sim_time:
            max_sim_time = seg.sim_time_s
        total_spikes += seg.spikes
        total_packets += seg.packets_sent
    return agg_e, max_sim_time, total_spikes, total_packets
