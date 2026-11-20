"""[U-ENERGY classes 1 and 2] The load censuses of the one load law (``cost/absolute/walls/loads.py``).

A load's census is the plane's passes with their producer writes left out (they are per inference, never a load), the
write policy at K = 1 and serial, and either the deployment's first load (``every_core``: every core of every pass
written once from an empty chip) or the program's own period census (the reprogramming per round). One census decides
a load's writes and its connectivity entries: each pass's entries are those of exactly the cores the census writes
(``PassCensus.connectivity_entries``), so a period census that rewrites nothing counts none. A plane builds its two
censuses only when a pricer reads them (``LoadInputs``), so a caller that prices no load pays nothing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Mapping, Optional, Tuple

from mimarsinan.deployment_record.build.program_census import record_program_passes
from mimarsinan.deployment_record.build.write_census import ProgramPasses, WriteCensusError
from mimarsinan.deployment_record.quantities import write_technology as wt
from mimarsinan.deployment_record.quantities.write_law import PlaneWrites, plane_write_census, write_policy_of

#: Why the candidate plane states no connectivity entries (a sealed record states them per core).
CANDIDATE_NO_ENTRIES = "the candidate plane counts no connectivity entries (a sealed record does)"
#: Why a sealed record's census states no entries (a built record states them per core, ``program_census``).
RECORD_NO_ENTRIES = "the record states no connectivity entries per core"


@dataclass(frozen=True)
class LoadCensus:
    """One load's census: the write law's inputs at K = 1, serial (``writes``), or why the plane states none
    (``unavailable``); and the connectivity entries of exactly the cores it writes (None: the plane states no entries
    per core, ``entries_absent`` its reason)."""

    writes: Optional[PlaneWrites]
    unavailable: str = ""
    entries: Optional[float] = None
    entries_absent: str = ""
    #: [m-2] What the census reads in place of the plane's own totals (a sealed record's segment totals that differ
    #: from its cores' sums), named in the load's basis.
    notes: Tuple[str, ...] = ()


def census_entries(census: Any) -> Optional[int]:
    """The connectivity entries a census writes over its passes, or None where a pass states none per core."""
    entries = [p.connectivity_entries for p in census.passes]
    return None if any(e is None for e in entries) else sum(entries)


def load_census(passes: Optional[ProgramPasses], platform: Mapping[str, Any], *, weight_bits: Any, every_core: bool,
                label: str, entries_absent: str = "", notes: Tuple[str, ...] = ()) -> LoadCensus:
    """One load's census (module docstring): ``every_core`` the deployment's first load, else the period census."""
    if passes is None:
        return LoadCensus(None, f"the {label} plane states no pass structure, so it has no load census",
                          None, entries_absent)
    rows = tuple(replace(row, producer_cells=0, producer_cores=frozenset(), producer_core_cells=None)
                 for row in passes.rows)
    program = replace(passes, rows=rows, operands_unstated=None)
    policy = write_policy_of({**platform, "inference_batch_K": 1})
    assert policy is not None
    policy = replace(policy, overlap_mode="serial", producer_write_source=None, payload_bits=None)
    census, reason = plane_write_census(program, platform, weight_bits=weight_bits, timesteps=None,
                                        every_core=every_core,
                                        label=f"{label} tiles, load" if every_core else f"{label} tiles, period")
    split = None if census is None or policy.core_technologies is None else wt.split_by_technology(
        program, census, policy, weight_bits=int(weight_bits))
    entries = None if census is None else census_entries(census.census)
    return LoadCensus(PlaneWrites(policy, census, reason, split), "", None if entries is None else float(entries),
                      "" if entries is not None else entries_absent, notes)


@dataclass(frozen=True)
class LoadInputs:
    """A plane's two load censuses (the deployment's first load, the reprogramming per round), each built when first
    read; ``unavailable``: why the plane's passes cannot be stated (both loads refuse with it)."""

    passes: Optional[ProgramPasses]
    platform: Mapping[str, Any]
    weight_bits: Any
    label: str
    entries_absent: str = ""
    unavailable: str = ""
    #: [m-2] What both censuses read in place of the plane's own totals (``LoadCensus.notes``).
    notes: Tuple[str, ...] = ()

    def _census(self, every_core: bool) -> LoadCensus:
        if self.unavailable:
            return LoadCensus(None, self.unavailable)
        return load_census(self.passes, self.platform, weight_bits=self.weight_bits, every_core=every_core,
                           label=self.label, entries_absent=self.entries_absent, notes=self.notes)

    @cached_property
    def first(self) -> LoadCensus:
        return self._census(True)

    @cached_property
    def period(self) -> LoadCensus:
        return self._census(False)


def load_inputs(passes: Optional[ProgramPasses], platform: Mapping[str, Any], *, weight_bits: Any, label: str,
                entries_absent: str = "") -> LoadInputs:
    """Both load censuses of a plane's passes, each with the entries of the cores it writes."""
    return LoadInputs(passes, platform, weight_bits, label, entries_absent)


#: [m-2] A segment's sealed totals that its cores' fields sum (the loads read the cores).
SEALED_TOTALS = ("connectivity_entries", "params_bytes")


def sealed_total_disagreements(record: Any) -> Tuple[str, ...]:
    """[m-2] Each segment whose sealed totals differ from its cores' sums, by name: a record's loads are priced by
    its cores (``record_program_passes``), so a total the cores contradict is named in each load's basis, never
    read."""
    out = []
    for segment in record.schedule.segments():
        differ = [(name, int(getattr(segment, name)), sum(int(getattr(c, name)) for c in segment.cores))
                  for name in SEALED_TOTALS]
        differ = [d for d in differ if d[1] != d[2]]
        if differ:
            out.append(f"segment {segment.segment_index} pass {segment.pass_index}: its sealed totals ("
                       + ", ".join(f"{name} {total}" for name, total, _ in differ) + ") differ from its cores' sums ("
                       + ", ".join(f"{name} {cores}" for name, _, cores in differ) + "); the load reads the cores")
    return tuple(out)


def record_load_inputs(record: Any) -> LoadInputs:
    """A sealed record's two load censuses: its passes state each core's connectivity entries
    (``record_program_passes``), so each census counts the entries of the cores it writes; [m-2] a segment whose
    sealed totals differ from its cores' sums is named in each load's basis (``sealed_total_disagreements``)."""
    platform, bits = record.identity.platform, record.utilization.crossbar.weight_bits
    try:
        passes = record_program_passes(record)
    except (ValueError, WriteCensusError) as refusal:  # a record whose passes cannot be stated
        reason = f"the record's passes cannot be stated: {refusal}"
        return LoadInputs(None, platform, bits, "record", unavailable=reason)
    return LoadInputs(passes, platform, bits, "record", RECORD_NO_ENTRIES, notes=sealed_total_disagreements(record))


def period_programming(loads: LoadInputs) -> Tuple[Optional[Tuple[Tuple[int, Optional[int]], ...]], str]:
    """Per pass of the period census, (payload bytes, connectivity entries) it rewrites per round, or None with why
    the plane states no period census: what every reader of a load's per-pass writes reads (one rule)."""
    period = loads.period
    if period.writes is None or period.writes.census is None:
        return None, period.unavailable or (period.writes.unavailable if period.writes is not None else "") or ""
    return tuple((int(p.payload_bytes), p.connectivity_entries) for p in period.writes.census.census.passes), ""
