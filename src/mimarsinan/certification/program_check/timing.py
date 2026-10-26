"""[W15] The pass-cut conditions on a deployment against its identity program: identity-program latency (H1-latency),
pinned latencies (R1), delivery delays on every slice the stage loop reads late (R2), the scoped carry store and edge
classes (R3), per-slice input kinds and positions (R4), and the static part of window coverage (H2, Proposition 2).

Latencies are the executed ones (``executed_core_latencies``: ChipLatency, pins honoured, the stored latencies left
as they were); the identity program's are its own executed latencies per IR node, its segments its program stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from mimarsinan.certification.program_check.declaration import CheckDeclaration
from mimarsinan.certification.program_check.reads import segment_carried
from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.tiles import SweepFacts
from mimarsinan.certification.program_check.view import K_CORE, K_INPUT, ProgramView, UnitView
from mimarsinan.mapping.packing.executed_reads import CACHED, REPLAYED, SliceDelivery
from mimarsinan.mapping.packing.pass_timing import carried_nodes, executed_core_latencies
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES, ChipLatency
from mimarsinan.mapping.packing import pass_positions


@dataclass
class IdentityTiming:
    """Per IR node: its identity latency, its identity segment (program stage) and its level unit (or None)."""

    latency: Dict[int, int] = field(default_factory=dict)
    segment: Dict[int, int] = field(default_factory=dict)
    level: Dict[int, Optional[int]] = field(default_factory=dict)
    unit: Dict[int, int] = field(default_factory=dict)
    gaps: List[Tuple[int, int, int]] = field(default_factory=list)


def _executed(hcm) -> Optional[List[int]]:
    """``executed_core_latencies``, or None where ChipLatency refuses (R6 names the cycle)."""
    try:
        return executed_core_latencies(hcm)
    except (RuntimeError, ValueError):
        return None


def identity_timing(iota: ProgramView, facts: SweepFacts, *, latencies: bool) -> IdentityTiming:
    out = IdentityTiming()
    for u, unit in enumerate(iota.units):
        executed = (_executed(unit.stage.hard_core_mapping) if latencies else None) or [0] * len(unit.cores)
        for h, cv in enumerate(unit.cores):
            for p in cv.placements:
                out.latency[p.node] = int(executed[h])
                out.segment[p.node] = unit.program_index
                out.level[p.node] = unit.index if unit.level else None
                out.unit[p.node] = u
        if latencies:
            for h, cv in enumerate(unit.cores):
                live = facts.identity_cores[(u, h)].live & (cv.kinds == K_CORE)
                for src in sorted(set(cv.aux[live].tolist())):
                    gap = int(executed[h]) - int(executed[src])
                    if gap != 1:
                        out.gaps.append((u, h, gap))
    return out


def walked_latencies(hcm) -> Optional[List[Optional[int]]]:
    """Per core the latency the executors walk it at: its pin (``ChipLatency`` sets every pinned core and keeps the
    rest), else its stored latency, else what ``ChipLatency`` gives it at run time (the stored latencies restored);
    None where ``ChipLatency`` refuses (a cyclic core graph). A core left at None is skipped by every walk."""
    stored = [core.latency for core in hcm.cores]
    pins = getattr(hcm, PINNED_LATENCIES, None)
    if pins is not None:
        return [int(pins[i]) if i < len(pins) else stored[i] for i in range(len(stored))]
    if all(latency is not None for latency in stored):
        return list(stored)
    try:
        ChipLatency(hcm).calculate()
        return [core.latency for core in hcm.cores]
    except (RuntimeError, ValueError):
        return None
    finally:
        for core, latency in zip(hcm.cores, stored):
            core.latency = latency


def _residents(unit: UnitView, h: int) -> Set[int]:
    return {p.node for p in unit.cores[h].placements}


def check_timing(program: Any, delta: ProgramView, facts: SweepFacts, ident: IdentityTiming,
                 results: Dict[str, ConditionResult], deliveries: Dict[int, List[SliceDelivery]], *,
                 timed: bool) -> None:
    """H1-latency, R1 and R2 under timed semantics; R3 (carry scope and edge classes) on every program. R2 reads
    the delays the stage loop applies (``deliveries``): a carried slice's against the fused delay of every live
    reader, and any other delayed slice fails by name (the executors shift every delayed slice)."""
    h1l, r1, r2, r3 = (results[c] for c in ("H1-latency", "R1", "R2", "R3"))
    for u, unit in enumerate(delta.units):
        hcm = unit.stage.hard_core_mapping
        pins = getattr(hcm, PINNED_LATENCIES, None)
        if timed and pins is not None and len(pins) != len(hcm.cores):
            r1.add(Violation("R1", "pin_count", f"unit {unit.index} ({unit.stage.name})",
                             f"{len(pins)} pinned latencies for {len(hcm.cores)} cores"))
        executed = _executed(hcm) if timed else None
        timed_here = timed and executed is not None
        carried = carried_nodes(program.stages, unit.stage)
        seg_carried = segment_carried(program, unit)
        by_slice = {d.index: d for d in deliveries[unit.index]}
        fused: List[Optional[int]] = []
        segs: List[Optional[int]] = []
        for h in range(len(unit.cores)):
            nodes = _residents(unit, h)
            lats = {ident.latency[n] for n in nodes if n in ident.latency}
            seg = {ident.segment[n] for n in nodes if n in ident.segment}
            if len(seg) > 1:
                r3.add(Violation("R3", "mixed_segments", f"unit {unit.index} core {h}",
                                 f"the core holds neurons of identity segments {sorted(seg)}"))
            segs.append(seg.pop() if len(seg) == 1 else None)
            if timed:
                h1l.checked += 1
                if len(lats) > 1:
                    h1l.add(Violation("H1-latency", "co_resident_latencies", f"unit {unit.index} core {h}",
                                      f"residents {sorted(nodes)} run at identity latencies {sorted(lats)}"))
            fused.append(lats.pop() if len(lats) == 1 else None)
        for h, cv in enumerate(unit.cores):
            live = facts.cores[(u, h)].live
            where = f"unit {unit.index} ({unit.stage.name}) core {h}"
            for src in sorted(set(cv.aux[live & (cv.kinds == K_CORE)].tolist())):
                r3.checked += 1
                if segs[h] is not None and segs[src] is not None and segs[h] != segs[src]:
                    r3.add(Violation("R3", "boundary_edge_in_unit", where, f"reads core {src} in-unit, an edge the "
                                     f"identity program crosses at a segment boundary"))
                lat_c, lat_p = fused[h], fused[src]
                if timed_here and executed is not None and lat_c is not None and lat_p is not None:
                    r1.checked += 1
                    got = int(executed[h]) - int(executed[src])
                    want = int(lat_c) - int(lat_p)
                    if got != want:
                        r1.add(Violation("R1", "pinned_latency", where, f"reads core {src} {got} cycle(s) apart as "
                                         f"executed, {want} in the fused segment"))
            for k in sorted(set(cv.aux[live & (cv.kinds == K_INPUT)].tolist())):
                node = unit.slices[k][0]
                in_segment = node in ident.segment and segs[h] is not None and ident.segment[node] == segs[h]
                is_carried = node in carried
                r3.checked += 1
                if not unit.level and is_carried != in_segment:
                    r3.add(Violation("R3", "carry_scope", where, f"slice {k} (node {node}) is "
                                     f"{'carried' if is_carried else 'encoded'} though the identity program reads it "
                                     f"{'inside its segment' if in_segment else 'across a segment boundary'}"))
                lat_c = fused[h]
                if timed and node in seg_carried and lat_c is not None and node in ident.latency:
                    r2.checked += 1
                    want = int(lat_c) - 1 - int(ident.latency[node])
                    got = by_slice[k].delay
                    if got != want:
                        r2.add(Violation("R2", "delivery_delay", where, f"slice {k} (node {node}) read at delay "
                                         f"{got} ({by_slice[k].kind}), the fused segment delivers it at {want}"))
        for d in deliveries[unit.index] if timed else ():
            if d.delay and d.node not in seg_carried:
                r2.checked += 1
                r2.add(Violation("R2", "delay_on_encoded_slice", f"unit {unit.index} ({unit.stage.name}) slice "
                                 f"{d.index}", f"node {d.node} is {d.kind}, not carried, yet the stage loop reads it "
                                 f"{-d.delay} cycle(s) late"))


def check_slices(program: Any, delta: ProgramView, iota: ProgramView, ident: IdentityTiming,
                 deliveries: Dict[int, List[SliceDelivery]], iota_deliveries: Dict[int, List[SliceDelivery]],
                 decl: CheckDeclaration, r4: ConditionResult) -> None:
    """R4 (G5's rule 4): every slice a unit reads that the identity's unit of its neurons reads too (a carried wire is
    in-unit there; H5-collapse reads it) is of the kind the identity reads it as (a cached train, a count re-encode,
    an encode: ``slice_deliveries``), and under a position-dependent encode (dither) every encoded one sits at that
    unit's channel offset (the executor's own fused positions, whose memo ``stale_memo`` holds to them). A unit
    holding neurons of two identity units fails by name (``multi_home_unit``), never a skip."""
    fused = {} if decl.encoder_position_free else pass_positions._fused_positions(program.stages)
    offsets: Dict[Tuple[int, int], int] = {}
    kinds: Dict[Tuple[int, int], str] = {}
    for u, unit in enumerate(iota.units):
        for node, offset, _ in unit.slices:
            offsets[(u, node)] = offset
        for d in iota_deliveries[unit.index]:
            kinds[(u, d.node)] = d.kind
    for unit in delta.units:
        where = f"unit {unit.index} ({unit.stage.name})"
        homes: Set[int] = {ident.unit[p.node] for cv in unit.cores for p in cv.placements if p.node in ident.unit}
        if len(homes) > 1:
            r4.add(Violation("R4", "multi_home_unit", where, f"holds neurons of identity units {sorted(homes)}: no "
                             "one identity unit's slice kinds and positions apply"))
            continue
        if not homes:
            continue
        home = next(iter(homes))
        key = pass_positions.stage_key(program.stages, unit.stage)
        moved = fused.get(key) if key is not None else None
        for d in deliveries[unit.index]:
            want_kind = kinds.get((home, d.node))
            if want_kind is None:          # in-unit in the identity: a carried wire, which H5-collapse reads
                continue
            r4.checked += 1
            if d.kind != want_kind:
                r4.add(Violation("R4", "slice_kind", f"{where} slice {d.index}", f"node {d.node} is read as "
                                 f"{d.kind}, the identity's unit reads it as {want_kind}"))
            if decl.encoder_position_free or d.kind in (REPLAYED, CACHED):
                continue
            fused_at = None if moved is None else moved.get(d.index)
            at = unit.slices[d.index][1] if fused_at is None else int(fused_at)
            want = offsets.get((home, d.node))
            if want is not None and at != want:
                r4.add(Violation("R4", "encode_position", f"{where} slice {d.index}",
                                 f"node {d.node} encoded at channel offset {at}, the identity stage encodes it at "
                                 f"{want}"))
    if decl.encoder_position_free:
        r4.note = "position-free encoder: the position clause holds by construction; slice kinds checked"
