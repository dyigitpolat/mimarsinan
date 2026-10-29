"""The pass timeline of an executed program with its write events, and the period's write summary.

Read on the record plane at build time: the built program's passes (``mapping_program_passes``) under the write law's
one period census (``plane_write_census``: the executed rotation) at the declared batch. Per pass: its cores, the cores
it keeps across periods, the static cells it rewrites per period and their bytes, the write units and write events at
the declared write granularity (``pass_units``, ``pass_events``; None where the granularity is undeclared), the operand
cells it writes per inference (the producer cells of a dynamic-operand pass) and their write events, the operators it
holds and the carry read out after it. Host stages sit between passes in stage order.

[U-C4 targeted M2] Written cells count each core type's cells: on a chip whose core types store a weight in different
numbers of cells, every cell count, write unit and write event is the sum of the census's shares by write technology,
each at its own count (``write_technology.cell_shares``, the shares the write terms are priced on); a census that has no
such split refuses by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import WriteCensus
from mimarsinan.deployment_record.quantities.write_law import (
    WritePolicy, pass_events, pass_units, plane_write_census, summed_cells, write_policy_of, write_quantities,
)
from mimarsinan.deployment_record.quantities.write_technology import (
    CellShare, TechnologySplit, cell_shares, split_by_technology,
)
from mimarsinan.mapping.capability.op_kinds import op_kind


class TimelineError(ToolchainRefusal, ValueError):
    """The executed program has no period census to summarize: no declared batch, or the census refused."""


@dataclass(frozen=True)
class ProgramWrites:
    """The executed program's period census at the declared batch, or the reason it has none."""

    policy: Optional[WritePolicy]
    census: Optional[WriteCensus]
    unavailable: Optional[str] = None
    #: [E-6, targeted pass; U-C4 targeted M2] N2b: the census split by write technology where the platform's core
    #: types declare several (``write_technology.split_by_technology``, the record plane's rule), else None.
    by_technology: Optional[TechnologySplit] = None


def program_writes(hybrid: Any, platform: Mapping[str, Any], *, weight_bits: Optional[int]) -> ProgramWrites:
    """The record plane's write census of ``hybrid`` on ``platform`` (which declares ``inference_batch_K``), its
    payload bytes at ``weight_bits`` (the caller reads the run's weight bits, the one-run reader set's rule), and its
    split by write technology where the core types declare several."""
    policy = write_policy_of(platform)
    passes = mapping_program_passes(hybrid)
    census, reason = plane_write_census(passes, platform, weight_bits=weight_bits, timesteps=None, label="record")
    split = None if census is None or policy is None or policy.core_technologies is None or weight_bits is None \
        else split_by_technology(passes, census, policy, weight_bits=int(weight_bits))
    return ProgramWrites(policy, census, reason, split)


def _shares(writes: ProgramWrites) -> Optional[Tuple[CellShare, ...]]:
    """The census in shares of one cell block per weight (``write_technology.cell_shares``), refused by name where it
    has none; without a declared batch the census at one cell per weight."""
    if writes.census is None:
        return None
    if writes.policy is None:
        return ((writes.census.census, None, writes.census.written_extents),)
    shares, reason = cell_shares(writes.census, writes.policy, writes.by_technology)
    if shares is None:
        raise TimelineError(str(reason))
    return shares


def _per(policy: Optional[WritePolicy]) -> int:
    return 1 if policy is None else policy.cells_per_weight


def _operators_of(stage: Any, operator_of: Mapping[int, str]) -> List[str]:
    seen: Dict[str, None] = {}
    for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
        for placement in placements:
            seen.setdefault(operator_of.get(int(placement["ir_node_id"]), str(placement["ir_node_id"])), None)
    return list(seen)


def _units(shares: Optional[Tuple[CellShare, ...]]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Per pass, the write units and write events summed over the shares; None where a share states none or the shares
    count units at two write granularities."""
    if not shares or any(policy is None for _, policy, _ in shares) or len(
            {policy.write_granularity for _, policy, _ in shares}) > 1:
        return None, None
    per = [pass_units(census, policy, extents) for census, policy, extents in shares]
    stated = [units for units in per if units is not None]
    if len(stated) != len(per):
        return None, None
    events = [pass_events(units, policy) for units, (_, policy, _) in zip(stated, shares)]
    return tuple(map(sum, zip(*stated))), tuple(map(sum, zip(*events)))


def pass_timeline(hybrid: Any, writes: ProgramWrites, operator_of: Mapping[int, str]) -> List[Dict[str, Any]]:
    """Stage-ordered entries: a pass with its write events, or a host stage with its op."""
    census = None if writes.census is None else writes.census.census
    shares = _shares(writes)
    units, events = _units(shares)
    carry = {} if census is None else {b.after_pass: b for b in census.boundaries}
    out: List[Dict[str, Any]] = []
    q = 0
    for position, stage in enumerate(hybrid.stages):
        if stage.kind == "compute":
            op = stage.compute_op
            out.append({"position": position, "kind": "host", "operator": operator_of.get(int(op.id), op.name),
                        "op_kind": op_kind(op)})
            continue
        entry: Dict[str, Any] = {"position": position, "kind": "pass", "pass": q,
                                 "operators": _operators_of(stage, operator_of)}
        if census is not None and shares is not None:
            row = census.passes[q]
            mine = [(share.passes[q], policy) for share, policy, _ in shares]  # [U-C4] each share's cells
            entry.update(segment_index=int(row.segment_index), cores=int(row.cores),
                         resident_cores=int(row.resident_cores),
                         static_written_cells_per_period=sum(int(p.written_cells_backing_store) * _per(policy)
                                                             for p, policy in mine),
                         operand_written_cells_per_inference=sum(int(p.written_cells_producer) * _per(policy)
                                                                 for p, policy in mine),
                         payload_bytes_per_period=int(row.payload_bytes),
                         write_units_per_period=None if units is None else int(units[q]),
                         write_events_per_period=None if events is None else int(events[q]),
                         operand_write_events_per_inference=(
                             None if writes.policy is None or not row.written_cells_producer
                             else sum(int(pass_events([int(p.written_cells_producer) * _per(policy)], policy)[0])
                                      for p, policy in mine)))
            boundary = carry.get(q)
            if boundary is not None:
                entry.update(carry_out_bytes=int(boundary.carry_out_bytes), carry_in_bytes=int(boundary.carry_in_bytes))
        out.append(entry)
        q += 1
    return out


def writes_summary(writes: ProgramWrites) -> Dict[str, Any]:
    """The period's static writes, the operand writes per inference, the most-written core and the write-law
    quantities at the declared batch (``write_quantities``)."""
    if writes.policy is None or writes.census is None:
        raise TimelineError(writes.unavailable or "the run declares no inference_batch_K")
    shares = _shares(writes) or ()
    census = writes.census.census
    most = writes.census.most_written_core
    quantities, _ = summed_cells(writes, write_quantities(census, writes.policy,
                                                          written_extents=writes.census.written_extents,
                                                          most_written=most))

    def value(name: str) -> Optional[float]:
        q = quantities.get(name)
        return None if q is None else float(q.value)

    def cells(field: str) -> int:  # [U-C4] written cells: each written weight's cells, per share
        return sum(int(sum(getattr(p, field) for p in share.passes)) * _per(policy) for share, policy, _ in shares)

    return {"declared_K": int(writes.policy.batch_k),
            "static_written_cells_per_period": cells("written_cells_backing_store"),
            "operand_written_cells_per_inference": cells("written_cells_producer"),
            "payload_bytes_per_period": int(sum(p.payload_bytes for p in census.passes)),
            "written_passes": int(sum(1 for p in census.passes if not p.resident)),
            "most_written_core_writes_per_period": None if most is None else int(most),
            "max_writes_per_cell_per_inference": value("max_writes_per_cell_per_inference"),
            "write_events_per_inference": value("write_events"),
            "write_granularity": writes.policy.write_granularity, "census_label": census.label}
