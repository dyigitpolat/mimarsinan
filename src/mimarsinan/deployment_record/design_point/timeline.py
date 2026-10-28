"""The pass timeline of an executed program with its write events, and the period's write summary.

Read on the record plane at build time: the built program's passes (``mapping_program_passes``) under the write law's
one period census (``plane_write_census``: the executed rotation) at the declared batch. Per pass: its cores, the cores
it keeps across periods, the static cells it rewrites per period and their bytes, the write units and write events at
the declared write granularity (``pass_units``, ``pass_events``; None where the granularity is undeclared), the operand
cells it writes per inference (the producer cells of a dynamic-operand pass) and their write events, the operators it
holds and the carry read out after it. Host stages sit between passes in stage order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import WriteCensus
from mimarsinan.deployment_record.quantities.write_law import (
    WritePolicy, pass_events, pass_units, plane_write_census, write_policy_of, write_quantities,
)
from mimarsinan.deployment_record.quantities.write_technology import TechnologySplit, split_by_technology
from mimarsinan.mapping.capability.op_kinds import op_kind


class TimelineError(ToolchainRefusal, ValueError):
    """The executed program has no period census to summarize: no declared batch, or the census refused."""


@dataclass(frozen=True)
class ProgramWrites:
    """The executed program's period census at the declared batch, or the reason it has none."""

    policy: Optional[WritePolicy]
    census: Optional[WriteCensus]
    unavailable: Optional[str] = None
    #: [E-6, targeted pass] N2b: the census split by write technology where the platform's core types declare
    #: several (``write_technology.split_by_technology``, the record plane's rule), else None.
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


def _operators_of(stage: Any, operator_of: Mapping[int, str]) -> List[str]:
    seen: Dict[str, None] = {}
    for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
        for placement in placements:
            seen.setdefault(operator_of.get(int(placement["ir_node_id"]), str(placement["ir_node_id"])), None)
    return list(seen)


def _units(writes: ProgramWrites) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    if writes.policy is None or writes.census is None:
        return None, None
    units = pass_units(writes.census.census, writes.policy, writes.census.written_extents)
    return units, None if units is None else pass_events(units, writes.policy)


def pass_timeline(hybrid: Any, writes: ProgramWrites, operator_of: Mapping[int, str]) -> List[Dict[str, Any]]:
    """Stage-ordered entries: a pass with its write events, or a host stage with its op."""
    census = None if writes.census is None else writes.census.census
    units, events = _units(writes)
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
        if census is not None:
            row = census.passes[q]
            entry.update(segment_index=int(row.segment_index), cores=int(row.cores),
                         resident_cores=int(row.resident_cores),
                         static_written_cells_per_period=int(row.written_cells_backing_store),
                         operand_written_cells_per_inference=int(row.written_cells_producer),
                         payload_bytes_per_period=int(row.payload_bytes),
                         write_units_per_period=None if units is None else int(units[q]),
                         write_events_per_period=None if events is None else int(events[q]),
                         operand_write_events_per_inference=(
                             None if writes.policy is None or not row.written_cells_producer
                             else int(pass_events([int(row.written_cells_producer)], writes.policy)[0])))
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
    census = writes.census.census
    most = writes.census.most_written_core
    quantities = write_quantities(census, writes.policy, written_extents=writes.census.written_extents,
                                  most_written=most)

    def value(name: str) -> Optional[float]:
        q = quantities.get(name)
        return None if q is None else float(q.value)

    return {"declared_K": int(writes.policy.batch_k),
            "static_written_cells_per_period": int(sum(p.written_cells_backing_store for p in census.passes)),
            "operand_written_cells_per_inference": int(sum(p.written_cells_producer for p in census.passes)),
            "payload_bytes_per_period": int(sum(p.payload_bytes for p in census.passes)),
            "written_passes": int(sum(1 for p in census.passes if not p.resident)),
            "most_written_core_writes_per_period": None if most is None else int(most),
            "max_writes_per_cell_per_inference": value("max_writes_per_cell_per_inference"),
            "write_events_per_inference": value("write_events"),
            "write_granularity": writes.policy.write_granularity, "census_label": census.label}
