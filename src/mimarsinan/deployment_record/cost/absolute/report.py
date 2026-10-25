"""The absolute terms folded into the report — one shape at both completenesses."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import mimarsinan.mapping.platform.target_bundles as target_bundles
from mimarsinan.deployment_record.cost.absolute.context import AbsolutePricing, PricingRefusal
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.output_stage import output_stage_note
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED, declared_geometry
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import CostTerm, DeploymentCostReport
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities
from mimarsinan.deployment_record.quantities.write_law import record_write_census
from mimarsinan.deployment_record.schema import DeploymentRecord

#: The headline axes the fidelity contract zips predicted-vs-measured over.
ABSOLUTE_TERM_NAMES: Tuple[str, ...] = (
    "chip_area_mm2",
    "energy_per_inference_mj",
    "e2e_latency_s",
    "throughput_inferences_s",
)

DEPLOYMENT_COST_REPORT_FILENAME = "deployment_cost_report.json"

_PARALLEL_NOTE = (
    "absolute vendor-physics terms are a parallel PREDICTION of the measured plane; "
    "they are never added to measured totals"
)


def _group_of(term: CostTerm) -> str:
    if term.unit == "mm^2":
        return "area"
    if term.unit == "mJ":
        return "energy"
    if term.unit == "s":
        return "latency"
    if term.unit == "inferences/s":
        return "throughput"
    if term.unit in ("inferences", "h"):
        return "lifetime"
    raise ValueError(f"absolute term {term.name!r} has unrouted unit {term.unit!r}")


def _grouped(pricing: AbsolutePricing) -> Dict[str, List[CostTerm]]:
    groups: Dict[str, List[CostTerm]] = {
        "area": [], "energy": [], "latency": [], "throughput": [], "lifetime": [],
    }
    for term in pricing.terms:
        groups[_group_of(term)].append(term)
    return groups


def _notes(pricing: AbsolutePricing) -> Tuple[str, ...]:
    refusal_notes = tuple(
        f"refused {refusal.name}: {refusal.reason}" for refusal in pricing.refusals
    )
    verdict_notes = tuple(
        f"admissibility {verdict.name}: {verdict.verdict} ({verdict.reason}; {verdict.evidence})"
        for verdict in pricing.verdicts
    )
    return (_PARALLEL_NOTE,) + refusal_notes + verdict_notes


def report_with_absolute_terms(
    report: DeploymentCostReport, pricing: AbsolutePricing
) -> DeploymentCostReport:
    """The record's report with the priced terms appended to their groups.

    Existing terms are untouched and the priced ones are never summed into the
    measured totals — the appended note states the parallel-prediction discipline.
    """
    groups = _grouped(pricing)
    return DeploymentCostReport(
        segments=report.segments,
        energy=report.energy + tuple(groups["energy"]),
        latency=report.latency + tuple(groups["latency"]),
        area=report.area + tuple(groups["area"]),
        throughput=report.throughput + tuple(groups["throughput"]),
        notes=report.notes + _notes(pricing),
        lifetime=report.lifetime + tuple(groups["lifetime"]),
    )


def refused_pricing(refusal: Tuple[str, str]) -> AbsolutePricing:
    """No absolute term, only the ``(name, reason)`` refusal (``target_bundles.absolute_pricing_refusal``)."""
    return AbsolutePricing(terms=(), refusals=(PricingRefusal(name=refusal[0], reason=refusal[1]),))


def candidate_cost_report(
    quantities: Quantities, physics: Optional[PlatformPhysics], *,
    writes: Optional[WriteInputs] = None, geometry: Optional[Tuple[Any, ...]] = None,
    refusal: Optional[Tuple[str, str]] = None,
    output_stage: Optional[Mapping[str, Any]] = None,
) -> DeploymentCostReport:
    """A candidate's report: priced terms only — no segments, no measured plane. A candidate that
    declares no complete core geometry prices the aggregates as declared (``UNJUDGED``); one on a
    target that declares no physics carries that target's ``refusal`` alone; an ``output_stage``
    (W08, ``mapping.lanes.output_stage``) is printed beside it."""
    if refusal is not None:
        pricing = refused_pricing(refusal)
    elif physics is None:
        raise ValueError("a candidate cost report needs the candidate's physics or its pricing refusal")
    else:
        pricing = price_absolute(quantities, physics, writes=writes,
                                 geometry=UNJUDGED if geometry is None else geometry)
    groups = _grouped(pricing)
    stage = () if output_stage is None else (output_stage_note(output_stage, pricing),)
    return DeploymentCostReport(
        segments=(),
        energy=tuple(groups["energy"]),
        latency=tuple(groups["latency"]),
        area=tuple(groups["area"]),
        throughput=tuple(groups["throughput"]),
        notes=_notes(pricing) + stage,
        lifetime=tuple(groups["lifetime"]),
    )


def declared_physics_of(record: DeploymentRecord) -> Optional[PlatformPhysics]:
    """The physics the record was sealed with, or None when none was declared."""
    payload = record.identity.platform.get("platform_physics_resolved")
    if payload is None:
        return None
    return PlatformPhysics.from_dict(payload)


def absolute_pricing_for_record(
    record: DeploymentRecord,
) -> Optional[AbsolutePricing]:
    """Price the sealed record's quantities with its own declared physics; a record of a target that
    declares none carries the target's refusal by name, and one without physics or target, nothing."""
    if (refusal := target_bundles.absolute_pricing_refusal(record.identity.platform)) is not None:
        return refused_pricing(refusal)
    physics = declared_physics_of(record)
    if physics is None:
        return None
    geometry = record_geometry(record)
    return price_absolute(from_record(record), physics, writes=record_write_inputs(record),
                          geometry=UNJUDGED if geometry is None else geometry)


def record_geometry(record: DeploymentRecord) -> Optional[Tuple[Any, ...]]:
    """The sealed chip's declared core types, or None when the identity declares none."""
    return declared_geometry(list(record.identity.platform.get("cores") or ()))


def record_write_inputs(record: DeploymentRecord) -> Optional[WriteInputs]:
    """The sealed record's write-law inputs, present only when it declares ``inference_batch_K``."""
    writes = record_write_census(record)
    return None if writes is None else WriteInputs.of(writes)


def emit_physics_report(record: DeploymentRecord, run_dir: str) -> Optional[str]:
    """Write the vendor-priced report iff the record declared physics or a target that declares none.

    A no-profile record writes NOTHING (byte-identity); a geometry-only target's writes its refusal; a
    declared one gets the priced terms over the measured base when the cost model's preconditions hold
    (measured energy + compute latency), or alone otherwise.
    """
    pricing = absolute_pricing_for_record(record)
    if pricing is None:
        return None
    costable = (
        record.energy is not None
        and record.timing.latency.compute_sim_time_s is not None
    )
    base = (
        DeploymentCostModel().evaluate(record)
        if costable
        else DeploymentCostReport(segments=(), energy=(), latency=(), area=(),
                                  throughput=(), notes=())
    )
    report = report_with_absolute_terms(base, pricing)
    if record.output_stage is not None:  # [W08] printed beside the priced point
        report = replace(report, notes=report.notes + (output_stage_note(record.output_stage, pricing),))
    return save_deployment_cost_report(report, run_dir)


def save_deployment_cost_report(report: DeploymentCostReport, run_dir: str) -> str:
    """Atomically write the report beside the record (the save_deployment_record way)."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, DEPLOYMENT_COST_REPORT_FILENAME)
    fd, tmp_path = tempfile.mkstemp(
        dir=run_dir, prefix=DEPLOYMENT_COST_REPORT_FILENAME, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path
