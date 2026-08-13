"""The absolute terms folded into the report — one shape at both completenesses."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

from mimarsinan.deployment_record.cost.absolute.context import AbsolutePricing
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import CostTerm, DeploymentCostReport
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities
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
    raise ValueError(f"absolute term {term.name!r} has unrouted unit {term.unit!r}")


def _grouped(pricing: AbsolutePricing) -> Dict[str, List[CostTerm]]:
    groups: Dict[str, List[CostTerm]] = {
        "area": [], "energy": [], "latency": [], "throughput": [],
    }
    for term in pricing.terms:
        groups[_group_of(term)].append(term)
    return groups


def _notes(pricing: AbsolutePricing) -> Tuple[str, ...]:
    refusal_notes = tuple(
        f"refused {refusal.name}: {refusal.reason}" for refusal in pricing.refusals
    )
    return (_PARALLEL_NOTE,) + refusal_notes


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
    )


def candidate_cost_report(
    quantities: Quantities, physics: PlatformPhysics
) -> DeploymentCostReport:
    """A candidate's report: priced terms only — no segments, no measured plane."""
    pricing = price_absolute(quantities, physics)
    groups = _grouped(pricing)
    return DeploymentCostReport(
        segments=(),
        energy=tuple(groups["energy"]),
        latency=tuple(groups["latency"]),
        area=tuple(groups["area"]),
        throughput=tuple(groups["throughput"]),
        notes=_notes(pricing),
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
    """Price the sealed record's quantities with its own declared physics."""
    physics = declared_physics_of(record)
    if physics is None:
        return None
    return price_absolute(from_record(record), physics)


def emit_physics_report(record: DeploymentRecord, run_dir: str) -> Optional[str]:
    """Write the vendor-priced report iff the record declared physics.

    A no-profile record writes NOTHING (byte-identity); a declared one gets the
    priced terms over the measured base when the cost model's preconditions hold
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
    return save_deployment_cost_report(
        report_with_absolute_terms(base, pricing), run_dir
    )


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
