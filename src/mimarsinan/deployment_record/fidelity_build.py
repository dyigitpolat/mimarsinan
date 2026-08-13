"""Building a fidelity report: one registry read per side, over the same axes."""

from __future__ import annotations

from typing import Optional, Tuple

from mimarsinan.deployment_record.cost.terms import CostTerm, find_term_or_none
from mimarsinan.deployment_record.fidelity import (
    AxisComparison,
    FidelityReport,
    compare_axis,
    save_fidelity_report,
)
from mimarsinan.deployment_record.objectives.catalog import OBJECTIVES
from mimarsinan.deployment_record.objectives.spec import ObjectiveSpecV2, RecordView
from mimarsinan.deployment_record.objectives.views import DeploymentRecordView
from mimarsinan.deployment_record.schema import DeploymentRecord

#: Which report group each priced axis' term lives in, so a predicted BAND can be
#: recovered beside its value. Absent here, an axis is compared without a band.
_BANDED_GROUPS = {
    "chip_area_mm2": "area",
    "energy_per_inference_mj": "energy",
    "e2e_latency_s": "latency",
    "throughput_inferences_s": "throughput",
}


def _value(spec: ObjectiveSpecV2, view: Optional[RecordView]) -> Optional[float]:
    if view is None or not spec.available(view):
        return None
    return float(spec.value(view))


def _predicted_band(
    spec: ObjectiveSpecV2, view: Optional[RecordView]
) -> Optional[Tuple[float, float]]:
    """The band the candidate's own priced term carries, when it has one."""
    group = _BANDED_GROUPS.get(spec.key)
    if group is None or view is None:
        return None
    report = view.cost_report()
    if report is None:
        return None
    term: Optional[CostTerm] = find_term_or_none(getattr(report, group), spec.key)
    if term is None or term.band is None:
        return None
    return (float(term.band.low), float(term.band.high))


def fidelity_report_for_record(
    record: DeploymentRecord, candidate: Optional[RecordView]
) -> FidelityReport:
    """Compare a sealed run against the candidate view of its own configuration.

    Every axis EITHER side can answer is reported, in catalog order: an axis only the
    record answers still belongs (it is what the search could not see), and so does
    one only the candidate predicted.
    """
    measured_view = DeploymentRecordView(record=record)
    axes: list[AxisComparison] = []
    for spec in OBJECTIVES.all():
        predicted = _value(spec, candidate)
        measured = _value(spec, measured_view)
        if predicted is None and measured is None:
            continue
        axes.append(compare_axis(
            key=spec.key,
            unit=spec.unit,
            direction=spec.direction,
            predicted=predicted,
            measured=measured,
            predicted_band=_predicted_band(spec, candidate),
        ))
    return FidelityReport(
        cell_key=record.identity.cell_key,
        run_dir=record.identity.run_dir,
        axes=tuple(axes),
    )


def emit_fidelity_report(
    record: DeploymentRecord, candidate: Optional[RecordView], run_dir: str
) -> Optional[str]:
    """Write ``fidelity.json`` when there is a prediction to compare against.

    No candidate view means the run was not searched, so there is nothing to
    correlate — which is different from a fidelity of zero and is written as
    nothing at all.
    """
    if candidate is None:
        return None
    return save_fidelity_report(fidelity_report_for_record(record, candidate), run_dir)
