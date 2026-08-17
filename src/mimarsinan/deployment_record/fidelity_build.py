"""Building a fidelity report: one registry read per side, over the same axes."""

from __future__ import annotations

from typing import Optional, Tuple

from mimarsinan.deployment_record.cost.terms import CostTerm, find_term_or_none
from mimarsinan.deployment_record.fidelity import (
    AxisComparison,
    FidelityReport,
    TermComparison,
    compare_axis,
    compare_term,
    save_fidelity_report,
)
from mimarsinan.deployment_record.objectives.probes import full_candidate_probe
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


def _axis_basis(
    spec: ObjectiveSpecV2,
    predicted: Optional[float],
    measured: Optional[float],
    candidate: Optional[RecordView],
) -> str:
    """[H3] WHY a side is absent — never a bare empty cell."""
    if predicted is not None and measured is not None:
        return ""
    reasons = []
    if predicted is None:
        if candidate is None:
            reasons.append("prediction: no candidate rebuild for this run")
        elif not spec.available(full_candidate_probe()):
            reasons.append("prediction: record-only axis (no candidate backing)")
        else:
            reasons.append(
                f"prediction unavailable on this run: requires {spec.requires}"
            )
    if measured is None:
        reasons.append(
            "measurement: the sealed record does not answer this axis on this run"
        )
    return "; ".join(reasons)


def _term_rows(
    candidate: Optional[RecordView], measured_view: RecordView
) -> Tuple[TermComparison, ...]:
    """[H3] Every priced term, zipped by NAME across the two completenesses.

    The absolute terms share names by construction (one pricer), so an
    axis-level split decomposes into named causes — the host term's estimated
    rates stop hiding inside the e2e headline.
    """
    def _terms_of(view: Optional[RecordView]):
        report = None if view is None else view.cost_report()
        if report is None:
            return {}
        return {term.name: term for term in report.all_terms()}

    predicted_terms = _terms_of(candidate)
    measured_terms = _terms_of(measured_view)
    rows = []
    for name in sorted(set(predicted_terms) | set(measured_terms)):
        predicted = predicted_terms.get(name)
        measured = measured_terms.get(name)
        predicted_value = getattr(predicted, "value", None)
        measured_value = getattr(measured, "value", None)
        if predicted_value is None and measured_value is None:
            continue
        band = getattr(predicted, "band", None)
        evidence = (getattr(measured, "source", "")
                    or getattr(predicted, "source", "") or "")
        rows.append(compare_term(
            name=name,
            unit=getattr(predicted or measured, "unit", ""),
            predicted=predicted_value,
            measured=measured_value,
            predicted_band=(
                None if band is None else (float(band.low), float(band.high))
            ),
            evidence=evidence,
        ))
    return tuple(rows)


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
            basis=_axis_basis(spec, predicted, measured, candidate),
        ))
    return FidelityReport(
        cell_key=record.identity.cell_key,
        run_dir=record.identity.run_dir,
        axes=tuple(axes),
        terms=_term_rows(candidate, measured_view),
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
