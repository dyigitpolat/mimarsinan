"""Building a fidelity report: one registry read per side, over the same axes."""

from __future__ import annotations

import logging
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


#: [R4] Warn when the measured switching anchor misses the declaration by
#: more than this ratio either way (owner decision: warn loudly, never gate —
#: the declaration is an assumption, and one that measurement refutes must be
#: impossible to not see). 1.5x: inside it, the energy bands still bracket.
ACTIVITY_MISS_WARN_RATIO = 1.5


def _effective_activity(view: RecordView) -> Optional[float]:
    quantities = getattr(view, "quantities", None)
    if quantities is None:
        return None
    needed = ("synaptic_events", "macs", "timesteps")
    if any(not quantities.has(key) for key in needed):
        return None
    denominator = (quantities.get("macs").value
                   * quantities.get("timesteps").value)
    if denominator <= 0:
        return None
    return quantities.get("synaptic_events").value / denominator


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
    declared = record.identity.platform.get("activity_factor")
    return FidelityReport(
        cell_key=record.identity.cell_key,
        run_dir=record.identity.run_dir,
        axes=tuple(axes),
        terms=_term_rows(candidate, measured_view),
        measured_effective_activity=_effective_activity(measured_view),
        declared_activity_factor=(
            None if not declared else float(declared)
        ),
    )


#: [R4] The axes proven closed become hard in-band checks at emission (owner
#: decision): a gate fires ONLY when both sides answered, and fails the run
#: loudly. Energy/e2e stay report-only until R5's bands are in evidence.
GATED_AXES = {
    "total_param_capacity": 1e-6,
    "param_utilization_pct": 1e-2,
    "neuron_wastage_pct": 1e-2,
    "axon_wastage_pct": 1e-2,
    "chip_area_mm2": 1e-3,
    "carried_raster_bytes": 1e-6,
    "carry_peak_live_bytes": 1e-6,
}


def enforce_fidelity_gates(report: FidelityReport) -> None:
    """Raise on any gated axis whose two answers disagree past its tolerance."""
    failures = []
    for axis in report.axes:
        tolerance = GATED_AXES.get(axis.key)
        if tolerance is None or axis.predicted is None or axis.measured is None:
            continue
        reference = max(abs(axis.measured), 1e-12)
        if abs(axis.predicted - axis.measured) / reference > tolerance:
            failures.append(
                f"{axis.key}: predicted {axis.predicted!r} vs measured "
                f"{axis.measured!r} (tolerance {tolerance})"
            )
    if failures:
        raise ValueError(
            "fidelity gate: the candidate and the sealed record disagree on "
            "axes proven closed — " + "; ".join(failures)
        )


def activity_warning(report: FidelityReport) -> Optional[str]:
    """[R4] The loud declaration-miss warning, or None inside the band."""
    measured = report.measured_effective_activity
    declared = report.declared_activity_factor
    if not measured or not declared:
        return None
    ratio = measured / declared
    if 1.0 / ACTIVITY_MISS_WARN_RATIO <= ratio <= ACTIVITY_MISS_WARN_RATIO:
        return None
    return (
        f"declared activity_factor={declared} but the sealed census measures "
        f"effective activity {measured:.4f} ({ratio:.2f}x off) — re-declare "
        f"from the measured anchor; every energy prediction scales with it"
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
    report = fidelity_report_for_record(record, candidate)
    warning = activity_warning(report)
    if warning is not None:
        logging.getLogger(__name__).warning("[fidelity] %s", warning)
    enforce_fidelity_gates(report)
    return save_fidelity_report(report, run_dir)
