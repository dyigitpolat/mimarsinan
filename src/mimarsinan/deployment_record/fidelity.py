"""The fidelity contract: what the search PREDICTED against what the run MEASURED.

The EDA estimate-vs-signoff correlation report. One pricer serves both
completenesses (a candidate's static shape, a sealed record's measurements), so the
same objective keys carry the same term names on both sides and this is a zip rather
than a second model.

Report-first by decision: it states per-axis agreement and never gates on a numeric
tolerance. Only STRUCTURAL equalities are gated, and those are gated where they
belong (the pass-count agreement is pinned in the mapping tests, not judged here).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema.serde import strict_kwargs, tuple_of

FIDELITY_FILENAME = "fidelity.json"


@dataclass(frozen=True)
class AxisComparison:
    """One objective axis, predicted and measured.

    Either side may be absent: an axis the candidate could not predict is still
    worth recording (the measurement stands alone), and so is a prediction the run
    never measured. ``in_band`` is a verdict only where a BAND was predicted — a
    point prediction is not a claim about a range, and reporting one as out-of-band
    would invent a tolerance nobody declared.
    """

    key: str
    unit: str
    direction: str
    predicted: Optional[float]
    measured: Optional[float]
    predicted_band: Optional[Tuple[float, float]]
    in_band: Optional[bool]
    relative_error: Optional[float]
    #: [H3] WHY a side is absent ("" when both sides answered) — the
    #: refusal-with-reason discipline extended to the report itself.
    basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "unit": self.unit,
            "direction": self.direction,
            "predicted": self.predicted,
            "measured": self.measured,
            "predicted_band": (
                None if self.predicted_band is None else list(self.predicted_band)
            ),
            "in_band": self.in_band,
            "relative_error": self.relative_error,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisComparison":
        kwargs = strict_kwargs(cls, data)
        band = kwargs.get("predicted_band")
        kwargs["predicted_band"] = (
            None if band is None else (float(band[0]), float(band[1]))
        )
        return cls(**kwargs)


def compare_axis(
    *,
    key: str,
    unit: str,
    direction: str,
    predicted: Optional[float],
    measured: Optional[float],
    predicted_band: Optional[Tuple[float, float]],
    basis: str = "",
) -> AxisComparison:
    """One axis' comparison, with the verdicts each side actually supports."""
    in_band: Optional[bool] = None
    if measured is not None and predicted_band is not None:
        low, high = predicted_band
        in_band = bool(low <= measured <= high)
    relative_error: Optional[float] = None
    if predicted is not None and measured not in (None, 0.0):
        # The MEASUREMENT is the reference: the prediction is what was wrong.
        relative_error = (float(predicted) - float(measured)) / float(measured)
    return AxisComparison(
        key=key,
        unit=unit,
        direction=direction,
        predicted=None if predicted is None else float(predicted),
        measured=None if measured is None else float(measured),
        predicted_band=(
            None if predicted_band is None
            else (float(predicted_band[0]), float(predicted_band[1]))
        ),
        in_band=in_band,
        relative_error=relative_error,
        basis=str(basis),
    )


@dataclass(frozen=True)
class TermComparison:
    """[H3] One priced COST TERM, predicted and measured-plane.

    The absolute terms share names across both completenesses by construction
    (one pricer), so an axis-level split decomposes into named causes here —
    "e2e off 100x" becomes "t_cycle x steps within X%, host term off Nx
    (estimated rates)". ``measured`` is the record-plane PRICED value (same
    constants, sealed multiplicands); truly measured energies/walls remain
    axis rows (``mj_per_sample``), never conflated with priced terms.
    """

    name: str
    unit: str
    predicted: Optional[float]
    measured: Optional[float]
    predicted_band: Optional[Tuple[float, float]]
    in_band: Optional[bool]
    relative_error: Optional[float]
    #: The evidence basis of the constants behind the term (from the record
    #: plane when present, else the candidate's).
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "predicted": self.predicted,
            "measured": self.measured,
            "predicted_band": (
                None if self.predicted_band is None else list(self.predicted_band)
            ),
            "in_band": self.in_band,
            "relative_error": self.relative_error,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TermComparison":
        kwargs = strict_kwargs(cls, data)
        band = kwargs.get("predicted_band")
        kwargs["predicted_band"] = (
            None if band is None else (float(band[0]), float(band[1]))
        )
        return cls(**kwargs)


def compare_term(
    *,
    name: str,
    unit: str,
    predicted: Optional[float],
    measured: Optional[float],
    predicted_band: Optional[Tuple[float, float]],
    evidence: str = "",
) -> TermComparison:
    """One term's comparison — the axis rules, applied to a priced term."""
    in_band: Optional[bool] = None
    if measured is not None and predicted_band is not None:
        low, high = predicted_band
        in_band = bool(low <= measured <= high)
    relative_error: Optional[float] = None
    if predicted is not None and measured not in (None, 0.0):
        relative_error = (float(predicted) - float(measured)) / float(measured)
    return TermComparison(
        name=name, unit=unit,
        predicted=None if predicted is None else float(predicted),
        measured=None if measured is None else float(measured),
        predicted_band=(
            None if predicted_band is None
            else (float(predicted_band[0]), float(predicted_band[1]))
        ),
        in_band=in_band, relative_error=relative_error, evidence=str(evidence),
    )


@dataclass(frozen=True)
class FidelityReport:
    """One run's estimate-vs-signoff correlation, axis by axis."""

    cell_key: str
    run_dir: str
    axes: Tuple[AxisComparison, ...]
    #: [H3] The per-term decomposition; additive-optional so pre-H3 reports load.
    terms: Tuple[TermComparison, ...] = ()
    #: [R4] The run's measured switching anchor (events / (macs x steps)) and
    #: the operator's declaration — every censused run states its own anchor.
    measured_effective_activity: Optional[float] = None
    declared_activity_factor: Optional[float] = None

    @property
    def compared_count(self) -> int:
        """Axes where BOTH sides answered — the ones that say anything."""
        return sum(
            1 for axis in self.axes
            if axis.predicted is not None and axis.measured is not None
        )

    @property
    def in_band_count(self) -> int:
        return sum(1 for axis in self.axes if axis.in_band)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_key": self.cell_key,
            "run_dir": self.run_dir,
            "axes": [axis.to_dict() for axis in self.axes],
            "terms": [term.to_dict() for term in self.terms],
            "measured_effective_activity": self.measured_effective_activity,
            "declared_activity_factor": self.declared_activity_factor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FidelityReport":
        kwargs = strict_kwargs(cls, data)
        kwargs["axes"] = tuple_of(AxisComparison.from_dict, kwargs["axes"])
        kwargs["terms"] = tuple_of(
            TermComparison.from_dict, kwargs.get("terms") or ()
        )
        return cls(**kwargs)


def save_fidelity_report(report: FidelityReport, run_dir: str) -> str:
    """Atomically write ``fidelity.json`` into ``run_dir``; return the path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, FIDELITY_FILENAME)
    fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=FIDELITY_FILENAME, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def load_fidelity_report(path: str) -> FidelityReport:
    """Load a written report (the aggregation script's reader)."""
    with open(path, "r", encoding="utf-8") as fh:
        return FidelityReport.from_dict(json.load(fh))
