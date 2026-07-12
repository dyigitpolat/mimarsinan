"""Advisory record, severity taxonomy, and the lossless-mandate predicate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.chip_simulation.spiking_semantics import is_lif

SEVERITY_UNSUPPORTED = "UNSUPPORTED"
SEVERITY_RISK = "RISK"
SEVERITY_INFO = "INFO"
SEVERITIES = frozenset({SEVERITY_UNSUPPORTED, SEVERITY_RISK, SEVERITY_INFO})


@dataclass(frozen=True)
class Advisory:
    """One fired deployment advisory: a tentative-theory warning, never a gate.

    ``mandate_violation`` marks a predicted DEPLOYMENT accuracy loss on a
    lif/synchronized-ttfs run — the lossless-refinement checklist entry.
    """

    id: str
    severity: str
    title: str
    detail: str
    tentative: bool
    mandate_violation: bool
    suggested_levers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"unknown advisory severity {self.severity!r}; "
                f"legal severities: {sorted(SEVERITIES)}"
            )

    def as_payload(self) -> dict[str, Any]:
        """Machine-readable form consumed by reporter events, the GUI payload,
        and the refinement tooling (which filters on ``mandate_violation``)."""
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "tentative": self.tentative,
            "mandate_violation": self.mandate_violation,
            "suggested_levers": list(self.suggested_levers),
        }


def lossless_mandate_applies(plan: Any) -> bool:
    """lif / synchronized-ttfs deployments are the lossless-refinement mandate's
    scope: any predicted deployment loss there is a refinement work item.
    Reads the resolved ``DeploymentPlan`` axes, never raw config flags."""
    return is_lif(plan.spiking_mode) or bool(plan.is_synchronized_ttfs)
