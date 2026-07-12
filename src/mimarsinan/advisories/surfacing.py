"""Loud [ADVISORY] console surfacing + machine-readable reporter events."""

from __future__ import annotations

from typing import Any, Iterable

from mimarsinan.advisories.advisory import Advisory
from mimarsinan.common.reporter import emit_reporter_event

ADVISORY_EVENT_KIND = "deployment_advisory"


def _flags_line(advisory: Advisory) -> str:
    flags = []
    if advisory.tentative:
        flags.append("tentative theory")
    if advisory.mandate_violation:
        flags.append("mandate-violation: predicted deployment loss on a "
                     "lossless-mandate (lif/sync) run")
    return "; ".join(flags)


def surface_advisories(
    reporter: Any, advisories: Iterable[Advisory], *, context: str
) -> None:
    """Print each advisory loudly and emit a ``deployment_advisory`` reporter
    event per advisory. Warnings only — never raises on advisory content."""
    for advisory in advisories:
        print(f"[ADVISORY][{advisory.severity}] {advisory.id}: {advisory.title}")
        print(f"            {advisory.detail}")
        if advisory.suggested_levers:
            print(f"            suggested levers: "
                  f"{'; '.join(advisory.suggested_levers)}")
        flags = _flags_line(advisory)
        if flags:
            print(f"            ({flags})")
        emit_reporter_event(
            reporter,
            ADVISORY_EVENT_KIND,
            {"context": context, **advisory.as_payload()},
        )
