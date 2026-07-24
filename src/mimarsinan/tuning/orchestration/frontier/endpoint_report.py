"""EndpointRecoveryReport and its console/reporter emission."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mimarsinan.common.reporter import emit_reporter_event


@dataclass(frozen=True)
class EndpointRecoveryReport:
    """One endpoint-stage engagement record (target, reads, budget, verdict)."""

    target: float
    entry: float
    exit: float
    budget_steps: int
    steps_used: int
    engaged: bool
    reached: bool
    rolled_back: bool
    target_floor: float = 0.0
    target_margin: float = 0.0
    floor_lifted: bool = False
    armed: bool = False
    divergence_rescued: bool = False
    decoupled: bool = False


def emit_endpoint_report(tuner, report: EndpointRecoveryReport, trajectory) -> None:
    # The console line's field names are a harvest-parsed contract: keep verbatim.
    print(
        f"[MBH-ENDPOINT] tuner={type(tuner).__name__} "
        f"target={report.target:.6f} entry={report.entry:.6f} "
        f"exit={report.exit:.6f} budget={report.budget_steps} "
        f"steps_used={report.steps_used} engaged={report.engaged} "
        f"reached={report.reached} rolled_back={report.rolled_back} "
        f"target_floor={report.target_floor:.6f} "
        f"floor_lifted={report.floor_lifted} "
        f"armed={report.armed} "
        f"divergence_rescued={report.divergence_rescued} "
        f"decoupled={report.decoupled}",
        flush=True,
    )
    fields = asdict(report)
    emit_reporter_event(tuner.pipeline.reporter, "mbh_endpoint", {
        "tuner": type(tuner).__name__, **fields, "trajectory": list(trajectory),
    })
    rounded = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in fields.items()
    }
    rounded.pop("decoupled")
    tuner.pipeline.reporter.report(f"{tuner.name} endpoint_recovery", rounded)
