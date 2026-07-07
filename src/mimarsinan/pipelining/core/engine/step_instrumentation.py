"""Per-step instrumentation: [PROFILE] line + structured events + retention gate."""

from __future__ import annotations


def emit_pipeline_event(pipeline, kind: str, payload: dict) -> None:
    """Structured-event side channel; a bare Pipeline (unit tests) has no reporter."""
    reporter = getattr(pipeline, "reporter", None)
    if reporter is not None and hasattr(reporter, "event"):
        reporter.event(kind, payload)


def emit_step_profile(pipeline, name: str, step, wall_s: float,
                      final_metric: float, previous_metric: float) -> None:
    """One [PROFILE] console line + one ``profile`` event per completed step."""
    delta = final_metric - previous_metric
    print(
        f"[PROFILE] step='{name}' wall={wall_s:7.2f}s "
        f"metric={final_metric:.4f} "
        f"Δ={delta:+.4f} (prev={previous_metric:.4f})"
    )
    emit_pipeline_event(pipeline, "profile", {
        "step": name,
        "wall_s": wall_s,
        "metric": final_metric,
        "delta": delta,
        "prev": previous_metric,
        "metric_kind": step.pipeline_metric_kind(),
    })


def assert_metric_retention(pipeline, step, previous_metric: float,
                            step_tolerance: float) -> None:
    """The per-step retention gate; a failure emits a ``retention`` event, then raises."""
    current = pipeline.get_target_metric()
    floor = previous_metric * step_tolerance
    if current >= floor:
        return
    emit_pipeline_event(pipeline, "retention", {
        "step": step.name,
        "metric": current,
        "previous": previous_metric,
        "tolerance": step_tolerance,
        "floor": floor,
    })
    raise AssertionError(
        f"[{step.name}] step failed to retain performance within tolerable "
        f"limits: {current} < ({previous_metric} * "
        f"{step_tolerance}) = {floor}"
    )
