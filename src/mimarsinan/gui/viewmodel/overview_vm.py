"""Overview progression view-model: measured points + verdict markers, never carried."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

_MEASURED = "measured"
_CARRIED = "carried"


def build_overview_chart(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Split completed steps into metric POINTS and verdict/carried MARKERS.

    A carried metric contributes NO y-point (plotting it would fabricate a
    measurement); a step with a verdict (or a failure) renders as a labeled
    vertical marker with a pass/fail glyph instead.
    """
    points: List[Dict[str, Any]] = []
    markers: List[Dict[str, Any]] = []
    for step in steps:
        name = step.get("name")
        status = step.get("status")
        done = status in ("completed", "failed") or step.get("end_time") is not None
        if not done:
            continue
        verdict = step.get("verdict")
        kind = step.get("metric_kind")
        if status == "failed":
            markers.append({
                "step": name, "status": "fail", "glyph": "✖",
                "label": step.get("error") or "step failed",
            })
            continue
        if verdict is not None:
            markers.append({
                "step": name,
                "status": verdict.get("status", "pass"),
                "glyph": "✓" if verdict.get("status", "pass") == "pass" else "✖",
                "label": verdict.get("rule", ""),
                "detail": verdict.get("detail"),
            })
            continue
        if kind == _CARRIED:
            continue
        # measured — legacy runs without metric_kind stay plotted (pre-honesty
        # data carries no kind; refusing to plot them would erase history).
        if step.get("target_metric") is not None:
            points.append({"step": name, "value": step["target_metric"]})
    return {"points": points, "markers": markers}


def persisted_step_view(name: str, sd: Mapping[str, Any], *, status: str) -> Dict[str, Any]:
    """One steps.json entry -> the step view dict every overview shares."""
    start_t = sd.get("start_time")
    end_t = sd.get("end_time")
    return {
        "name": name,
        "status": status,
        "start_time": start_t,
        "end_time": end_t,
        "duration": (end_t - start_t) if start_t and end_t else None,
        "target_metric": sd.get("target_metric"),
        "metric_kind": sd.get("metric_kind"),
        "verdict": sd.get("verdict"),
    }


def step_bar_badge(step: Mapping[str, Any]) -> Dict[str, Any]:
    """What the pipeline-bar cell shows for one step: a metric or a verdict."""
    verdict = step.get("verdict")
    if verdict is not None:
        status = verdict.get("status", "pass")
        return {"kind": "verdict", "text": "PASS" if status == "pass" else "FAIL",
                "status": status}
    if step.get("metric_kind") == _CARRIED:
        return {"kind": "carried", "text": ""}
    if step.get("target_metric") is not None:
        return {"kind": "metric", "text": f"{step['target_metric']:.3f}"}
    return {"kind": "none", "text": ""}
