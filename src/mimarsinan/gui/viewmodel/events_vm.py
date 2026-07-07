"""Event display hints + per-chart annotation lists (the annotation lanes)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

# One row per event kind: which metric-chart categories it annotates, its
# color class, and how its label renders from the payload.
_KIND_DISPLAY = {
    "mbh_gate": {
        "categories": ("Adaptation", "Accuracy"),
        "labels": {
            "entry": ("entry", "neutral"),
            "accept": ("▲ accept", "good"),
            "reject": ("▽ reject", "warn"),
            "stall": ("┃ constructive stall", "bad"),
        },
    },
    "mbh_endpoint": {"categories": ("Adaptation", "Accuracy"),
                     "label": "endpoint", "tone": "neutral"},
    "mbh_hop": {"categories": ("Adaptation",), "label": "hop reaffine", "tone": "neutral"},
    "mbh_prefix": {"categories": ("Adaptation",), "label": "prefix reaffine", "tone": "neutral"},
    "mbh_a6": {"categories": ("Adaptation",), "label": "A6 gauge", "tone": "neutral"},
    "lr_refusal": {"categories": ("Loss", "Learning Rate", "Adaptation"),
                   "label": "┃ LR refused — entry preserved", "tone": "bad"},
    "retention": {"categories": ("Accuracy",), "label": "retention gate FAILED", "tone": "bad"},
    "parity": {"categories": ("Accuracy",), "label": "parity", "tone": "good"},
    "profile": {"categories": (), "label": "step boundary", "tone": "neutral"},
}


def display_hints(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Render hints for one event: label, tone, and target chart categories."""
    spec = _KIND_DISPLAY.get(kind)
    if spec is None:
        return {"label": kind, "tone": "neutral", "categories": []}
    if "labels" in spec:
        action = str(payload.get("action", ""))
        label, tone = spec["labels"].get(action, (action or kind, "neutral"))
        extras = []
        if "rung" in payload:
            extras.append(f"rung {payload['rung']}")
        if "rate" in payload:
            extras.append(f"rate {payload['rate']:.3f}")
        if extras:
            label = f"{label} ({', '.join(extras)})"
        return {"label": label, "tone": tone, "categories": list(spec["categories"])}
    label = spec["label"]
    if kind == "mbh_endpoint":
        outcome = "reached" if payload.get("reached") else "exhausted"
        label = f"endpoint {outcome} ({payload.get('steps_used', '?')} steps)"
    if kind == "parity" and "agreement" in payload:
        label = f"parity {payload['agreement']:.4f}"
    if kind == "mbh_a6":
        label = f"A6 {payload.get('gauge', '')}: {payload.get('verdict', '')}"
    return {"label": label, "tone": str(spec.get("tone", "neutral")),
            "categories": list(spec["categories"])}


def decorate(record: Mapping[str, Any]) -> Dict[str, Any]:
    """An event record plus its display hints (what the routes serve)."""
    out = dict(record)
    out["display"] = display_hints(str(record.get("kind", "")), record.get("payload") or {})
    return out


def annotations_for_step(
    events: Sequence[Mapping[str, Any]],
    step_name: str,
    step_start: Optional[float],
) -> List[Dict[str, Any]]:
    """Chart annotations for one step: x = elapsed seconds since step start."""
    annotations: List[Dict[str, Any]] = []
    for record in events:
        if record.get("step") != step_name:
            continue
        hints = display_hints(str(record.get("kind", "")), record.get("payload") or {})
        if not hints["categories"]:
            continue
        timestamp = record.get("timestamp")
        x = None
        if timestamp is not None and step_start is not None:
            x = max(0.0, float(timestamp) - float(step_start))
        annotations.append({
            "x": x,
            "kind": record.get("kind"),
            "label": hints["label"],
            "tone": hints["tone"],
            "categories": hints["categories"],
        })
    return annotations
