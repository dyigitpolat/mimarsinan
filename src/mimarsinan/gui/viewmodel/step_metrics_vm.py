"""Metric-name categorization for step-detail charts (out of the JS)."""

from __future__ import annotations

from typing import Dict, Iterable


def metric_category(name: str) -> str:
    """The chart a metric renders on. One rule table, served to the frontend."""
    lowered = name.lower()
    if "loss" in lowered:
        return "Loss"
    if "adaptation target" in lowered:
        return "Accuracy"
    if "accuracy" in lowered or "acc" in lowered:
        return "Accuracy"
    if lowered == "lr" or "learning rate" in lowered:
        return "Learning Rate"
    if "adaptation" in lowered or "tuning rate" in lowered:
        return "Adaptation"
    if "search" in lowered:
        return f"Search: {name}"
    return "Other"


def categories_for(names: Iterable[str]) -> Dict[str, str]:
    return {name: metric_category(name) for name in names if name != "search_event"}
