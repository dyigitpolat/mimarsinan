from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib.pyplot as plt

from mimarsinan.visualization.search_viz.series import (
    best_metric_series,
    goal_by_metric,
    nan_gapped,
)

__all__ = ["plot_history_best_metrics", "plot_history_metrics_separate"]


def plot_history_best_metrics(result_json: Dict[str, Any], out_path: str) -> None:
    hist = result_json.get("history", []) or []
    if not hist:
        return

    metric_names = list(goal_by_metric(result_json).keys())
    if not metric_names:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    plt.figure(figsize=(10, 6))
    for name in metric_names:
        ys = best_metric_series(bests, name)
        if all(y is None for y in ys):
            continue
        plt.plot(gens, nan_gapped(ys), label=name)

    plt.xlabel("generation")
    plt.ylabel("best metric value")
    plt.title("NSGA-II best objective values per generation (direction-aware)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_history_metrics_separate(result_json: Dict[str, Any], out_dir: str) -> None:
    """Write one plot per metric (best value per generation)."""
    hist = result_json.get("history", []) or []
    if not hist:
        return

    goal_by_name = goal_by_metric(result_json)
    if not goal_by_name:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    os.makedirs(out_dir, exist_ok=True)

    for key, goal in goal_by_name.items():
        label = f"{key} ({goal})"
        ys = best_metric_series(bests, key)
        if all(y is None for y in ys):
            continue

        plt.figure(figsize=(8, 4.8))
        plt.plot(gens, nan_gapped(ys), marker="o", linewidth=1.5, markersize=3.5)
        plt.xlabel("generation")
        plt.ylabel(label)
        plt.title(f"NSGA-II best {key} per generation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"history_{key}.png"), dpi=140)
        plt.close()
