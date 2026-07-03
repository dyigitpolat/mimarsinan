from __future__ import annotations

import os
from itertools import combinations
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt

from mimarsinan.visualization.search_viz.series import (
    finite_pairs,
    goal_by_metric,
    pareto_metric_series,
)

__all__ = ["plot_default_pareto_scatters", "plot_scatter"]


def plot_scatter(
    *,
    xs: Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=16, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_default_pareto_scatters(result_json: Dict[str, Any], out_dir: str) -> None:
    pareto = result_json.get("pareto_front", []) or []
    if not pareto:
        return

    goal_by_name = goal_by_metric(result_json)
    metric_names = list(goal_by_name.keys())
    if len(metric_names) < 2:
        return

    series: Dict[str, List[float | None]] = {
        name: pareto_metric_series(pareto, name) for name in metric_names
    }

    os.makedirs(out_dir, exist_ok=True)
    for name_a, name_b in combinations(metric_names, 2):
        goal_a = goal_by_name.get(name_a, "")
        goal_b = goal_by_name.get(name_b, "")
        xs, ys = finite_pairs(series[name_a], series[name_b])
        if not xs:
            continue
        plot_scatter(
            xs=xs,
            ys=ys,
            xlabel=f"{name_a} ({goal_a})",
            ylabel=f"{name_b} ({goal_b})",
            title=f"Pareto: {name_a} vs {name_b}",
            out_path=os.path.join(out_dir, f"scatter_{name_a}_vs_{name_b}.png"),
        )
