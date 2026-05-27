from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib.pyplot as plt

from mimarsinan.common.safe_numeric import safe_float

__all__ = ["plot_history_best_metrics", "plot_history_metrics_separate"]


def plot_history_best_metrics(result_json: Dict[str, Any], out_path: str) -> None:
    hist = result_json.get("history", []) or []
    if not hist:
        return

    objectives = result_json.get("objectives", []) or []
    metric_names = [o.get("name") for o in objectives if isinstance(o, dict) and o.get("name")]
    if not metric_names:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    plt.figure(figsize=(10, 6))
    for name in metric_names:
        ys = [safe_float(b.get(name)) for b in bests]
        if all(y is None for y in ys):
            continue
        plt.plot(gens, ys, label=name)

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

    objectives = result_json.get("objectives", []) or []
    goal_by_name = {o.get("name"): o.get("goal") for o in objectives if isinstance(o, dict)}
    if not goal_by_name:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    os.makedirs(out_dir, exist_ok=True)

    for key, goal in goal_by_name.items():
        label = f"{key} ({goal})"
        ys = [safe_float(b.get(key)) for b in bests]
        if all(y is None for y in ys):
            continue

        plt.figure(figsize=(8, 4.8))
        plt.plot(gens, ys, marker="o", linewidth=1.5, markersize=3.5)
        plt.xlabel("generation")
        plt.ylabel(label)
        plt.title(f"NSGA-II best {key} per generation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"history_{key}.png"), dpi=140)
        plt.close()
