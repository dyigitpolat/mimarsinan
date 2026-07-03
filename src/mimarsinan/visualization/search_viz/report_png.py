from __future__ import annotations

import os
from itertools import combinations
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mimarsinan.visualization.search_viz.series import (
    best_metric_series,
    finite_pairs,
    goal_by_metric,
    nan_gapped,
    pareto_metric_series,
)

__all__ = ["write_search_report_png"]


def write_search_report_png(result_json: Dict[str, Any], out_path: str) -> None:
    """Single-file PNG report: objective history, 2D Pareto projections, and normalized 3D Pareto views."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    goal_by_name = goal_by_metric(result_json)

    pareto = result_json.get("pareto_front", []) or []
    hist = result_json.get("history", []) or []

    gens = [h["gen"] for h in hist if isinstance(h, dict) and "gen" in h]
    bests = [h.get("best", {}) if isinstance(h, dict) else {} for h in hist]

    def _normalize(vals: List[float | None], *, goal: str | None) -> List[float | None]:
        xs = [v for v in vals if v is not None and np.isfinite(v)]
        if not xs:
            return [None for _ in vals]
        vmin = float(min(xs))
        vmax = float(max(xs))
        if vmax == vmin:
            norm = [0.5 if v is not None else None for v in vals]
        else:
            norm = [((float(v) - vmin) / (vmax - vmin)) if v is not None else None for v in vals]
        if goal == "min":
            norm = [(1.0 - v) if v is not None else None for v in norm]
        return norm

    metric_names = list(goal_by_name.keys())
    if not metric_names:
        plt.close("all")
        return

    pareto_series = {name: pareto_metric_series(pareto, name) for name in metric_names}
    norm_series = {
        name: _normalize(pareto_series[name], goal=goal_by_name.get(name))
        for name in metric_names
    }

    scatter_pairs = list(combinations(metric_names, 2))

    triplets = []
    if len(metric_names) >= 3:
        first3 = metric_names[:3]
        for i in range(3):
            rotated = first3[i:] + first3[:i]
            xn, yn, zn = rotated
            triplets.append((xn, norm_series[xn], yn, norm_series[yn],
                             zn, norm_series[zn], zn, norm_series[zn]))

    views = [(22, 45), (22, 135), (22, 225), (22, 315)]

    n_hist = min(len(metric_names), 4)
    n_scatter = min(len(scatter_pairs), 4)
    n_3d_rows = len(triplets)
    nrows = (1 if n_hist else 0) + (1 if n_scatter else 0) + n_3d_rows
    if nrows == 0:
        return

    height_ratios = []
    if n_hist:
        height_ratios.append(1.2)
    if n_scatter:
        height_ratios.append(1.2)
    height_ratios.extend([1.6] * n_3d_rows)

    fig = plt.figure(figsize=(20, 4.5 * nrows))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=4,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.25,
    )

    fig.suptitle("NSGA-II Search Report (single file)", fontsize=14, y=0.995)

    current_row = 0

    if n_hist:
        hist_gs = gs[current_row, 0:4].subgridspec(1, n_hist, wspace=0.25)
        for i, name in enumerate(metric_names[:n_hist]):
            goal = goal_by_name.get(name, "")
            ax = fig.add_subplot(hist_gs[0, i])
            ys = best_metric_series(bests, name)
            ax.set_title(f"History: {name}", fontsize=11)
            if gens and not all(y is None for y in ys):
                ax.plot(gens, nan_gapped(ys), marker="o", linewidth=1.2, markersize=3.0)
            ax.set_xlabel("generation")
            ax.set_ylabel(f"{name} ({goal})")
            ax.grid(True, alpha=0.25)
        current_row += 1

    if n_scatter:
        p2_gs = gs[current_row, 0:4].subgridspec(1, n_scatter, wspace=0.25)
        for i, (na, nb) in enumerate(scatter_pairs[:n_scatter]):
            ga = goal_by_name.get(na, "")
            gb = goal_by_name.get(nb, "")
            ax = fig.add_subplot(p2_gs[0, i])
            xs, ys = finite_pairs(pareto_series[na], pareto_series[nb])
            ax.scatter(xs, ys, s=14, alpha=0.8)
            ax.set_title(f"Pareto: {na} vs {nb}", fontsize=11)
            ax.set_xlabel(f"{na} ({ga})")
            ax.set_ylabel(f"{nb} ({gb})")
            ax.grid(True, alpha=0.25)
        current_row += 1

    def _collect_xyz(xv, yv, zv, cv):
        X, Y, Z, C = [], [], [], []
        for a, b, c, d in zip(xv, yv, zv, cv):
            if a is None or b is None or c is None or d is None:
                continue
            X.append(float(a))
            Y.append(float(b))
            Z.append(float(c))
            C.append(float(d))
        return np.array(X), np.array(Y), np.array(Z), np.array(C)

    for row_idx, (xname, xv, yname, yv, zname, zv, cname, cv) in enumerate(triplets):
        X, Y, Z, C = _collect_xyz(xv, yv, zv, cv)
        if X.size == 0:
            continue

        last_sc = None
        for col_idx, (elev, azim) in enumerate(views):
            ax = cast(Any, fig.add_subplot(gs[current_row + row_idx, col_idx], projection="3d"))
            sc = ax.scatter(X, Y, Z, c=C, s=14, alpha=0.9, cmap="viridis")
            last_sc = sc

            if col_idx == 0 and X.size >= 3:
                try:
                    ax.plot_trisurf(X, Y, Z, color="lightgray", alpha=0.12, linewidth=0.2)
                except Exception:
                    pass

            ax.set_xlabel(f"{xname} (norm)", labelpad=6)
            ax.set_ylabel(f"{yname} (norm)", labelpad=6)
            ax.set_zlabel(f"{zname} (norm)", labelpad=6)
            ax.set_title(f"{xname}-{yname}-{zname} | elev={elev}, azim={azim}", fontsize=9, pad=8)
            ax.view_init(elev=elev, azim=azim)

        if last_sc is not None:
            ax_last = fig.axes[-1]
            cax = inset_axes(ax_last, width="3%", height="60%", loc="center right", borderpad=2.0)
            cb = fig.colorbar(last_sc, cax=cax)
            cb.set_label(f"{cname} (norm)", fontsize=8)
            cb.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

