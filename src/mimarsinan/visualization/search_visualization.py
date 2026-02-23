from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plotly.subplots import make_subplots


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def write_final_population_json(result_json: Dict[str, Any], out_path: str) -> None:
    """
    Write a compact list of candidate configurations + objective values for quick inspection.
    """
    pop = []
    for c in result_json.get("pareto_front", []):
        row = {}
        row.update(c.get("configuration", {}).get("model_config", {}))
        row.update(c.get("configuration", {}).get("platform_constraints", {}))
        row["threshold_groups"] = c.get("configuration", {}).get("threshold_groups")
        row.update(c.get("objectives", {}))
        pop.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pop, f, indent=2)


def plot_history_best_metrics(result_json: Dict[str, Any], out_path: str) -> None:
    hist = result_json.get("history", []) or []
    if not hist:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    # Pull series if present
    series = {
        "accuracy": [b.get("accuracy") for b in bests],
        "wasted_area": [b.get("wasted_area") for b in bests],
        "total_params": [b.get("total_params") for b in bests],
    }

    plt.figure(figsize=(10, 6))
    for name, ys in series.items():
        ys2 = [_safe_float(y) for y in ys]
        if all(y is None for y in ys2):
            continue
        plt.plot(gens, ys2, label=name)

    plt.xlabel("generation")
    plt.ylabel("best metric value")
    plt.title("NSGA-II best objective values per generation (direction-aware)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_history_metrics_separate(result_json: Dict[str, Any], out_dir: str) -> None:
    """
    Write one plot per metric (best value per generation).
    """
    hist = result_json.get("history", []) or []
    if not hist:
        return

    gens = [h.get("gen") for h in hist if "gen" in h]
    bests = [h.get("best", {}) for h in hist]

    metrics = [
        ("accuracy", "accuracy (max)"),
        ("wasted_area", "wasted_area (min)"),
        ("total_params", "total_params (min)"),
    ]

    os.makedirs(out_dir, exist_ok=True)

    for key, label in metrics:
        ys = [_safe_float(b.get(key)) for b in bests]
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

    wasted = [_safe_float(c.get("objectives", {}).get("wasted_area")) for c in pareto]
    params = [_safe_float(c.get("objectives", {}).get("total_params")) for c in pareto]
    acc = [_safe_float(c.get("objectives", {}).get("accuracy")) for c in pareto]

    # Filter out obvious penalty values so plots remain readable.
    PENALTY_CUTOFF = 1e17
    for i in range(len(pareto)):
        if wasted[i] is not None and wasted[i] >= PENALTY_CUTOFF:
            wasted[i] = None
        if params[i] is not None and params[i] >= PENALTY_CUTOFF:
            params[i] = None

    # filter None pairs
    def _pairs(a, b):
        xs, ys = [], []
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys

    xs, ys = _pairs(wasted, acc)
    if xs:
        plot_scatter(
            xs=xs,
            ys=ys,
            xlabel="wasted_area (min)",
            ylabel="accuracy (max)",
            title="Pareto: wasted area vs accuracy",
            out_path=os.path.join(out_dir, "scatter_wasted_vs_acc.png"),
        )

    xs, ys = _pairs(params, acc)
    if xs:
        plot_scatter(
            xs=xs,
            ys=ys,
            xlabel="total_params (min)",
            ylabel="accuracy (max)",
            title="Pareto: params vs accuracy",
            out_path=os.path.join(out_dir, "scatter_params_vs_acc.png"),
        )

    xs, ys = _pairs(wasted, params)
    if xs:
        plot_scatter(
            xs=xs,
            ys=ys,
            xlabel="wasted_area (min)",
            ylabel="total_params (min)",
            title="Pareto: wasted area vs params",
            out_path=os.path.join(out_dir, "scatter_wasted_vs_params.png"),
        )


def write_search_report_png(result_json: Dict[str, Any], out_path: str) -> None:
    """
    Single-file report (PNG):
    - Per-generation objective history (subplots)
    - 2D Pareto projections (subplots)
    - Multiple normalized 3D Pareto surfaces/views (grid, higher=better)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    objectives = result_json.get("objectives", []) or []
    goal_by_name = {o.get("name"): o.get("goal") for o in objectives if isinstance(o, dict)}

    pareto = result_json.get("pareto_front", []) or []
    hist = result_json.get("history", []) or []

    gens = [h.get("gen") for h in hist if isinstance(h, dict) and "gen" in h]
    bests = [h.get("best", {}) if isinstance(h, dict) else {} for h in hist]

    def _hist_series(name: str):
        return [_safe_float(b.get(name)) for b in bests]

    PENALTY_CUTOFF = 1e17

    def _pareto_series(name: str):
        vals = []
        for c in pareto:
            obj = (c.get("objectives", {}) if isinstance(c, dict) else {}) or {}
            v = _safe_float(obj.get(name))
            if v is None or (v is not None and v >= PENALTY_CUTOFF):
                vals.append(None)
            else:
                vals.append(v)
        return vals

    def _pairs(a, b):
        xs, ys = [], []
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys

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
        # Flip minimization so higher is better for all axes
        if goal == "min":
            norm = [(1.0 - v) if v is not None else None for v in norm]
        return norm

    wasted = _pareto_series("wasted_area")
    params = _pareto_series("total_params")
    acc = _pareto_series("accuracy")

    n_wasted = _normalize(wasted, goal=goal_by_name.get("wasted_area"))
    n_params = _normalize(params, goal=goal_by_name.get("total_params"))
    n_acc = _normalize(acc, goal=goal_by_name.get("accuracy"))

    triplets = [
        ("wasted_area", n_wasted, "params", n_params, "accuracy", n_acc, "accuracy", n_acc),
        ("params", n_params, "accuracy", n_acc, "wasted_area", n_wasted, "wasted_area", n_wasted),
        ("wasted_area", n_wasted, "accuracy", n_acc, "params", n_params, "params", n_params),
    ]

    views = [(22, 45), (22, 135), (22, 225), (22, 315)]

    # Layout: 5 rows x 4 cols
    # - row 0: history 1x3
    # - row 1: 2D pareto 1x3
    # - rows 2-4: 3D triplets, one row per triplet, 4 views per row
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=[1.2, 1.2, 1.6, 1.6, 1.6],
        hspace=0.45,
        wspace=0.25,
    )

    fig.suptitle("NSGA-II Search Report (single file)", fontsize=14, y=0.995)

    # History block
    hist_gs = gs[0, 0:4].subgridspec(1, 3, wspace=0.25)
    hist_metrics = [
        ("accuracy", "accuracy (max)"),
        ("wasted_area", "wasted_area (min)"),
        ("total_params", "total_params (min)"),
    ]
    for i, (k, label) in enumerate(hist_metrics):
        ax = fig.add_subplot(hist_gs[0, i])
        ys = _hist_series(k)
        ax.set_title(f"History: {k}", fontsize=11)
        if gens and not all(y is None for y in ys):
            ax.plot(gens, ys, marker="o", linewidth=1.2, markersize=3.0)
        ax.set_xlabel("generation")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    # 2D Pareto block
    p2_gs = gs[1, 0:4].subgridspec(1, 3, wspace=0.25)
    p2 = [
        ("wasted_area (min)", wasted, "accuracy (max)", acc, "Pareto: wasted area vs accuracy"),
        ("total_params (min)", params, "accuracy (max)", acc, "Pareto: params vs accuracy"),
        ("wasted_area (min)", wasted, "total_params (min)", params, "Pareto: wasted area vs params"),
    ]
    for i, (xl, xv, yl, yv, title) in enumerate(p2):
        ax = fig.add_subplot(p2_gs[0, i])
        xs, ys = _pairs(xv, yv)
        ax.scatter(xs, ys, s=14, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.25)

    # 3D block: normalized (higher=better)
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

        # create 4 views in this row
        last_sc = None
        for col_idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(gs[2 + row_idx, col_idx], projection="3d")
            sc = ax.scatter(X, Y, Z, c=C, s=14, alpha=0.9, cmap="viridis")
            last_sc = sc

            # light surface on first view only (if possible)
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

        # One compact colorbar for the row, inset into the last axis
        if last_sc is not None:
            ax_last = fig.axes[-1]
            cax = inset_axes(ax_last, width="3%", height="60%", loc="center right", borderpad=2.0)
            cb = fig.colorbar(last_sc, cax=cax)
            cb.set_label(f"{cname} (norm)", fontsize=8)
            cb.ax.tick_params(labelsize=8)

    # Avoid tight_layout (3D axes don't play nice); manual spacing already applied.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_interactive_search_report(result_json: Dict[str, Any], out_path: str) -> None:
    """
    Generate a modern single-page interactive HTML report with Plotly:
    - History line charts (all metrics)
    - Two configurable 2D Pareto scatter plots side-by-side
    - Two configurable 3D Pareto visualizations with surface rendering
    - Comprehensive table of all evaluated candidates with Pareto/non-Pareto toggle
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    objectives = result_json.get("objectives", []) or []
    goal_by_name = {o.get("name"): o.get("goal") for o in objectives if isinstance(o, dict)}
    metric_names = list(goal_by_name.keys())

    pareto = result_json.get("pareto_front", []) or []
    all_candidates = result_json.get("all_candidates", []) or []
    hist = result_json.get("history", []) or []

    gens = [h.get("gen") for h in hist if isinstance(h, dict) and "gen" in h]
    bests = [h.get("best", {}) if isinstance(h, dict) else {} for h in hist]

    PENALTY_CUTOFF = 1e17

    def _hist_series(name: str):
        return [_safe_float(b.get(name)) for b in bests]

    def _extract_candidate_data(candidates, check_pareto=False):
        """Extract objective data from candidates list."""
        data = {name: [] for name in metric_names}
        data["is_pareto"] = []
        data["generation"] = []
        data["hover_info"] = []
        
        for c in candidates:
            obj = (c.get("objectives", {}) if isinstance(c, dict) else {}) or {}
            meta = (c.get("metadata", {}) if isinstance(c, dict) else {}) or {}
            
            is_pareto = meta.get("is_pareto", check_pareto)
            gen = meta.get("generation", -1)
            
            # Build hover info
            info_parts = [f"Gen: {gen}", f"Pareto: {'✓' if is_pareto else '✗'}"]
            info_parts.extend([f"{k}: {_safe_float(v):.4f}" for k, v in obj.items()])
            hover_info = "<br>".join(info_parts)
            
            valid = True
            for name in metric_names:
                v = _safe_float(obj.get(name))
                if v is None or v >= PENALTY_CUTOFF:
                    valid = False
                    break
                data[name].append(v)
            
            if valid:
                data["is_pareto"].append(is_pareto)
                data["generation"].append(gen)
                data["hover_info"].append(hover_info)
            else:
                # Remove partial data
                for name in metric_names:
                    if data[name] and len(data[name]) > len(data["is_pareto"]):
                        data[name].pop()
        
        return data

    # Extract data from all candidates (includes Pareto info in metadata)
    if all_candidates:
        candidate_data = _extract_candidate_data(all_candidates)
    else:
        # Fallback to pareto front only
        candidate_data = _extract_candidate_data(pareto, check_pareto=True)
    
    # Build table data from all candidates
    table_rows = []
    source_candidates = all_candidates if all_candidates else pareto
    for idx, c in enumerate(source_candidates):
        config = c.get("configuration", {}) if isinstance(c, dict) else {}
        model_cfg = config.get("model_config", {})
        platform_cfg = config.get("platform_constraints", {})
        objectives_data = c.get("objectives", {}) if isinstance(c, dict) else {}
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        
        is_pareto = meta.get("is_pareto", True if not all_candidates else False)
        gen = meta.get("generation", -1)
        
        # Skip invalid candidates
        has_penalty = any(_safe_float(v, 0) >= PENALTY_CUTOFF for v in objectives_data.values())
        if has_penalty:
            continue
        
        row = {
            "id": idx,
            "generation": gen,
            "is_pareto": is_pareto,
        }
        # Add model config
        for k, v in model_cfg.items():
            row[f"model_{k}"] = v
        # Add platform config summary
        cores = platform_cfg.get("cores", [])
        if cores:
            row["hw_core_types"] = len(cores)
            row["hw_max_axons"] = platform_cfg.get("max_axons", "")
            row["hw_max_neurons"] = platform_cfg.get("max_neurons", "")
        # Add objectives
        for k, v in objectives_data.items():
            val = _safe_float(v)
            if val is not None and val < PENALTY_CUTOFF:
                row[f"obj_{k}"] = val
        
        table_rows.append(row)

    # Create HTML with modern dark design
    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSGA-II Architecture Search Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --accent: #f59e0b;
            --success: #10b981;
            --surface: #0f172a;
            --surface-light: #1e293b;
            --surface-lighter: #334155;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #475569;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--surface);
            min-height: 100vh;
            color: var(--text);
        }
        .container { max-width: 1920px; margin: 0 auto; padding: 30px; }
        .header {
            background: linear-gradient(135deg, var(--surface-light) 0%, var(--surface) 100%);
            padding: 40px 50px;
            border-radius: 24px;
            border: 1px solid var(--border);
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--accent), var(--success));
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 8px;
        }
        .header h1 span {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p { color: var(--text-muted); font-size: 1.1rem; }
        .stats-bar {
            display: flex; gap: 30px; margin-top: 25px; padding-top: 25px;
            border-top: 1px solid var(--border);
        }
        .stat-item { text-align: left; }
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem; font-weight: 600; color: var(--accent);
        }
        .stat-label {
            font-size: 0.85rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .card {
            background: var(--surface-light);
            border-radius: 20px; padding: 30px; margin-bottom: 30px;
            border: 1px solid var(--border);
        }
        .card-title {
            font-size: 1.4rem; font-weight: 600; color: var(--text);
            margin-bottom: 25px; display: flex; align-items: center; gap: 12px;
        }
        .card-title::before {
            content: ''; width: 4px; height: 24px;
            background: linear-gradient(180deg, var(--primary), var(--accent));
            border-radius: 2px;
        }
        .controls {
            background: var(--surface); padding: 20px 25px;
            border-radius: 14px; margin-bottom: 25px;
            border: 1px solid var(--border);
        }
        .control-row { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; }
        .control-group { display: flex; align-items: center; gap: 10px; }
        label {
            font-weight: 500; color: var(--text-muted);
            font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em;
        }
        select {
            padding: 10px 14px; font-size: 14px;
            font-family: 'JetBrains Mono', monospace;
            border: 2px solid var(--border); border-radius: 10px;
            background: var(--surface-light); color: var(--text);
            cursor: pointer; transition: all 0.2s; min-width: 180px;
        }
        select:hover, select:focus { border-color: var(--primary); outline: none; }
        .toggle-btn {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 20px; background: var(--surface-lighter);
            border: 2px solid var(--border); border-radius: 10px;
            color: var(--text-muted); cursor: pointer;
            transition: all 0.2s; font-family: inherit; font-size: 14px;
        }
        .toggle-btn:hover { border-color: var(--primary); color: var(--text); }
        .toggle-btn.active { background: var(--primary); border-color: var(--primary); color: white; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
        .plot-panel {
            background: var(--surface); border-radius: 16px;
            padding: 20px; border: 1px solid var(--border);
        }
        .plot-panel-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 15px; padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }
        .plot-panel-title { font-size: 1rem; font-weight: 600; color: var(--text); }
        .plot-controls { display: flex; gap: 10px; align-items: center; }
        .plot-controls select { min-width: 140px; padding: 6px 10px; font-size: 12px; }
        .table-wrapper {
            background: var(--surface); border-radius: 16px;
            border: 1px solid var(--border); overflow: hidden;
        }
        .table-controls {
            padding: 20px; border-bottom: 1px solid var(--border);
            display: flex; gap: 15px; align-items: center; flex-wrap: wrap;
        }
        .search-input {
            flex: 1; min-width: 250px; padding: 12px 16px;
            border: 2px solid var(--border); border-radius: 10px;
            background: var(--surface-light);
            font-size: 14px; font-family: inherit; color: var(--text);
            transition: all 0.2s;
        }
        .search-input:focus { outline: none; border-color: var(--primary); }
        .search-input::placeholder { color: var(--text-muted); }
        .table-container { overflow-x: auto; max-height: 600px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        thead { background: var(--surface-light); position: sticky; top: 0; z-index: 10; }
        th {
            padding: 14px 16px; text-align: left; font-weight: 600;
            color: var(--text-muted); text-transform: uppercase;
            font-size: 0.75rem; letter-spacing: 0.05em;
            white-space: nowrap; cursor: pointer; user-select: none;
            border-bottom: 2px solid var(--border); transition: color 0.2s;
        }
        th:hover { color: var(--text); }
        th.sortable::after { content: ' ↕'; opacity: 0.4; }
        th.sorted-asc::after { content: ' ↑'; opacity: 1; color: var(--primary); }
        th.sorted-desc::after { content: ' ↓'; opacity: 1; color: var(--primary); }
        tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
        tbody tr:hover { background: var(--surface-lighter); }
        tbody tr.non-pareto { opacity: 0.45; }
        tbody tr.non-pareto:hover { opacity: 0.75; }
        td {
            padding: 14px 16px; color: var(--text);
            font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
        }
        .badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 10px; border-radius: 6px;
            font-size: 0.75rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.02em;
        }
        .badge-pareto { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
        .badge-other { background: rgba(148,163,184,0.1); color: var(--text-muted); border: 1px solid rgba(148,163,184,0.2); }
        .badge-gen { background: rgba(99,102,241,0.15); color: var(--primary); border: 1px solid rgba(99,102,241,0.3); }
        @media (max-width: 1400px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🧬 <span>NSGA-II</span> Architecture Search</h1>
        <p>Multi-Objective Neural Architecture & Hardware Co-Design Optimization</p>
        <div class="stats-bar">
            <div class="stat-item"><div class="stat-value">''' + str(len(gens)) + '''</div><div class="stat-label">Generations</div></div>
            <div class="stat-item"><div class="stat-value">''' + str(sum(1 for r in table_rows if r.get("is_pareto"))) + '''</div><div class="stat-label">Pareto Solutions</div></div>
            <div class="stat-item"><div class="stat-value">''' + str(len(table_rows)) + '''</div><div class="stat-label">Total Evaluated</div></div>
            <div class="stat-item"><div class="stat-value">''' + str(len(metric_names)) + '''</div><div class="stat-label">Objectives</div></div>
        </div>
    </div>
''')

    # History plot - separate subplot for each metric
    html_parts.append('''
    <div class="card">
        <div class="card-title">📈 Evolution History</div>
        <div class="grid-2" id="history-container">
''')
    for i, name in enumerate(metric_names):
        html_parts.append(f'            <div class="plot-panel"><div id="history-{i}"></div></div>\n')
    html_parts.append('''        </div>
    </div>
''')
    
    # 2D controls and plots - Two independent plots
    html_parts.append('''
    <div class="card">
        <div class="card-title">📊 2D Pareto Projections</div>
        <div class="controls">
            <div class="control-row">
                <button class="toggle-btn active" id="show-nonpareto-2d" onclick="toggle2DNonPareto()">
                    Show Non-Pareto
                </button>
            </div>
        </div>
        <div class="grid-2">
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">Plot A</span>
                    <div class="plot-controls">
                        <select id="x2d-a">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 0 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="y2d-a">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 1 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                    </div>
                </div>
                <div id="plot2d-a"></div>
            </div>
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">Plot B</span>
                    <div class="plot-controls">
                        <select id="x2d-b">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 2 % len(metric_names) else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="y2d-b">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 3 % len(metric_names) else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                    </div>
                </div>
                <div id="plot2d-b"></div>
            </div>
        </div>
    </div>
''')

    # 3D controls and plots - Two independent plots with surface
    html_parts.append('''
    <div class="card">
        <div class="card-title">🎯 3D Pareto Surface Visualization</div>
        <div class="controls">
            <div class="control-row">
                <button class="toggle-btn active" id="show-nonpareto-3d" onclick="toggle3DNonPareto()">
                    Show Non-Pareto
                </button>
                <button class="toggle-btn active" id="show-surface" onclick="toggleSurface()">
                    Show Pareto Surface
                </button>
            </div>
        </div>
        <div class="grid-2">
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">3D View A</span>
                    <div class="plot-controls">
                        <select id="x3d-a">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 0 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="y3d-a">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 1 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="z3d-a">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 2 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                    </div>
                </div>
                <div id="plot3d-a"></div>
            </div>
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">3D View B</span>
                    <div class="plot-controls">
                        <select id="x3d-b">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 0 else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="y3d-b">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 2 % len(metric_names) else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                        <select id="z3d-b">''')
    for i, name in enumerate(metric_names):
        selected = ' selected' if i == 3 % len(metric_names) else ''
        html_parts.append(f'<option value="{name}"{selected}>{name}</option>')
    html_parts.append('''</select>
                    </div>
                </div>
                <div id="plot3d-b"></div>
            </div>
        </div>
    </div>
''')
    
    # Candidates table with toggle for non-Pareto
    html_parts.append('''
    <div class="card">
        <div class="card-title">🔍 All Candidates Browser</div>
        <div class="table-wrapper">
            <div class="table-controls">
                <input type="text" class="search-input" id="table-search" placeholder="Search candidates..." onkeyup="filterTable()">
                <button class="toggle-btn active" id="show-nonpareto-table" onclick="toggleTableNonPareto()">
                    Show Non-Pareto
                </button>
            </div>
            <div class="table-container">
                <table id="candidates-table">
                    <thead>
                        <tr>
                            <th class="sortable" onclick="sortTable(0)">ID</th>
                            <th class="sortable" onclick="sortTable(1)">Gen</th>
                            <th class="sortable" onclick="sortTable(2)">Status</th>''')
    
    # Add headers for all columns in table_rows
    col_idx = 3
    if table_rows:
        sample_row = table_rows[0]
        for key in sorted(sample_row.keys()):
            if key not in ["id", "generation", "is_pareto"]:
                display_name = key.replace("_", " ").replace("model ", "").replace("obj ", "").replace("hw ", "HW ").title()
                html_parts.append(f'<th class="sortable" onclick="sortTable({col_idx})">{display_name}</th>')
                col_idx += 1
    
    html_parts.append('''
                        </tr>
                    </thead>
                    <tbody>''')
    
    # Add table rows
    for row in table_rows:
        is_pareto = row.get("is_pareto", False)
        row_class = "" if is_pareto else "non-pareto"
        html_parts.append(f'<tr class="{row_class}" data-pareto="{str(is_pareto).lower()}">')
        html_parts.append(f'<td>{row.get("id", "")}</td>')
        html_parts.append(f'<td><span class="badge badge-gen">G{row.get("generation", -1)}</span></td>')
        badge_class = "badge-pareto" if is_pareto else "badge-other"
        badge_text = "PARETO" if is_pareto else "OTHER"
        html_parts.append(f'<td><span class="badge {badge_class}">{badge_text}</span></td>')
        for key in sorted(row.keys()):
            if key not in ["id", "generation", "is_pareto"]:
                value = row.get(key, "")
                if isinstance(value, float):
                    html_parts.append(f'<td>{value:.4f}</td>')
                else:
                    html_parts.append(f'<td>{value}</td>')
        html_parts.append('</tr>')
    
    html_parts.append('''
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>
''')

    # JavaScript data and functions
    html_parts.append('<script>')
    
    # Embed data
    html_parts.append(f'const metricNames = {json.dumps(metric_names)};')
    html_parts.append(f'const candidateData = {json.dumps(candidate_data)};')
    html_parts.append(f'const historyGens = {json.dumps(gens)};')
    
    history_series_data = {name: _hist_series(name) for name in metric_names}
    html_parts.append(f'const historySeries = {json.dumps(history_series_data)};')
    
    html_parts.append('''
// State
let show2DNonPareto = true;
let show3DNonPareto = true;
let showSurface = true;
let showTableNonPareto = true;

// Dark theme layout
const darkLayout = {
    font: { family: 'Space Grotesk, sans-serif', size: 12, color: '#f1f5f9' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#0f172a',
    margin: { t: 40, r: 20, b: 50, l: 70 }
};

const gridStyle = { gridcolor: '#334155', gridwidth: 1, zerolinecolor: '#475569' };

// History plots - one per metric with its own scale
const historyColors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444'];
metricNames.forEach((name, idx) => {
    const trace = {
        x: historyGens,
        y: historySeries[name],
        mode: 'lines+markers',
        name: name,
        line: { width: 3, color: historyColors[idx % historyColors.length], shape: 'spline' },
        marker: { size: 8, color: historyColors[idx % historyColors.length] },
        fill: 'tozeroy',
        fillcolor: historyColors[idx % historyColors.length] + '15'
    };
    
    Plotly.newPlot('history-' + idx, [trace], {
        ...darkLayout,
        title: { text: name, font: { size: 14, color: '#f1f5f9' } },
        xaxis: { title: 'Generation', ...gridStyle },
        yaxis: { title: 'Best Value', ...gridStyle, tickformat: name === 'accuracy' ? '.2%' : '.3s' },
        hovermode: 'x unified',
        height: 280,
        showlegend: false
    });
});

// 2D plot functions
function get2DTraces(xMetric, yMetric, showNonPareto) {
    const traces = [];
    const x = candidateData[xMetric];
    const y = candidateData[yMetric];
    const isPareto = candidateData.is_pareto;
    const info = candidateData.hover_info;
    
    // Non-pareto (faded, behind)
    if (showNonPareto) {
        const npX = [], npY = [], npInfo = [];
        for (let i = 0; i < x.length; i++) {
            if (!isPareto[i]) { npX.push(x[i]); npY.push(y[i]); npInfo.push(info[i]); }
        }
        if (npX.length > 0) {
            traces.push({
                x: npX, y: npY, mode: 'markers', type: 'scatter', name: 'Non-Pareto',
                marker: { size: 8, color: '#475569', opacity: 0.3, line: { width: 0 } },
                text: npInfo, hovertemplate: '%{text}<extra></extra>'
            });
        }
    }
    
    // Pareto (solid, front)
    const pX = [], pY = [], pInfo = [];
    for (let i = 0; i < x.length; i++) {
        if (isPareto[i]) { pX.push(x[i]); pY.push(y[i]); pInfo.push(info[i]); }
    }
    if (pX.length > 0) {
        traces.push({
            x: pX, y: pY, mode: 'markers', type: 'scatter', name: 'Pareto',
            marker: { 
                size: 14, color: '#10b981', opacity: 0.9,
                line: { color: '#059669', width: 2 },
                symbol: 'diamond'
            },
            text: pInfo, hovertemplate: '%{text}<extra></extra>'
        });
    }
    return traces;
}

function update2DPlots() {
    // Plot A
    const xA = document.getElementById('x2d-a').value;
    const yA = document.getElementById('y2d-a').value;
    Plotly.react('plot2d-a', get2DTraces(xA, yA, show2DNonPareto), {
        ...darkLayout,
        xaxis: { title: xA, ...gridStyle },
        yaxis: { title: yA, ...gridStyle },
        hovermode: 'closest', height: 420, showlegend: false
    });
    
    // Plot B
    const xB = document.getElementById('x2d-b').value;
    const yB = document.getElementById('y2d-b').value;
    Plotly.react('plot2d-b', get2DTraces(xB, yB, show2DNonPareto), {
        ...darkLayout,
        xaxis: { title: xB, ...gridStyle },
        yaxis: { title: yB, ...gridStyle },
        hovermode: 'closest', height: 420, showlegend: false
    });
}

function toggle2DNonPareto() {
    show2DNonPareto = !show2DNonPareto;
    document.getElementById('show-nonpareto-2d').classList.toggle('active', show2DNonPareto);
    update2DPlots();
}

// Objective goals (min or max) - used to determine best values
const objectiveGoals = ''' + json.dumps(goal_by_name) + ''';

// 3D plot functions with Pareto surface and best-point markers
function get3DTraces(xMetric, yMetric, zMetric, showNonPareto, addSurface) {
    const traces = [];
    const x = candidateData[xMetric];
    const y = candidateData[yMetric];
    const z = candidateData[zMetric];
    const isPareto = candidateData.is_pareto;
    const info = candidateData.hover_info;
    
    // Non-pareto scatter
    if (showNonPareto) {
        const npX = [], npY = [], npZ = [], npInfo = [];
        for (let i = 0; i < x.length; i++) {
            if (!isPareto[i]) { npX.push(x[i]); npY.push(y[i]); npZ.push(z[i]); npInfo.push(info[i]); }
        }
        if (npX.length > 0) {
            traces.push({
                x: npX, y: npY, z: npZ, mode: 'markers', type: 'scatter3d', name: 'Non-Pareto',
                marker: { size: 4, color: '#475569', opacity: 0.25 },
                text: npInfo, hovertemplate: '%{text}<extra></extra>'
            });
        }
    }
    
    // Pareto scatter
    const pX = [], pY = [], pZ = [], pInfo = [];
    for (let i = 0; i < x.length; i++) {
        if (isPareto[i]) { pX.push(x[i]); pY.push(y[i]); pZ.push(z[i]); pInfo.push(info[i]); }
    }
    
    // Calculate best values for each axis (considering goal direction)
    function getBest(arr, metric) {
        if (arr.length === 0) return null;
        const goal = objectiveGoals[metric] || 'min';
        return goal === 'max' ? Math.max(...arr) : Math.min(...arr);
    }
    
    const bestX = getBest(pX, xMetric);
    const bestY = getBest(pY, yMetric);
    const bestZ = getBest(pZ, zMetric);
    
    // Find indices of best points for each axis
    const bestXIdx = pX.findIndex(v => v === bestX);
    const bestYIdx = pY.findIndex(v => v === bestY);
    const bestZIdx = pZ.findIndex(v => v === bestZ);
    
    // Draw faded lines from best points to all Pareto candidates
    if (pX.length > 0 && bestX !== null) {
        const lineColors = ['#ef4444', '#22c55e', '#3b82f6']; // X=red, Y=green, Z=blue
        const bestPoints = [
            { idx: bestXIdx, color: lineColors[0], label: 'Best ' + xMetric },
            { idx: bestYIdx, color: lineColors[1], label: 'Best ' + yMetric },
            { idx: bestZIdx, color: lineColors[2], label: 'Best ' + zMetric }
        ];
        
        // Remove duplicates (if same point is best for multiple axes)
        const uniqueBests = [];
        const seenIdx = new Set();
        bestPoints.forEach(bp => {
            if (!seenIdx.has(bp.idx)) {
                seenIdx.add(bp.idx);
                uniqueBests.push(bp);
            }
        });
        
        // Draw lines from each best point to all Pareto candidates
        bestPoints.forEach((bp, bpIdx) => {
            if (bp.idx < 0) return;
            const bx = pX[bp.idx], by = pY[bp.idx], bz = pZ[bp.idx];
            
            // Lines to all Pareto points
            for (let i = 0; i < pX.length; i++) {
                if (i === bp.idx) continue;
                traces.push({
                    x: [bx, pX[i]], y: [by, pY[i]], z: [bz, pZ[i]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: bp.color, width: 2 },
                    opacity: 0.15,
                    hoverinfo: 'skip',
                    showlegend: false
                });
            }
        });
        
        // Add best point markers (stars)
        const starX = [], starY = [], starZ = [], starColors = [], starLabels = [];
        bestPoints.forEach(bp => {
            if (bp.idx >= 0) {
                starX.push(pX[bp.idx]);
                starY.push(pY[bp.idx]);
                starZ.push(pZ[bp.idx]);
                starColors.push(bp.color);
                starLabels.push(bp.label);
            }
        });
        
        if (starX.length > 0) {
            traces.push({
                x: starX, y: starY, z: starZ,
                mode: 'markers', type: 'scatter3d', name: 'Best Points',
                marker: { 
                    size: 14, 
                    color: starColors,
                    symbol: 'diamond',
                    line: { color: '#fff', width: 2 }
                },
                text: starLabels,
                hovertemplate: '%{text}<extra></extra>'
            });
        }
    }
    
    // Pareto surface (mesh3d with Delaunay triangulation)
    if (addSurface && pX.length >= 3) {
        traces.push({
            x: pX, y: pY, z: pZ, type: 'mesh3d', name: 'Pareto Surface',
            opacity: 0.35,
            color: '#6366f1',
            flatshading: true,
            lighting: { ambient: 0.8, diffuse: 0.5, specular: 0.3 },
            hoverinfo: 'skip'
        });
    }
    
    // Pareto points on top
    if (pX.length > 0) {
        traces.push({
            x: pX, y: pY, z: pZ, mode: 'markers', type: 'scatter3d', name: 'Pareto',
            marker: { 
                size: 7, 
                color: pZ,
                colorscale: [[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
                opacity: 1,
                line: { color: '#fff', width: 1 },
                showscale: true,
                colorbar: { thickness: 15, len: 0.6, tickfont: { color: '#94a3b8' } }
            },
            text: pInfo, hovertemplate: '%{text}<extra></extra>'
        });
    }
    
    return traces;
}

function update3DPlots() {
    const sceneStyle = {
        xaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        yaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        zaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
    };
    
    // Plot A
    const xA = document.getElementById('x3d-a').value;
    const yA = document.getElementById('y3d-a').value;
    const zA = document.getElementById('z3d-a').value;
    Plotly.react('plot3d-a', get3DTraces(xA, yA, zA, show3DNonPareto, showSurface), {
        ...darkLayout,
        scene: { ...sceneStyle, xaxis: { ...sceneStyle.xaxis, title: xA }, yaxis: { ...sceneStyle.yaxis, title: yA }, zaxis: { ...sceneStyle.zaxis, title: zA } },
        height: 550, showlegend: false
    });
    
    // Plot B
    const xB = document.getElementById('x3d-b').value;
    const yB = document.getElementById('y3d-b').value;
    const zB = document.getElementById('z3d-b').value;
    Plotly.react('plot3d-b', get3DTraces(xB, yB, zB, show3DNonPareto, showSurface), {
        ...darkLayout,
        scene: { ...sceneStyle, xaxis: { ...sceneStyle.xaxis, title: xB }, yaxis: { ...sceneStyle.yaxis, title: yB }, zaxis: { ...sceneStyle.zaxis, title: zB } },
        height: 550, showlegend: false
    });
}

function toggle3DNonPareto() {
    show3DNonPareto = !show3DNonPareto;
    document.getElementById('show-nonpareto-3d').classList.toggle('active', show3DNonPareto);
    update3DPlots();
}

function toggleSurface() {
    showSurface = !showSurface;
    document.getElementById('show-surface').classList.toggle('active', showSurface);
    update3DPlots();
}

// Table functions
function toggleTableNonPareto() {
    showTableNonPareto = !showTableNonPareto;
    document.getElementById('show-nonpareto-table').classList.toggle('active', showTableNonPareto);
    const rows = document.querySelectorAll('#candidates-table tbody tr.non-pareto');
    rows.forEach(row => {
        row.style.display = showTableNonPareto ? '' : 'none';
    });
}

function filterTable() {
    const filter = document.getElementById('table-search').value.toUpperCase();
    const rows = document.querySelectorAll('#candidates-table tbody tr');
    rows.forEach(row => {
        const isNonPareto = row.classList.contains('non-pareto');
        if (!showTableNonPareto && isNonPareto) {
            row.style.display = 'none';
            return;
        }
        const text = row.textContent || row.innerText;
        row.style.display = text.toUpperCase().includes(filter) ? '' : 'none';
    });
}

let sortDirection = {};
function sortTable(columnIndex) {
    const table = document.getElementById('candidates-table');
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = Array.from(tbody.getElementsByTagName('tr'));
    
    const direction = sortDirection[columnIndex] === 'asc' ? 'desc' : 'asc';
    sortDirection[columnIndex] = direction;
    
    const headers = table.getElementsByTagName('th');
    for (let i = 0; i < headers.length; i++) headers[i].className = 'sortable';
    headers[columnIndex].className = direction === 'asc' ? 'sortable sorted-asc' : 'sortable sorted-desc';
    
    rows.sort((a, b) => {
        const aValue = a.getElementsByTagName('td')[columnIndex].textContent;
        const bValue = b.getElementsByTagName('td')[columnIndex].textContent;
        const aNum = parseFloat(aValue), bNum = parseFloat(bValue);
        if (!isNaN(aNum) && !isNaN(bNum)) return direction === 'asc' ? aNum - bNum : bNum - aNum;
        return direction === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

// Event listeners for select changes
['x2d-a', 'y2d-a', 'x2d-b', 'y2d-b'].forEach(id => {
    document.getElementById(id).addEventListener('change', update2DPlots);
});
['x3d-a', 'y3d-a', 'z3d-a', 'x3d-b', 'y3d-b', 'z3d-b'].forEach(id => {
    document.getElementById(id).addEventListener('change', update3DPlots);
});

// Initial render
update2DPlots();
update3DPlots();
</script>
</body>
</html>
''')

    with open(out_path, 'w') as f:
        f.write(''.join(html_parts))





