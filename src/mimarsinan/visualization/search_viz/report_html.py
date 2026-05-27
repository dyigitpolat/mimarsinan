from __future__ import annotations

import json
import os
from itertools import combinations
from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mimarsinan.common.safe_numeric import safe_float

__all__ = ["create_interactive_search_report"]


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
        return [safe_float(b.get(name)) for b in bests]

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
            info_parts.extend([f"{k}: {safe_float(v):.4f}" for k, v in obj.items()])
            hover_info = "<br>".join(info_parts)
            
            valid = True
            for name in metric_names:
                v = safe_float(obj.get(name))
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
        has_penalty = any(safe_float(v, 0) >= PENALTY_CUTOFF for v in objectives_data.values())
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
            row["hw_max_axons"] = max((int(ct.get("max_axons", 0)) for ct in cores), default="")
            row["hw_max_neurons"] = max((int(ct.get("max_neurons", 0)) for ct in cores), default="")
        # Add objectives
        for k, v in objectives_data.items():
            val = safe_float(v)
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
''')

    # Compilagent-only "Layout details" panel. Renders one collapsible
    # block per Pareto candidate that carries `metadata.layout` (set by
    # `CompilagentOptimizer._build_result`). AgentEvolve / NSGA2
    # candidates lack the field, so the panel renders empty for those
    # runs — by design.
    layout_pareto = [
        c for c in (pareto or [])
        if isinstance(c, dict)
        and isinstance(c.get("metadata"), dict)
        and isinstance(c["metadata"].get("layout"), dict)
    ]
    if layout_pareto:
        html_parts.append('''
    <div class="card">
        <div class="card-title">🧩 Layout Details (Compilagent)</div>
        <div style="padding: 0 16px 16px; max-height: 600px; overflow-y: auto;">
''')
        for idx, c in enumerate(layout_pareto):
            layout = c["metadata"]["layout"]
            summary = layout.get("summary", {}) or {}
            per_layer = layout.get("per_layer", []) or []
            sc_count = layout.get("softcore_count", 0)
            frag = summary.get("fragmentation_pct", 0) or 0
            try:
                frag_str = f"{float(frag):.2f}"
            except Exception:
                frag_str = str(frag)
            html_parts.append(
                f'<details style="margin-bottom: 12px; padding: 8px; '
                f'border: 1px solid #334155; border-radius: 4px;">'
                f'<summary style="cursor: pointer; font-weight: 500;">'
                f'Candidate #{idx} — {sc_count} softcores, '
                f'{len(per_layer)} layers, {frag_str}% fragmentation'
                f'</summary>'
                f'<div style="padding: 8px 0;">'
            )
            html_parts.append('<table style="width: 100%; margin-bottom: 12px;"><tbody>')
            summary_rows = [
                ("Total cores used", summary.get("total_cores")),
                ("Total softcores", summary.get("total_softcores")),
                ("Neural segments", summary.get("neural_segment_count")),
                ("Threshold groups", summary.get("threshold_group_count")),
                ("Fragmentation %", summary.get("fragmentation_pct")),
                ("Mapped params %", summary.get("mapped_params_pct")),
                ("Schedule passes", summary.get("schedule_pass_count")),
                ("Sync barriers", summary.get("schedule_sync_count")),
            ]
            for label, val in summary_rows:
                if isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val) if val is not None else "-"
                html_parts.append(
                    f'<tr><td style="padding: 4px;">{label}</td>'
                    f'<td style="padding: 4px;">{val_str}</td></tr>'
                )
            html_parts.append('</tbody></table>')
            if per_layer:
                html_parts.append('<div style="font-weight: 500; margin-bottom: 6px;">Per-layer breakdown</div>')
                html_parts.append(
                    '<table style="width: 100%; font-size: 0.9em;"><thead><tr>'
                    '<th>Layer</th><th>Softcores</th><th>Total Area</th>'
                    '<th>Max In</th><th>Max Out</th><th>TG</th>'
                    '<th>Lat</th><th>Seg</th></tr></thead><tbody>'
                )
                for row in per_layer:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td>{row.get("layer", "-")}</td>')
                    html_parts.append(f'<td>{row.get("softcore_count", 0)}</td>')
                    html_parts.append(f'<td>{row.get("total_area", 0)}</td>')
                    html_parts.append(f'<td>{row.get("max_input_count", 0)}</td>')
                    html_parts.append(f'<td>{row.get("max_output_count", 0)}</td>')
                    html_parts.append(f'<td>{row.get("threshold_group_count", 0)}</td>')
                    html_parts.append(f'<td>{row.get("latency_tag_count", 0)}</td>')
                    html_parts.append(f'<td>{row.get("segment_count", 0)}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</tbody></table>')
            html_parts.append('</div></details>')
        html_parts.append('''
        </div>
    </div>
''')

    html_parts.append('''
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





