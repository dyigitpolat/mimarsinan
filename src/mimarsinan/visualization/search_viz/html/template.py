"""HTML shell and report assembly for interactive search reports."""

from __future__ import annotations

import os
from typing import Any, Dict

from mimarsinan.visualization.search_viz.html.data import ReportData, parse_search_result
from mimarsinan.visualization.search_viz.html.sections_history import (
    render_history_script,
    render_history_section,
    render_pareto_2d_script,
)
from mimarsinan.visualization.search_viz.html.sections_layout import PAGE_STYLES, render_layout_section
from mimarsinan.visualization.search_viz.html.sections_pareto import (
    render_pareto_2d_section,
    render_pareto_3d_section,
    render_pareto_3d_script,
    render_pareto_data_script,
)
from mimarsinan.visualization.search_viz.html.sections_table import (
    render_table_script,
    render_table_section,
)

__all__ = ["create_interactive_search_report"]


def _render_header(data: ReportData) -> str:
    pareto_count = sum(1 for row in data.table_rows if row.get("is_pareto"))
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSGA-II Architecture Search Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>{PAGE_STYLES}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🧬 <span>NSGA-II</span> Architecture Search</h1>
        <p>Multi-Objective Neural Architecture & Hardware Co-Design Optimization</p>
        <div class="stats-bar">
            <div class="stat-item"><div class="stat-value">{len(data.gens)}</div><div class="stat-label">Generations</div></div>
            <div class="stat-item"><div class="stat-value">{pareto_count}</div><div class="stat-label">Pareto Solutions</div></div>
            <div class="stat-item"><div class="stat-value">{len(data.table_rows)}</div><div class="stat-label">Total Evaluated</div></div>
            <div class="stat-item"><div class="stat-value">{len(data.metric_names)}</div><div class="stat-label">Objectives</div></div>
        </div>
    </div>
'''


def _render_script(data: ReportData) -> str:
    return f"""<script>
let showTableNonPareto = true;

const darkLayout = {{
    font: {{ family: 'Space Grotesk, sans-serif', size: 12, color: '#f1f5f9' }},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#0f172a',
    margin: {{ t: 40, r: 20, b: 50, l: 70 }}
}};

const gridStyle = {{ gridcolor: '#334155', gridwidth: 1, zerolinecolor: '#475569' }};

{render_history_script(data.metric_names, data.gens, data.history_series)}
{render_pareto_data_script(data.metric_names, data.candidate_data, data.goal_by_name)}
{render_pareto_2d_script()}
{render_pareto_3d_script()}
{render_table_script()}
</script>
</body>
</html>
"""


def create_interactive_search_report(result_json: Dict[str, Any], out_path: str) -> None:
    """Generate a single-page Plotly HTML dashboard for search results."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = parse_search_result(result_json)

    html_parts = [
        _render_header(data),
        render_history_section(data.metric_names),
        render_pareto_2d_section(data.metric_names),
        render_pareto_3d_section(data.metric_names),
        render_table_section(data.table_rows),
        render_layout_section(data.pareto),
        """
</div>
""",
        _render_script(data),
    ]

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("".join(html_parts))
