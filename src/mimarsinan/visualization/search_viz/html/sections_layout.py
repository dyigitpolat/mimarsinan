"""Page styles and Compilagent layout section for the interactive search report."""

from __future__ import annotations

from typing import Any, List

PAGE_STYLES = """
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
"""


def render_layout_section(pareto: List[Any]) -> str:
    layout_pareto = [
        c
        for c in (pareto or [])
        if isinstance(c, dict)
        and isinstance(c.get("metadata"), dict)
        and isinstance(c["metadata"].get("layout"), dict)
    ]
    if not layout_pareto:
        return ""

    parts = [
        '''
    <div class="card">
        <div class="card-title">🧩 Layout Details (Compilagent)</div>
        <div style="padding: 0 16px 16px; max-height: 600px; overflow-y: auto;">
''',
    ]

    for idx, candidate in enumerate(layout_pareto):
        layout = candidate["metadata"]["layout"]
        summary = layout.get("summary", {}) or {}
        per_layer = layout.get("per_layer", []) or []
        sc_count = layout.get("softcore_count", 0)
        frag = summary.get("fragmentation_pct", 0) or 0
        try:
            frag_str = f"{float(frag):.2f}"
        except Exception:
            frag_str = str(frag)
        parts.append(
            f'<details style="margin-bottom: 12px; padding: 8px; '
            f'border: 1px solid #334155; border-radius: 4px;">'
            f'<summary style="cursor: pointer; font-weight: 500;">'
            f"Candidate #{idx} — {sc_count} softcores, "
            f"{len(per_layer)} layers, {frag_str}% fragmentation"
            f"</summary>"
            f'<div style="padding: 8px 0;">'
        )
        parts.append('<table style="width: 100%; margin-bottom: 12px;"><tbody>')
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
            parts.append(
                f'<tr><td style="padding: 4px;">{label}</td>'
                f'<td style="padding: 4px;">{val_str}</td></tr>'
            )
        parts.append("</tbody></table>")
        if per_layer:
            parts.append('<div style="font-weight: 500; margin-bottom: 6px;">Per-layer breakdown</div>')
            parts.append(
                '<table style="width: 100%; font-size: 0.9em;"><thead><tr>'
                "<th>Layer</th><th>Softcores</th><th>Total Area</th>"
                "<th>Max In</th><th>Max Out</th><th>TG</th>"
                "<th>Lat</th><th>Seg</th></tr></thead><tbody>"
            )
            for row in per_layer:
                parts.append("<tr>")
                parts.append(f'<td>{row.get("layer", "-")}</td>')
                parts.append(f'<td>{row.get("softcore_count", 0)}</td>')
                parts.append(f'<td>{row.get("total_area", 0)}</td>')
                parts.append(f'<td>{row.get("max_input_count", 0)}</td>')
                parts.append(f'<td>{row.get("max_output_count", 0)}</td>')
                parts.append(f'<td>{row.get("threshold_group_count", 0)}</td>')
                parts.append(f'<td>{row.get("latency_tag_count", 0)}</td>')
                parts.append(f'<td>{row.get("segment_count", 0)}</td>')
                parts.append("</tr>")
            parts.append("</tbody></table>")
        parts.append("</div></details>")

    parts.append(
        """
        </div>
    </div>
"""
    )
    return "".join(parts)
