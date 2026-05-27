# visualization/search_viz/html/ — Interactive Search Report

Plotly HTML dashboard split from the former monolithic `report_html.py`.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `template.py` | `create_interactive_search_report` | Page assembly and file write |
| `data.py` | `parse_search_result`, `ReportData` | Result JSON parsing |
| `sections_history.py` | `render_history_*` | Evolution history panel |
| `sections_pareto.py` | `render_pareto_*` | 2D/3D Pareto panels and 3D JS |
| `sections_table.py` | `render_table_*` | Candidates browser table |
| `sections_layout.py` | `PAGE_STYLES`, `render_layout_section` | Shared CSS and Compilagent layout panel |

## Dependents

- `visualization.search_viz` (re-exports `create_interactive_search_report`)
- `pipelining.pipeline_steps.config.architecture_search_step`
