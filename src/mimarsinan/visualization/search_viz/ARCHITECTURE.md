# visualization/search_viz/ — Architecture Search Reports

Static PNG and interactive HTML reports for NSGA-II / search results.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `population.py` | `write_final_population_json` | Compact JSON export of Pareto front |
| `series.py` | `goal_by_metric`, `best_metric_series`, `pareto_metric_series`, `finite_pairs`, `nan_gapped`, `PENALTY_CUTOFF` | Shared metric-series extraction/cleaning used by all report writers |
| `history.py` | `plot_history_*` | Per-generation best-metric plots |
| `scatter.py` | `plot_scatter`, `plot_default_pareto_scatters` | 2D Pareto projections |
| `report_png.py` | `write_search_report_png` | Single-file matplotlib report |
| `html/template.py` | `create_interactive_search_report` | Plotly HTML dashboard assembly |
| `html/data.py` | `parse_search_result`, `ReportData` | Result JSON parsing |
| `html/sections_*.py` | section renderers | History, Pareto, table, layout HTML/JS |

## Dependents

- `pipelining.pipeline_steps.config.architecture_search_step`
