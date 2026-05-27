# visualization/search_viz/ — Architecture Search Reports

Static PNG and interactive HTML reports for NSGA-II / search results.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `population.py` | `write_final_population_json` | Compact JSON export of Pareto front |
| `history.py` | `plot_history_*` | Per-generation best-metric plots |
| `scatter.py` | `plot_scatter`, `plot_default_pareto_scatters` | 2D Pareto projections |
| `report_png.py` | `write_search_report_png` | Single-file matplotlib report |
| `report_html.py` | `create_interactive_search_report` | Plotly HTML dashboard |

## Dependents

- `pipelining.pipeline_steps.config.architecture_search_step`
