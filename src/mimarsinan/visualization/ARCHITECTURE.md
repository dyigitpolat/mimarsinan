# visualization/ — Pipeline Artifact Visualizations

Static visualization utilities for IR graphs, hardware mappings, activations, and search results.

## Subpackages

| Directory | Doc | Role |
|-----------|-----|------|
| `graphviz/` | [graphviz/ARCHITECTURE.md](graphviz/ARCHITECTURE.md) | DOT writers for IR / SCM / HCM / hybrid |
| `search_viz/` | [search_viz/ARCHITECTURE.md](search_viz/ARCHITECTURE.md) | Architecture search PNG/HTML reports |

## Other modules

| File | Role |
|------|------|
| `softcore_flowchart.py` | Mapper flowchart DOT + `estimate_fc_cores` |
| `activation_function_visualization.py` | Activation plots |
| `histogram_visualization.py` | Activation histograms |
| `hardcore_visualization.py` | Core heatmaps |
| `data_point_visualization.py` | Raw data plots |

## Dependencies

- **Internal**: `mapping.*`, `code_generation.cpp_chip_model`
- **External**: `matplotlib`, `plotly`, system `graphviz`

## Dependents

- `pipelining.pipeline_steps`, GUI snapshots
