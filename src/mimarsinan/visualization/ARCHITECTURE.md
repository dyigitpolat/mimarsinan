# visualization/ -- Pipeline Artifact Visualizations

Static visualization utilities for inspecting pipeline artifacts at various
stages: IR graphs, hardware mappings, activation functions, weight
distributions, and architecture search results.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `mapping_graphviz.py` | `write_ir_graph_dot`, `write_ir_graph_summary_dot`, `write_hardcore_mapping_dot`, `write_hybrid_hardcore_mapping_dots`, etc. | Graphviz DOT generation for IR, SoftCore, HardCore, and Hybrid mappings |
| `softcore_flowchart.py` | `write_softcore_flowchart_dot` | Flowchart-style DOT for mapper graph with hardware estimates |
| `activation_function_visualization.py` | `ActivationFunctionVisualizer` | Plots activation functions over a range |
| `histogram_visualization.py` | `HistogramVisualizer` | Activation distribution histograms |
| `hardcore_visualization.py` | `HardCoreMappingVisualizer` | Heatmaps and utilization charts for hardware cores |
| `search_visualization.py` | `write_search_report_png`, `create_interactive_search_report`, etc. | Architecture search result plots (Pareto fronts, history) |
| `data_point_visualization.py` | `DataPointVisualizer` | Raw data point visualization |

## Dependencies

- **Internal**: `mapping.ir`, `mapping.softcore_mapping`, `mapping.hybrid_hardcore_mapping`, `mapping.mapping_utils`, `code_generation.cpp_chip_model`.
- **External**: `matplotlib`, `plotly`, `graphviz` (system binary), `numpy`.

## Dependents

- `pipelining.pipeline_steps` (soft core mapping, hard core mapping, architecture search steps).

## Exported API (\_\_init\_\_.py)

All primary visualization functions and classes are re-exported.
