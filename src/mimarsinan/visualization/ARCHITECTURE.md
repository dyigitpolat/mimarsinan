# visualization/ — Graphviz, matplotlib, and Plotly.js visualizations of pipeline artifacts

Write-to-file rendering of what the deployment pipeline produces at each stage:
IR graphs and soft/hard-core mappings as Graphviz DOT, core weight matrices and
activation statistics as matplotlib PNGs, and architecture-search results as
PNG/HTML reports. Everything here is a passive consumer — functions take a
finished artifact (an `IRGraph`, a `HardCoreMapping`, a search-result dict)
plus an output path and emit a file; nothing feeds back into the pipeline.

## Key files
| File | Purpose |
|---|---|
| `activation_function_visualization.py` | `ActivationFunctionVisualizer`: plot an activation function over a fixed input range to PNG. |
| `data_point_visualization.py` | `DataPointVisualizer`: grid plots of dataset samples as images or 1-D curves (currently unused by the pipeline). |
| `hardcore_visualization.py` | `HardCoreMappingVisualizer`: per-core weight-matrix heatmap grid for a `HardCoreMapping`, sized to axon/neuron dimensions. |
| `histogram_visualization.py` | `HistogramVisualizer`: bar plot of a precomputed activation histogram with optional threshold marker. |
| `softcore_flowchart_dot.py` | DOT flowchart of the mapper graph via forward shape tracing, annotating each node with its SW estimate and predicted HW core cost (no mapping required). |
| `softcore_flowchart_estimate.py` | FC-layer core-count estimator (`estimate_fc_cores`, `HWEstimate`) mirroring layout tiling modes: single, output-tiled, coalescing. |
| `graphviz/` | DOT writers for IR graphs (full + summary), softcore/hardcore mappings, and hybrid segment/combined views, plus shared label helpers and `try_render_dot`. |
| `search_viz/` | Architecture-search reporting: metric-history plots, Pareto scatters, summary PNG, final-population JSON, and an interactive Plotly.js HTML report. |

## Dependencies
- `mapping` — `mapping.ir` node types (`IRGraph`, `ComputeOp`, `NeuralCore`, ...) for IR DOT rendering; `mapping.packing` (`HardCoreMapping`, `HybridHardCoreMapping`) for hardcore/hybrid DOT writers; `mapping.mapping_utils` (`ModelRepresentation`, `Mapper`, `PerceptronMapper`) for flowchart tracing; `mapping.platform` (`coalescing_fragment_count`, `compute_core_input_count`, `compute_fc_tiling_mode`) so core estimates match real layout semantics.
- `code_generation` — `cpp_chip_model.SpikeSource` for decoding core input connectivity in softcore/hardcore DOT edges.
- `common` — `safe_numeric.safe_float` for tolerant numeric formatting in labels and metric series; `layer_key.layer_key_from_node_name` for grouping nodes in the IR summary DOT.

## Dependents
- `pipelining` — mapping pipeline steps (`soft_core_mapping_step`/`soft_core_mapping_viz`, `hard_core_mapping_step`) emit DOT/heatmap artifacts; `config/architecture_search_helpers` writes search reports.

## Exported API
`__init__.py` re-exports:
- Graphviz writers: `write_ir_graph_dot`, `write_ir_graph_summary_dot`, `write_softcore_mapping_dot`, `write_hardcore_mapping_dot`, `write_hybrid_hardcore_mapping_dots`, `write_hybrid_hardcore_mapping_combined_dot` — DOT files per artifact kind; `try_render_dot` — best-effort DOT→image via system `dot`; `HybridVizArtifacts` — output-path bundle for hybrid mappings.
- `write_softcore_flowchart_dot` — annotated mapper-graph flowchart.
- Matplotlib visualizers: `ActivationFunctionVisualizer`, `HistogramVisualizer`, `HardCoreMappingVisualizer`.
- Search reports: `write_search_report_png`, `create_interactive_search_report`, `write_final_population_json`, `plot_history_best_metrics`, `plot_history_metrics_separate`, `plot_default_pareto_scatters`.
