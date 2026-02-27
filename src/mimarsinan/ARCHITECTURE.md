# mimarsinan/ -- Package Root

This is the top-level Python package for the Mimarsinan framework.

For the comprehensive architecture documentation, see the project-level
[ARCHITECTURE.md](../../ARCHITECTURE.md) which covers:
- High-level overview and pipeline architecture
- All submodule descriptions and their interactions
- Dependency graph, conventions, and contributing guide

Each subdirectory has its own `ARCHITECTURE.md` describing that module's
purpose, components, dependencies, and exported API.

## Submodule Map

| Module | Purpose |
|--------|---------|
| `common/` | Shared utilities (file I/O, compiler discovery, WandB) |
| `data_handling/` | Dataset providers, factories, and data loaders |
| `model_training/` | Training loops and utilities |
| `models/` | Neural network models, layers, spiking simulators |
| `mapping/` | Model-to-hardware mapping (IR, mapper graph, core packing) |
| `transformations/` | Weight/activation quantization and fusion transforms |
| `tuning/` | Progressive adaptation and tuning framework |
| `search/` | Multi-objective architecture search |
| `pipelining/` | Pipeline engine, steps, and cache |
| `chip_simulation/` | Nevresim C++ simulator interface |
| `code_generation/` | C++ code generation for nevresim |
| `visualization/` | Graphviz, matplotlib, Plotly visualizations |
| `gui/` | Browser-based pipeline monitoring dashboard |
| `model_evaluation/` | Training-free evaluation metrics |
| `torch_mapping/` | Convert native PyTorch models to Mapper DAG / Supermodel |
