# common/ -- Shared Utilities

Leaf-dependency module providing file I/O helpers, C++ compiler discovery, and
experiment-tracking utilities. No mimarsinan-internal imports.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `file_utils.py` | `prepare_containing_directory`, `input_to_file`, `save_inputs_to_files`, `save_weights_and_chip_code` | File I/O for pipeline cache, simulator input files, and chip code generation |
| `build_utils.py` | `find_cpp20_compiler` | Discovers a C++20-capable compiler (Clang >= 17 preferred, g++-11 fallback) |
| `reporter.py` | `Reporter` (Protocol), `DefaultReporter` | Reporter protocol used by Pipeline; DefaultReporter provides throttled console output |
| `layer_key.py` | `layer_key_from_node_name` | Display/serialization: grouping key from IR or mapper node names (conv/fc/psum collapse); used by gui.snapshot and visualization.mapping_graphviz |
| `safe_numeric.py` | `safe_float` | Display/serialization: safe float conversion for plots/labels; used by visualization.search_visualization and visualization.mapping_graphviz |

## Dependencies

- **Internal**: None (leaf module).
- **External**: `torch`, `json`, `os`, `subprocess`.

## Dependents

- `chip_simulation` uses `file_utils` and `build_utils` for nevresim compilation
- `code_generation` uses `file_utils` for writing generated C++ code
- `pipelining` uses `prepare_containing_directory` for cache persistence
- `gui` uses `Reporter` protocol for dispatching metrics; `gui.snapshot` uses `layer_key_from_node_name`
- `visualization.mapping_graphviz` uses `layer_key_from_node_name` and `safe_float`
- `visualization.search_visualization` uses `safe_float`
- Entry point (`main.py`) uses `DefaultReporter`

## Exported API (\_\_init\_\_.py)

All public symbols from each file are re-exported at the package level.
