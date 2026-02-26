# common/ -- Shared Utilities

Leaf-dependency module providing file I/O helpers, C++ compiler discovery, and
experiment-tracking utilities. No mimarsinan-internal imports.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `file_utils.py` | `prepare_containing_directory`, `input_to_file`, `save_inputs_to_files`, `save_weights_and_chip_code` | File I/O for pipeline cache, simulator input files, and chip code generation |
| `build_utils.py` | `find_cpp20_compiler` | Discovers a C++20-capable compiler (Clang >= 17 preferred, g++-11 fallback) |
| `wandb_utils.py` | `Reporter` (Protocol), `WandB_Reporter` | Weights & Biases integration; `Reporter` is the protocol used by `Pipeline` |

## Dependencies

- **Internal**: None (leaf module).
- **External**: `wandb`, `torch`, `json`, `os`, `subprocess`.

## Dependents

- `chip_simulation` uses `file_utils` and `build_utils` for nevresim compilation
- `code_generation` uses `file_utils` for writing generated C++ code
- `pipelining` uses `prepare_containing_directory` for cache persistence
- `gui` uses `Reporter` protocol for dispatching metrics
- Entry point (`main.py`) uses `WandB_Reporter`

## Exported API (\_\_init\_\_.py)

All public symbols from each file are re-exported at the package level.
