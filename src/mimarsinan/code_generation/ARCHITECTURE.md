# code_generation/ -- C++ Code Generation

Generates C++ source files for the nevresim chip simulator from hardware
mapping artifacts.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `cpp_chip_model.py` | `SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel` | Data model representing the chip; generates C++ struct initializers |
| `generate_main.py` | `generate_main_function`, `get_config` | Template instantiation for `main.cpp` with simulation parameters |
| `main_cpp_template.py` | `main_cpp_template` | C++ template string for standard spiking execution |
| `main_cpp_template_real_valued_exec.py` | `main_cpp_template_real_valued_exec` | C++ template for real-valued (non-spiking) execution |
| `main_cpp_template_debug_spikes.py` | `main_cpp_template_debug_spikes` | C++ template with spike debug output |

## Dependencies

- **Internal**: `common.file_utils` (for `prepare_containing_directory`).
- **External**: `numpy`.

## Dependents

- `mapping` imports `SpikeSource` extensively (IR, softcore mapping, mapping utilities)
- `chip_simulation` uses `generate_main_function` for simulator compilation
- `visualization` uses `SpikeSource` for graphviz rendering

## Exported API (\_\_init\_\_.py)

`SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel`,
`generate_main_function`, `generate_main_function_for_real_valued_exec`, `get_config`.
