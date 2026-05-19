# code_generation/ -- C++ Code Generation

Generates C++ source files for the nevresim chip simulator from hardware
mapping artifacts.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `cpp_chip_model.py` | `SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel` | Chip data model → C++ struct initializers. Template params: **`weight_type`** (`int`/`float`) and **`threshold_type`** (independent; TTFS uses `double` thresholds with `int` weights when quantised). |
| `generate_main.py` | `generate_main_function`, `get_config` | Instantiates `main.cpp` with simulation length, spike/firing modes, **`weight_type`**, **`threshold_type`**. |
| `main_cpp_template.py` | `main_cpp_template` | Standard spiking execution |
| `main_cpp_template_real_valued_exec.py` | … | Real-valued (non-spiking) execution |
| `main_cpp_template_debug_spikes.py` | … | Spike debug output |

`NevresimDriver` passes both types into codegen so nevresim C++ matches Python simulator numeric semantics (decoupled since nevresim threshold-type refactor).

## Dependencies

- **Internal**: `common.file_utils`.
- **External**: `numpy`.

## Dependents

- `mapping` (`SpikeSource`), `chip_simulation` (`generate_main_function`), `visualization`.

## Exported API (`__init__.py`)

`SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel`,
`generate_main_function`, `generate_main_function_for_real_valued_exec`, `get_config`.
