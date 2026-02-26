# chip_simulation/ -- Nevresim C++ Simulator Interface

Bridges Python and the nevresim C++ spiking neural network simulator.
Handles code generation, compilation, execution, and result parsing.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `nevresim_driver.py` | `NevresimDriver` | Python-C++ bridge: generates code, compiles, runs, and parses simulator output |
| `simulation_runner.py` | `SimulationRunner` | Orchestrates end-to-end simulation for single-segment and multi-segment (hybrid) mappings |
| `compile_nevresim.py` | `compile_simulator` | Compiles generated C++ code with C++20 compiler |
| `execute_nevresim.py` | `execute_simulator` | Runs compiled simulator binary, collects output |

## Dependencies

- **Internal**: `code_generation` (generate_main, cpp_chip_model), `mapping` (IR types, chip_latency, softcore/hybrid mappings), `common` (file_utils, build_utils), `data_handling` (`DataLoaderFactory`).
- **External**: `subprocess`, `numpy`, `torch`.

## Dependents

- `pipelining.pipeline_steps.simulation_step` uses `SimulationRunner` for final verification.
- Entry point (`init.py`) sets `NevresimDriver.nevresim_path`.

## Exported API (\_\_init\_\_.py)

`NevresimDriver`, `SimulationRunner`, `compile_simulator`, `execute_simulator`.
