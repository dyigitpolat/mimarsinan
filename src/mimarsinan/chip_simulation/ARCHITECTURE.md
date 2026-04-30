# chip_simulation/ -- Nevresim C++ Simulator Interface

Bridges Python and the nevresim C++ spiking neural network simulator.
Handles code generation, compilation, execution, and result parsing.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `nevresim_driver.py` | `NevresimDriver` | Python-C++ bridge: generates code, compiles, runs, and parses simulator output |
| `simulation_runner.py` | `SimulationRunner` | Orchestrates end-to-end simulation for single-segment and multi-segment (hybrid) mappings. In TTFS mode, `_execute_compute_op_np` rescales ComputeOp inputs from [0,1] back to training range via `node_activation_scales` (from `HybridHardCoreMapping`) so module bias terms remain correct. Parallel emit+compile and simulator execution are capped at `cpu_count // 2` processes to prevent server overload. |
| `lava_loihi_runner.py` | `LavaLoihiRunner` | Optional Lava-backed Loihi LIF runner. Production validation uses `run_segments_from_reference()` as a one-sample HCM-vs-Lava spike parity surface; `run()` remains an exploratory accuracy runner capped to one sample by default. |
| `spike_recorder.py` | `RunRecord`, `SegmentSpikeRecord`, `CoreSpikeCounts`, `compare_records` | HCM/Loihi spike-count recording and diff utilities for segment inputs, per-core inputs/outputs, and segment outputs. |
| `subtractive_lif.py` | `SubtractiveLIFReset` | Lava LIF process with subtractive reset, no decay, active-window gating, and buffer-latch behavior to match HCM hard-core simulation. |
| `compile_nevresim.py` | `compile_simulator` | Compiles generated C++ code with C++20 compiler |
| `execute_nevresim.py` | `execute_simulator` | Runs compiled simulator binary, collects output. Defaults to `cpu_count // 2` processes when `num_proc=0`. |

## Dependencies

- **Internal**: `code_generation` (generate_main, cpp_chip_model), `mapping` (IR types, chip_latency, softcore/hybrid mappings), `common` (file_utils, build_utils), `data_handling` (`DataLoaderFactory`).
- **External**: `subprocess`, `numpy`, `torch`.

## Dependents

- `pipelining.pipeline_steps.simulation_step` uses `SimulationRunner` for final verification.
- `pipelining.pipeline_steps.loihi_simulation_step` uses `LavaLoihiRunner` and spike record diffs for optional Loihi parity validation.
- Entry point (`init.py`) sets `NevresimDriver.nevresim_path`.

## Exported API (\_\_init\_\_.py)

`NevresimDriver`, `SimulationRunner`, `compile_simulator`, `execute_simulator`.
