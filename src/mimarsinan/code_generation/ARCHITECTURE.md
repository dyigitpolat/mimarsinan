# code_generation/ — Generates nevresim C++ sources from hardware mapping artifacts

This module turns a mapped chip (cores, neurons, axon connections) into the C++
files the nevresim simulator compiles and runs: a `generate_chip.hpp` /
`generate_chip_config.hpp` header describing the chip, a `main.cpp` instantiated
from a template, plus weight and span data files. Its central abstractions are
`ChipModel` (the chip data model with `weight_type` / `threshold_type` emitters)
and `resolve_exec_policy`, which delegates the spiking-mode → C++
(ComputePolicy, Execution) type choice to `chip_simulation`'s
`SpikingModePolicy` SSOT rather than hardcoding it here.

## Key files
| File | Purpose |
|---|---|
| `cpp_chip_model.py` | `ChipModel`: chip dimensions + cores/connections/outputs; emits the consteval `generate_chip` header, the weights text file, and chip JSON (save/load). |
| `cpp_chip_model_types.py` | Codegen value types — `SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core` — and `compress_sources_to_spans` (run-length encodes axon sources into spans). |
| `generate_main.py` | Instantiates `main.cpp` from a template: `get_config`, `resolve_compare_policy` / `resolve_lif_fire_policy` (comparator/reset strings), `resolve_exec_policy` (delegates to `SpikingModePolicy.nevresim_exec_policy`), `generate_main_function`, `generate_main_function_runtime`, `generate_main_function_for_real_valued_exec`. |
| `main_cpp_template.py` | Standard compile-time-chip spiking main; loads inputs via `load_input_n` or `load_spike_train_input_n`. |
| `main_cpp_template_runtime.py` | Runtime-chip main: loads connectivity from `chip_spans.txt` at run time instead of baking it into a consteval chip (avoids recompiling per mapping). |
| `main_cpp_template_debug_spikes.py` | Debug main that prints per-cycle firing neuron indices instead of output counts. |
| `main_cpp_template_real_valued_exec.py` | Non-spiking (real-valued compute/execution) main for ANN-equivalent reference runs. |
| `mapping_spans_export.py` | Writes `chip_spans.txt` (consumed by nevresim `mapping_loader.hpp`) and emits the dimensions-only `RuntimeChipConfig` header for runtime chips. |

## Dependencies
- `chip_simulation` — `spiking_mode_policy` (`ExecPolicySpec`, `NevresimExecParams`, `policy_for_spiking_mode`): the SSOT deciding which C++ compute/execution types a spiking mode compiles to.
- `common` — `file_utils` (`prepare_containing_directory`) for writing generated files.
- `mapping` — `support.spike_source_spans.compress_spike_sources` (lazy import) to run-length encode `SpikeSource` lists into spans.

## Dependents
- `chip_simulation` — `nevresim_driver`, `compile_cache`, and `nevresim/profiling` use `generate_main_function*`, `get_config`, `ChipModel`, `SpikeSource`.
- `common` — `file_utils` lazily imports `mapping_spans_export` to write runtime-chip span/config files.
- `mapping` — chip export, softcore packing/compaction, IR legacy conversion, and span utilities build on `SpikeSource` / `ChipModel`.
- `visualization` — graphviz softcore/hardcore renderers consume `SpikeSource`.

## Exported API
- `SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core` — codegen value types for chip connectivity and parameters.
- `ChipModel` — the chip data model and C++/JSON emitter.
- `generate_main_function` — write a spiking `main.cpp` from a template + config.
- `generate_main_function_for_real_valued_exec` — write the real-valued reference `main.cpp`.
- `get_config` — build the simulation-config dict (spike gen, firing, thresholding, weight/threshold types, spiking mode).
