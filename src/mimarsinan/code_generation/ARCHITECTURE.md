# code_generation/ -- C++ Code Generation

Generates C++ source files for the nevresim chip simulator from hardware
mapping artifacts.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `cpp_chip_model.py` | `SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel` | Chip data model → C++ struct initializers. Template params: **`weight_type`** (`int`/`float`) and **`threshold_type`** (independent; TTFS uses `double` thresholds with `int` weights when quantised). |
| `generate_main.py` | `generate_main_function`, `get_config`, `resolve_exec_policy`, `resolve_compare_policy`, `resolve_lif_fire_policy` | Instantiates `main.cpp` with simulation length, spike/firing modes, **`weight_type`**, **`threshold_type`**. `resolve_compare_policy(thresholding_mode)` → `Strict`/`InclusiveCompare` and `resolve_lif_fire_policy(firing,thresholding)` → `LIFirePolicy<reset,compare>` are the comparator/reset string selection (config-driven, never hardcoded). `resolve_exec_policy` **delegates the firing × sync → codegen choice to `SpikingModePolicy.nevresim_exec_policy(NevresimExecParams)`** (V2, `chip_simulation/spiking_mode_policy.py`): it selects the comparator/reset strings, packs the chip-shape scalars into `NevresimExecParams`, and the resolved policy returns the `(compute, exec)` `ExecPolicySpec`. A new spiking mode updates the **policy**, not this dispatch. (`ExecPolicySpec` lives in `spiking_mode_policy.py`, re-exported here.) Per-family C++ types: rate/LIF → `SpikingCompute<LIFirePolicy<reset,compare>>`; `ttfs` (analytical) → `TTFSContinuousExecution`; `ttfs_quantized` → `TTFSQuantizedCompute<S,compare>` + `TTFSExecution<S,lat,compare>`; **`ttfs_cycle_based` (cascaded) → `TTFSCascadeCompute<compare>` + `TTFSCascadeExecution<…,TTFSSpikeGenerator,…,compare>`** (each neuron fires once; the cascade reconstructs the ramp via `ramp_current`; `leak=0`). The synchronized schedule disables nevresim (its policy `nevresim_exec_policy` raises), so any nevresim run of `ttfs_cycle_based` is cascaded. |
| `main_cpp_template.py` | `main_cpp_template` | Standard spiking execution. Inputs are loaded via `load_input_n` (rate-coded path). |
| `main_cpp_template_real_valued_exec.py` | … | Real-valued (non-spiking) execution |
| `main_cpp_template_debug_spikes.py` | … | Spike debug output |

`NevresimDriver` passes both types into codegen so nevresim C++ matches Python simulator numeric semantics (decoupled since nevresim threshold-type refactor).

## Dependencies

- **Internal**: `common.file_utils`; `chip_simulation.spiking_mode_policy` (the V2
  `SpikingModePolicy.nevresim_exec_policy` SSOT for the firing × sync → codegen choice).
- **External**: `numpy`.

## Dependents

- `mapping` (`SpikeSource`), `chip_simulation` (`generate_main_function`), `visualization`.

## Exported API (`__init__.py`)

`SpikeSource`, `CodegenSpan`, `Connection`, `Neuron`, `Core`, `ChipModel`,
`generate_main_function`, `generate_main_function_for_real_valued_exec`, `get_config`.
