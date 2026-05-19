# chip_simulation/ -- Nevresim C++ Simulator Interface

Bridges Python and the nevresim C++ spiking neural network simulator.
Handles code generation, compilation, execution, and result parsing.
Also hosts optional Lava Loihi parity and SANA-FE detailed-stats backends.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `nevresim_driver.py` | `NevresimDriver` | Python-C++ bridge: generates code, compiles, runs, parses output. Accepts separate **`weight_type`** and **`threshold_type`** template parameters (TTFS uses float/double thresholds with int weights when quantised). |
| `simulation_runner.py` | `SimulationRunner` | End-to-end simulation for single-segment and multi-segment (hybrid) mappings. TTFS: rescales ComputeOp inputs via `node_activation_scales`. Sets `threshold_type = float` for TTFS, else matches `weight_type`. Parallel emit+compile capped at `cpu_count // 2`. |
| `hybrid_execution.py` | `assemble_segment_input_*`, `execute_compute_op_*`, `store_segment_output_*` | Shared host-side ComputeOp + segment I/O for HCM, SANA-FE, and tests. **Segment assembly** for SANA-FE uses **float64** so rate sums match HCM. **ComputeOp execution** uses float32 module math (same as nevresim `SimulationRunner._execute_compute_op_np`), then results are cast back to the caller's compute dtype. |
| `spike_recorder.py` | `RunRecord`, `SegmentSpikeRecord`, `CoreSpikeCounts`, `compare_records` | HCM/Loihi spike-count recording and diff utilities (segment inputs, per-core in/out, segment outputs). |
| `lava_loihi_runner.py` | `LavaLoihiRunner` | Optional Lava Loihi LIF runner. **`run_segments_from_reference()`** — production HCM-vs-Lava parity (one sample, segment replay). **`run()`** — exploratory accuracy (capped samples). |
| `subtractive_lif.py` | `SubtractiveLIFReset` | Lava LIF process: subtractive reset, no decay, active-window gating, buffer latch — matches HCM hard-core LIF. |
| `compile_nevresim.py` | `compile_simulator` | Compiles generated C++ with C++20 |
| `execute_nevresim.py` | `execute_simulator` | Runs binary; defaults to `cpu_count // 2` when `num_proc=0` |
| `_spike_encoding.py` | `uniform_rate_encode` | Shared uniform-rate encoder (HCM, Lava, SANA-FE) |
| `sanafe/` (sub-package) | `SanafeRunner`, `SanafeRunRecord`, … | Optional SANA-FE detailed-stats + parity. See `sanafe/ARCHITECTURE.md`. |

### Cross-backend numeric alignment

| Path | Dtype note |
|------|------------|
| HCM (`SpikingHybridCoreFlow`) | `torch.float64` default for membrane/TTFS math; weights float32 Parameters |
| SANA-FE (`SanafeRunner`) | `numpy.float64` segment assembly; float32 boundary drift can change ±1 spike at rate-encoding edges |
| nevresim TTFS | C++ `signal_t = double`; int rate-coded path uses exact integer arithmetic |
| Loihi parity | HCM `forward_with_recording` reference; per-core counts only at `B=1` |

All three deployment simulators depend on correct **`ChipLatency`** scheduling (see `mapping/ARCHITECTURE.md` § ChipLatency pitfalls) before parity gates are meaningful.

## Dependencies

- **Internal**: `code_generation`, `mapping`, `common`, `data_handling`.
- **External**: `subprocess`, `numpy`, `torch`; optional `lava`, `sanafe` (runtime only).

## Dependents

- `pipelining.pipeline_steps.simulation_step` — `SimulationRunner` / nevresim.
- `pipelining.pipeline_steps.loihi_simulation_step` — `LavaLoihiRunner` + `compare_records`.
- `pipelining.pipeline_steps.sanafe_simulation_step` — `SanafeRunner`.
- Entry point sets `NevresimDriver.nevresim_path`.

## Exported API (`__init__.py`)

`NevresimDriver`, `SimulationRunner`, `compile_simulator`, `execute_simulator`.
