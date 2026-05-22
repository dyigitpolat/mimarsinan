# chip_simulation/ -- Nevresim C++ Simulator Interface

Bridges Python and the nevresim C++ spiking neural network simulator.
Handles code generation, compilation, execution, and result parsing.
Also hosts optional Lava Loihi parity and SANA-FE detailed-stats backends.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `nevresim_driver.py` | `NevresimDriver` | Python-C++ bridge: generates code, compiles, runs, parses output. Accepts separate **`weight_type`** and **`threshold_type`** template parameters (TTFS uses float/double thresholds with int weights when quantised). |
| `simulation_runner.py` | `SimulationRunner` | End-to-end simulation for single-segment and multi-segment (hybrid) mappings. TTFS: rescales ComputeOp inputs via `node_activation_scales`. Sets `threshold_type = float` for TTFS, else matches `weight_type`. Parallel emit+compile capped at `cpu_count // 2`. |
| `hybrid_execution.py` | `assemble_segment_input_*`, `execute_compute_op_*`, `store_segment_output_*`, `resolve_stage_compute_scales` | Shared segment I/O and compute ops. **`resolve_stage_compute_scales(mapping, op_id, apply_ttfs=…)`**: nevresim passes `apply_ttfs` only in TTFS modes; SANA-FE and HCM hybrid flow use always-on scales (`apply_ttfs=True`). SANA-FE segment assembly uses **float64**. |
| `hybrid_semantics.py` | `NeuralSegmentResult`, `store_neural_segment_output`, `lif_inter_stage_from_spike_counts` | **Inter-stage contract**: LIF/rate → spike count / T; TTFS → activation in [0, 1]. All hybrid backends must use `store_neural_segment_output` after neural stages. |
| `ttfs_executor.py` | `TtfsAnalyticalExecutor`, `run_ttfs_hybrid_contract` | Canonical TTFS segment semantics and shared hybrid contract runner (numpy float64 compute ops). Used by `record_ttfs_hcm_reference` and SANA-FE `contract_ttfs_*` fields. |
| `ttfs_segment.py` | `segment_ttfs_arrays_from_mapping`, `run_ttfs_*`, `gather_segment_ttfs_output_from_cores` | Numpy TTFS execution and output gather. |
| `ttfs_recorder.py` | `TtfsRunRecord`, `compare_ttfs_contract_records`, `compare_ttfs_hardware_records` | TTFS activation parity records; contract vs hardware tolerance policies. |
| `neural_segment_executor.py` | `execute_neural_segment_analytical` | TTFS analytical dispatch helper for shared hybrid loop. |
| `hybrid_stage_runner.py` | `run_hybrid_stages`, `HybridStageContext` | Shared stage loop with `on_neural` / `on_compute` (and optional `after_neural` / `after_compute`). Used by **nevresim** (`SimulationRunner._run_hybrid`), **SANA-FE**, **Lava**, and **`SpikingHybridCoreFlow`** (HCM). |
| `firing_strategy.py` | `resolve_firing_mode`, `apply_firing_to_lif` | Maps deployment `firing_mode` / reset semantics to training (`LIFActivation`), Lava (`SubtractiveLIFReset`), and SANA-FE attrs. See `mapping/FIRING.md`. |
| `spike_modes.py` | `to_spikes`, stochastic/deterministic helpers | Single torch spike-encoding implementation for unified and hybrid core flows. |
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
| TTFS reference | `TtfsAnalyticalExecutor` + `ttfs_segment.py` — single numpy semantics for HCM, Nevresim (raw activations), and SANA-FE buffer propagation |
| SANA-FE TTFS | Plugins + `TtfsAnalyticalExecutor` contract; hardware parity via `potential_trace`; no `sana_fe` core forks |
| Loihi/Lava | LIF only — TTFS does not map onto Loihi LIF dynamics; pipeline rejects `enable_loihi_simulation` + TTFS |
| SANA-FE (`SanafeRunner`) | `numpy.float64` segment assembly; float32 boundary drift can change ±1 spike at rate-encoding edges |
| nevresim TTFS | C++ `signal_t = double`; int rate-coded path uses exact integer arithmetic |
| Loihi parity | HCM `forward_with_recording` reference; per-core counts only at `B=1` |

All three deployment simulators depend on correct **`ChipLatency`** scheduling (see `mapping/ARCHITECTURE.md` § ChipLatency pitfalls) before parity gates are meaningful.

## Dependencies

- **Internal**: `code_generation`, `mapping`, `common`, `data_handling`.
- **External**: `subprocess`, `numpy`, `torch`; optional `lava`, `sanafe` (runtime only).

## Dependents

- `pipelining.pipeline_steps.simulation_step` — `SimulationRunner` / nevresim.
- `pipelining.pipeline_steps.loihi_simulation_step` — `record_hcm_reference` + `LavaLoihiRunner` + `assert_spike_parity_or_raise`.
- `pipelining.pipeline_steps.sanafe_simulation_step` — `SanafeRunner` + optional HCM parity via `simulation_factory`.
- `pipelining.simulation_factory` — shared HCM build and metric test for SCM/HCM.
- Entry point sets `NevresimDriver.nevresim_path`.

## Exported API (`__init__.py`)

`NevresimDriver`, `SimulationRunner`, `compile_simulator`, `execute_simulator`.
