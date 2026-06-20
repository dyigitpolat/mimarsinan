# chip_simulation/ -- Nevresim C++ Simulator Interface

Bridges Python and the nevresim C++ spiking neural network simulator.
Handles code generation, compilation, execution, and result parsing.
Also hosts optional Lava Loihi parity and SANA-FE detailed-stats backends.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `nevresim/nevresim_driver.py` | `NevresimDriver` | Python-C++ bridge. Default **`connectivity_mode=runtime`** (`nevresim/connectivity.py`); override via config `nevresim_connectivity_mode`. Optional **`compile_cache_dir`**. |
| `nevresim/connectivity.py` | `DEFAULT_NEVRESIM_CONNECTIVITY_MODE`, `resolve_nevresim_connectivity_mode` | Deployment default and pipeline config resolution for chip wiring mode. |
| `nevresim/compile_nevresim.py` | `compile_simulator`, `CompileResult` | Compiles generated C++ with C++20; optional `-ftime-trace`, `-O0` |
| `nevresim/execute_nevresim.py` | `execute_simulator` | Runs binary; defaults to `cpu_count // 2` when `num_proc=0`. Drains stdout via `communicate()`, validates optional `expected_values`, surfaces stderr on failure. Protocol: one line per sample (`output_size` floats + newline). |
| `nevresim/segment_execute.py` | `run_binary_raw`, `save_segment_inputs` | SSOT for segment binary execution (input serialization, run, reshape). Used by `NevresimDriver` and hybrid precompiled path. |
| `nevresim/compile_cache.py` | `NevresimCompileCache` | Segment binary cache keyed by mapping + policy hash |
| `nevresim/profiling/` | `profile_mapping_compile`, `build_synthetic_mapping`, … | Synthetic compile sweeps and timing records (`scripts/profile_nevresim_compile.py`) |
| `code_generation/main_cpp_template_runtime.py` | Runtime main template | ``RuntimeChip`` uses **static** storage (not stack) to support large core counts |
| `simulation_runner/core.py` | `SimulationRunner` | End-to-end simulation for single-segment and multi-segment (hybrid) mappings. Reads `nevresim_connectivity_mode` from pipeline config. Parallel emit+compile capped at `cpu_count // 2`. |
| `hybrid_execution.py` | `assemble_segment_input_*`, `apply_input_shifts_numpy`, `execute_compute_op_*`, … | Shared segment I/O and compute ops for hybrid backends. `apply_input_shifts_numpy` is the numpy mirror of HCM's `_apply_input_shifts` (negative-value shift before the encode clamp); ComputeOp execution runs in the op module's own floating dtype. |
| `hybrid_semantics.py` | `NeuralSegmentResult`, `store_neural_segment_output`, `lif_inter_stage_from_spike_counts` | **Inter-stage contract**: LIF/rate → spike count / T; TTFS → activation in [0, 1]. All hybrid backends must use `store_neural_segment_output` after neural stages. `lif_inter_stage_from_spike_counts` is re-exported from the SSOT `mimarsinan.spiking.segment_boundary.decode_segment_output` (single source across sims). |
| `ttfs_executor.py` | `TtfsAnalyticalExecutor`, `run_ttfs_hybrid_contract` | Canonical TTFS segment semantics and shared hybrid contract runner (numpy float64 compute ops). Applies `node_output_shifts` (negative-value shift) to each assembled neural-stage input; under the **synchronized** `ttfs_cycle_based` schedule it then snaps the stage input to the single-spike timing grid (`ttfs_input_grid_quantize`) — the spike-time-encoded hardware boundary cannot carry off-grid values. Used by `record_ttfs_hcm_reference` and SANA-FE `contract_ttfs_*` fields. |
| `ttfs_encoding.py` | `ttfs_spike_time`, `ttfs_single_spike_train`, `ttfs_latched_spike_train`, `ttfs_input_grid_quantize` | TTFS input spike encoding (`round(S·(1−clamp(x)))`), spike-train builders, and the encode→decode round-trip `ttfs_input_grid_quantize`. The scalar kernels delegate to the `models/spiking/wire_semantics.py` numpy twins (one source for torch and numpy). |
| `ttfs_segment.py` | `segment_ttfs_arrays_from_mapping`, `run_ttfs_*`, `gather_segment_ttfs_output_from_cores` | Numpy TTFS execution and output gather. |
| `ttfs_recorder.py` | `TtfsRunRecord`, `compare_ttfs_contract_records`, `compare_ttfs_hardware_records` | TTFS activation parity records; contract vs hardware tolerance policies. |
| `neural_segment_executor.py` | `execute_neural_segment_analytical` | TTFS analytical dispatch helper for shared hybrid loop. |
| `hybrid_stage_runner.py` | `run_hybrid_stages`, `HybridStageContext` | Shared stage loop with `on_neural` / `on_compute` (and optional `after_neural` / `after_compute`). Used by **nevresim** (`SimulationRunner._run_hybrid`), **SANA-FE**, **Lava**, and **`SpikingHybridCoreFlow`** (HCM). |
| `firing_strategy.py` | `FiringStrategy`, `FiringStrategyFactory` | Reset + threshold validation; backend capability gates. See `mapping/FIRING.md`. |
| `behavior_config.py` | `NeuralBehaviorConfig` | Identity axes of the simulator-facing activation semantics (reset, comparison, spike encode). Used by Lava, SANA-FE, and codegen helpers; composed into `SpikingDeploymentContract`. |
| `deployment_contract.py` | `SpikingDeploymentContract` | **Cross-side deployment-semantics SSOT** (torch NF + simulators): composes `NeuralBehaviorConfig` and adds `simulation_steps`, `ttfs_cycle_schedule`, `encoding_layer_placement`, `bias_mode`. **Single-reader invariant**: `from_pipeline_config` is the only place these config keys are read (grep-guard test); everything downstream takes the contract. Derived getters (`is_synchronized`, `quantize_stage_input_to_grid`, `training_forward_kind`) answer schedule-derived questions so consumers stop re-deriving implications; all accept a reserved `core=None` kwarg for future per-core heterogeneity. `mode_policy()` returns the (firing × sync) `SpikingModePolicy` (V2) that carries `training_forward_kind` and the soma/calibration behavior. |
| `spiking_mode_policy.py` | `SpikingModePolicy`, `LifModePolicy`, `TtfsAnalyticalModePolicy`, `TtfsSyncCycleModePolicy`, `TtfsCascadeModePolicy`, `policy_for_spiking_mode` | **Behavior-carrying policy per (firing × sync)** (Vector V2). One polymorphic class per `(spiking_mode, schedule)` family carries `training_forward_kind()` (the NF the fine-tuners train through), `calibration_forward()` (the negative-shift NF forward), `soma_hw_name()` / `soma_model_attributes(...)` (the SANA-FE soma chain), `log_potential`, `decode_mode()`, and `valid_backends()`. `policy_for_spiking_mode(mode, schedule)` (= `from_contract`) is the SSOT dispatch; callers take the resolved policy instead of re-branching on `cascaded`/`synchronized`. Mirrors `spiking/segment_policies.py`. |
| `spike_modes.py` | `to_spikes`, stochastic/deterministic helpers | Single torch spike-encoding implementation for unified and hybrid core flows. |
| `spike_recorder.py` | `RunRecord`, `SegmentSpikeRecord`, `CoreSpikeCounts`, `compare_records` | HCM/Loihi spike-count recording and diff utilities (segment inputs, per-core in/out, segment outputs). |
| `lava_loihi/runner.py` | `LavaLoihiRunner` | Optional Lava Loihi LIF runner. Accepts `NeuralBehaviorConfig`; **`run_segments_from_reference()`** — production HCM-vs-Lava parity. |
| `subtractive_lif.py` | `SubtractiveLIFReset` | Lava LIF process: configurable reset (`zero_reset`), thresholding, active-window gating. |
| `_spike_encoding.py` | `encode_segment_input`, `uniform_rate_encode`, … | Shared numpy batch encoder for Lava/SANA-FE segment injection; Uniform uses torch parity path. |
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
