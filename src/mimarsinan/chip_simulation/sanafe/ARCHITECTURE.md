# chip_simulation/sanafe/ -- SANA-FE Integration

[SANA-FE](https://github.com/SLAM-Lab/SANA-FE) (GPL-3.0) is a calibrated
neuromorphic simulator that decomposes a run into per-tile / per-core
energy (synapse + dendrite + soma + network), latency (`sim_time`),
spike traffic, NoC packet/hop traces, and optional per-neuron spike +
potential traces.  This sub-package is the bridge between a
`HybridHardCoreMapping` and SANA-FE's `SpikingChip.sim()`.

The integration is **opt-in**: every `import sanafe` is gated by the
lazy `arch_synth._sanafe()` accessor, so the mimarsinan package itself
imports cleanly without a SANA-FE install.  Run
`scripts/bootstrap_sanafe.sh` to clone/build SANA-FE and compile mimarsinan
plugins into `build/mimarsinan_sanafe_plugins/` (`libmimarsinan_soma.so`,
`libmimarsinan_dendrite.so`).

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `records.py` | `SanafeRunRecord`, `SanafeSegmentRecord`, `SanafeCoreRecord`, `SanafeTileRecord`, `SanafeEnergyBreakdown`, `SanafeNocLink`, `SanafeNocLinkLoad`, `SanafeCascadePoint`, … | Rich per-sample record dataclasses.  `SanafeRunRecord.to_hcm_subset()` projects spike-count fields to `spike_recorder.RunRecord` so `compare_records()` is reused for the parity gate. |
| `stats.py` | `SanafeStepReport` | Step-level aggregate over `sanafe_sample_count` runs.  `to_snapshot_dict()` feeds `gui/snapshot/builders.py::snapshot_sanafe_simulation`. |
| `presets.py` | `LOIHI_PRESET`, `TRUENORTH_PRESET`, `PRESETS` | Per-event energy / latency constants for synthesized architectures (`sanafe_arch_preset` in deployment config). |
| `arch_synth.py` | `ArchSpec`, `derive_arch_spec`, `build_architecture`, `_sanafe`, `_plugin_path` | Geometry from HCM → SANA-FE `Architecture`; optional custom YAML via `sanafe_custom_arch_path`.  **`_plugin_path(name)`** resolves `build/mimarsinan_sanafe_plugins/libmimarsinan_<name>.so` (required for production YAML). |
| `net_synth.py` | `build_network_for_segment`, `set_input_spike_trains`, `set_always_on_spike_trains` | Per-segment `Network`: one neuron group per `HardCore`, `input` / `always_on` groups; synapses from `axon_sources` (zeros elided, duplicate sources summed — Lava convention). |
| `neuron_model.py` | `lif_model_attributes`, `ttfs_*_model_attributes`, `ttfs_cascade_model_attributes`, `input_neuron_attributes`, `soma_hw_name_for_spiking_mode` | Builds `model_attributes` for each soma + input somas: subtractive LIF semantics, optional `active_start` / `active_length` per core. **`soma_hw_name_for_spiking_mode(mode, schedule)`** is a thin wrapper delegating to the `SpikingModePolicy` (V2, `chip_simulation/spiking_mode_policy.py`): `ttfs_cycle_based` → `ttfs_cycle` soma (synchronized) or `ttfs_cascade` soma (cascaded). `net_synth/build.py` resolves the policy once per segment and routes both `soma_hw_name()` and `soma_model_attributes(...)` through it (no per-mode `if`-chain). |
| `runner.py` | `SanafeRunner` | Only module calling `sanafe.SpikingChip`.  Neural stages on SANA-FE; compute stages via `hybrid_execution`.  **`_COMPUTE_DTYPE = float64`** for segment input assembly (must match HCM; float32 can flip ±1 spike at rate-encoding boundaries).  Simulation length **`T + max_latency + 1`** (+1 cycle: SANA-FE applies input spikes one cycle after emission).  Per-core **`active_start` / `active_length`** from `ChipLatency` (via `lif_model_attributes`). |
| `plugins/CMakeLists.txt` | — | Builds `libmimarsinan_soma.so`, `libmimarsinan_dendrite.so`. |
| `plugins/mimarsinan_soma.cpp` | — | Custom soma: configurable reset (`reset_mode` soft/hard), strict/inclusive thresholding, active-window gating. |
| `plugins/mimarsinan_ttfs_continuous_soma.cpp` | — | Analytical TTFS: `relu(I+b)/θ` in one active step. |
| `plugins/mimarsinan_ttfs_quantized_soma.cpp` | — | TTFS quantized: analytical V injected via `preset_membrane`, fires at the precomputed step. Cores effectively exchange real values. |
| `plugins/mimarsinan_ttfs_cycle_soma.cpp` | — | **Genuine single-spike TTFS** (`ttfs_cycle_based`, **synchronized** schedule): no preset; reconstructs `V` from incoming single-spike *timings* (decode-on-arrival, weight `(S−(cyc−1)%S)/S`), fires once at `k_fire=⌈S(1−V/θ)⌉`. Runs the `S×num_groups` schedule — each latency group owns window `[(g+1)·S, (g+2)·S)`, input window `[0,S)` — set by `net_synth/build.py`; single-shot input encoding (`ttfs_single_spike_train` via the binary-mask `set_input_spike_trains`); no `apply_ttfs_preset_membranes`. Off-grid inputs are 1/S-grid-quantized on the wire; the analytical reference applies the same snap to the stage input (`ttfs_input_grid_quantize`, see `ttfs_executor.py`). Debug: `MIMARSINAN_TTFS_CYCLE_TRACE=<path>` dumps per-event CSV (input decode, dropped-late deliveries, hexfloat fire-step resolution). Parity vs analytical: `tests/integration/test_sanafe_ttfs_cycle_parity.py`, `test_sanafe_ttfs_cycle_offgrid_parity.py`. |
| `plugins/mimarsinan_ttfs_cascade_soma.cpp` | — | **Genuine cascaded (greedy) TTFS** (`ttfs_cycle_based`, **cascaded** schedule): hardware-faithful **single-spike** — each neuron fires exactly **once** (one spike on the wire); the integration is a **ramp** via a persistent `ramp_current` (`ramp_current += current_in; membrane += ramp_current + bias`), so one input spike at `t_j` contributes its weight every later cycle. No reset, no preset; config-driven compare (`thresholding_mode`). **Single-spike** TTFS input (`ttfs_single_spike_train`), **latency-gated** somas (`active_start=lat+1, active_length=T` — active only in their window, like LIF), `T_eff = ChipLatency + T + 1`. The decoded segment **value** is per-source windowed: `(core.latency + 1 + T) − first_fire` clamped to `[0,T]` (`_single_spike_ramp_outputs`); the per-core spike **count** is single-spike *traffic* (≤1). Per-source windowing is essential — a global `T_eff − fire` overcounts shallow sources and saturates when latency >> T. Mirrors nevresim `TTFSCascadeCompute`/`TTFSCascadeExecution` + HCM `TTFSGreedyCyclePolicy`. Parity: `tests/integration/test_cascaded_ttfs_backend_parity.py`. |
| `plugins/mimarsinan_dendrite.cpp` | — | Custom dendrite plugin. |

## Neuron-model / architecture synthesis

Production YAML from `build_architecture()` wires **plugin somas and dendrites**
(`model: mimarsinan_soma`, `model: mimarsinan_dendrite`), not SANA-FE's stock
`leaky_integrate_fire`.  `lif_model_attributes()` supplies the attribute dict
(`leak_decay`, `reset_mode`, `force_update`, per-core `active_start` /
`active_length`, etc.) consumed by the plugin.

`bootstrap_sanafe.sh` must build the plugins; `derive_arch_spec` / `build_architecture`
fail fast with a clear message if the `.so` files are missing.

## Parity gate (HCM ↔ SANA-FE)

`SanafeSimulationStep` cross-checks against HCM for each sample:

1. **TTFS:** `record_ttfs_hcm_reference()` (canonical `TtfsAnalyticalExecutor` + shared preprocessor via `preprocess_hybrid_sample`). **LIF:** `forward_with_recording()` → `RunRecord`.
2. `SanafeRunner` with the same preprocessed sample; TTFS stages call `store_neural_segment_output` (activations, not spike/T).
3. **LIF:** `to_hcm_subset()` vs HCM via `compare_records()`.
4. **TTFS (two layers, both must pass):**
   - **Contract:** `to_ttfs_contract_subset()` — `TtfsAnalyticalExecutor` on each segment's `seg_input_rates` (aligned with Nevresim/HCM); inter-stage buffer uses this result.
   - **Hardware:** `to_ttfs_hardware_subset()` — plugin soma activations via existing `potential_trace` (no SANA-FE core / `pymodule` changes).

**Do not patch** [`sana_fe/src/pymodule.cpp`](../../../sana_fe/src/pymodule.cpp). TTFS surface area is mimarsinan plugins + Python only.

`ChipLatency` post-passes on the mapping must be correct before either simulator runs.

### Inter-stage TTFS contract

Between hybrid stages, the state buffer carries **TTFS activations** from `TtfsAnalyticalExecutor` (same as Nevresim raw TTFS and HCM `seg_out`), not `output_spike_count / T`. Multi-stage parity fails if only per-core hardware trace matches but buffer propagation used LIF rates.

## SANA-FE trace wire formats

- **Tile mesh coords** — column-major: `x = tile_id // mesh_height`, `y = tile_id % mesh_height` (matches SANA-FE `Architecture::calculate_tile_coordinates`).
- **`spike_trace`** — per cycle, list of `NeuronAddress` objects or `"<group_name>.<neuron_index>"` strings. LIF groups are `coreN`; input-path groups are `coreN_in` / `coreN_on` (with `log_spikes=True` in `net_synth`).
- **`message_trace`** — per cycle, list of dicts with `src_tile_id`, `dest_tile_id`, tile-local `src_core_id` / `dest_core_id`, `src_x`/`src_y`/`dest_x`/`dest_y`, `hops`, `spikes`, `src_neuron_group_id`; entries with `placeholder: true` are ignored. GUI taxonomy: `inter_tile_packets`, `intra_tile_packets`, `input_path_packets`; playback fallback uses `tile_packets_per_cycle` when inter-tile mesh crossings are empty.
- **Global core id** — `tile_id * cores_per_tile + core_id_within_tile` when mapping trace events back to HCM cores.
- **Per-core energy** — mirrors SANA-FE `sim_calculate_core_energy`: synapse × incoming spikes, dendrite × `n_neurons × T_eff`, soma access/update/spike-out counts; hop energy rolls up at the destination tile.

## Dependencies

- **Internal**: `chip_simulation.spike_recorder`, `chip_simulation.hybrid_execution`,
  `chip_simulation._spike_encoding`, `mapping.chip_latency`, `mapping` (HCM types),
  `code_generation.SpikeSource`.
- **External**: `numpy`, `torch`; lazily `sanafe` (GPL-3.0).

## Dependents

- `pipelining.pipeline_steps.sanafe_simulation_step` — loops `sanafe_sample_count` samples.
- `gui.snapshot.builders.snapshot_sanafe_simulation` — summary + deferred heatmap/NoC resources.
- `gui/static/js/sanafe-tab.js` — monitor tab: energy cards, heatmaps, **NoC link playback**.

## Exported API (`__init__.py`)

`SanafeRunRecord`, `SanafeSegmentRecord`, `SanafeCoreRecord`,
`SanafeTileRecord`, `SanafeEnergyBreakdown`.
