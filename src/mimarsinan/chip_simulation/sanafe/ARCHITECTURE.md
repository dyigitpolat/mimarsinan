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
| `neuron_model.py` | `lif_model_attributes`, `input_neuron_attributes` | Builds `model_attributes` for **`mimarsinan_soma`** / input somas: subtractive LIF semantics, optional `active_start` / `active_length` per core (aligned with HCM active windows). |
| `runner.py` | `SanafeRunner` | Only module calling `sanafe.SpikingChip`.  Neural stages on SANA-FE; compute stages via `hybrid_execution`.  **`_COMPUTE_DTYPE = float64`** for segment input assembly (must match HCM; float32 can flip ±1 spike at rate-encoding boundaries).  Simulation length **`T + max_latency + 1`** (+1 cycle: SANA-FE applies input spikes one cycle after emission).  Per-core **`active_start` / `active_length`** from `ChipLatency` (via `lif_model_attributes`). |
| `plugins/CMakeLists.txt` | — | Builds `libmimarsinan_soma.so`, `libmimarsinan_dendrite.so`. |
| `plugins/mimarsinan_soma.cpp` | — | Custom soma: subtractive LIF, strict/inclusive thresholding, active-window gating (replaces SANA-FE built-in `leaky_integrate_fire` Loihi caps). |
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

When `sanafe_parity_check` is true (default), `SanafeSimulationStep` for each sample:

1. Runs `SpikingHybridCoreFlow.forward_with_recording()` (HCM) → `RunRecord`.
2. Runs `SanafeRunner` → `SanafeRunRecord`.
3. Compares `sanafe_record.to_hcm_subset()` vs HCM via `compare_records()`; any mismatch fails the pipeline.

`ChipLatency` post-passes on the mapping must be correct before either simulator runs.

## SANA-FE trace wire formats

- **Tile mesh coords** — column-major: `x = tile_id // mesh_height`, `y = tile_id % mesh_height` (matches SANA-FE `Architecture::calculate_tile_coordinates`).
- **`spike_trace`** — per cycle, list of `"<group_name>.<neuron_index>"` strings.
- **`message_trace`** — per cycle, list of dicts with `src_tile_id`, `dest_tile_id`, tile-local `src_core_id` / `dest_core_id`, `src_x`/`src_y`/`dest_x`/`dest_y`, `hops`, `spikes`; entries with `placeholder: true` are ignored.
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
