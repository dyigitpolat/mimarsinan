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
`scripts/bootstrap_sanafe.sh` to enable it.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `records.py` | `SanafeRunRecord`, `SanafeSegmentRecord`, `SanafeCoreRecord`, `SanafeTileRecord`, `SanafeEnergyBreakdown` | Rich per-sample record dataclasses.  `SanafeRunRecord.to_hcm_subset()` is the single lossless projection to `spike_recorder.RunRecord` so `compare_records()` is reused verbatim for the parity gate. |
| `stats.py` | `SanafeStepReport` | Step-level aggregate of one or more `SanafeRunRecord`s.  `to_snapshot_dict()` produces the JSON-safe payload consumed by `gui/snapshot/builders.py::snapshot_sanafe_simulation`. |
| `presets.py` | `LOIHI_PRESET`, `TRUENORTH_PRESET`, `PRESETS` | Per-event energy / latency constants injected into the synthesized SANA-FE architecture. |
| `arch_synth.py` | `ArchSpec`, `derive_arch_spec`, `build_architecture`, `_sanafe` | Two-stage architecture synthesis: pure-Python geometry derivation from an HCM, then SANA-FE-touching `Architecture` construction (or `sanafe.load_arch(custom_yaml)` when a custom path is supplied). |
| `net_synth.py` | `build_network_for_segment` | Per-segment SANA-FE network construction.  One neuron group per HardCore, one `input` group, optional `always_on` group; walks `core.axon_sources` and synthesizes weighted synapses (zero entries elided; duplicate sources collapse into a single summed-weight synapse — matches the Lava convention). |
| `neuron_model.py` | `soma_attributes`, `needs_plugin`, `resolve_plugin_path`, `THRESHOLDING_MODE_TO_SANAFE` | Mapping from `SubtractiveLIFReset` semantics to SANA-FE's `loihi_lif_soma` model_attributes.  `needs_plugin()` selects Strategy A (built-in soma + attributes) vs Strategy B (custom C++ plugin); the runner is plugin-agnostic. |
| `runner.py` | `SanafeRunner` | The only module that calls `sanafe.SpikingChip`.  Walks hybrid stages, runs neural stages on SANA-FE and compute stages on host (via `hybrid_execution.execute_compute_op_numpy`), aggregates into `SanafeRunRecord`. |

## Neuron-model parity strategy

SANA-FE's stock `loihi_lif_soma` is configured with `leak_decay=0`,
`input_decay=0`, `reset_mode="subtract"`, and a strict / inclusive
`threshold_mode`.  Active-window gating (the original
`SubtractiveLIFReset` contract from
`mimarsinan/chip_simulation/subtractive_lif.py`) is emulated host-side
by gating which timesteps receive input spikes; with `leak_decay=0` the
voltage is preserved through zero-input cycles, so soma- and input-side
gating are equivalent.

If the slow single-core parity test in
`tests/unit/chip_simulation/test_sanafe_runner_single_core.py` (yet to
land — see plan) reveals a knob the built-in soma cannot match,
`plugins/mimarsinan_subtractive_lif.cpp` is built by
`scripts/bootstrap_sanafe.sh MIMARSINAN_SANAFE_PLUGIN=1` and the runner
picks it up via `neuron_model.resolve_plugin_path`.

## Dependencies

- **Internal**: `chip_simulation` (`spike_recorder`, `hybrid_execution`,
  `_spike_encoding`); `mapping` (`HybridHardCoreMapping`,
  `HardCoreMapping`, `HardCore` shape); `code_generation`
  (`SpikeSource` axon source kind).
- **External**: `numpy`, `torch`; lazily, `sanafe` (GPL-3.0; only when
  the step is enabled at runtime).

## Dependents

- `pipelining.pipeline_steps.sanafe_simulation_step` instantiates
  `SanafeRunner` and persists `SanafeStepReport` to the cache.
- `gui.snapshot.builders.snapshot_sanafe_simulation` consumes
  `SanafeStepReport.to_snapshot_dict()` for the GUI tab.

## Exported API (__init__.py)

`SanafeRunRecord`, `SanafeSegmentRecord`, `SanafeCoreRecord`,
`SanafeTileRecord`, `SanafeEnergyBreakdown`.
