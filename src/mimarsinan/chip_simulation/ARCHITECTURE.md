# chip_simulation/ — Chip simulation backends, deployment-semantics SSOTs, and deployment-science instruments

Runs mapped hard-core networks on the chip simulators (nevresim C++, SANA-FE, Lava Loihi,
and analytical TTFS/LIF references) to produce the deployed-forward accuracy number the
pipeline reports. The central abstractions are the `(firing × sync)` semantics SSOTs —
`SpikingDeploymentContract`, `SpikingModePolicy`, and the per-backend capability matrix in
`spiking_semantics` — consumed by every backend and by the torch spiking nodes, plus the
capability-validated `BACKEND_REGISTRY` that selects and validates simulation steps at
pipeline assembly. It also hosts pure-data deployment-science instruments: certification
floors, cost extraction, the hypervolume coverage ledger, parity/semantic screens, and the
Pareto decision layer.

## Key files
| File | Purpose |
|---|---|
| `backend.py` | `Backend` interface + capability-validated `BACKEND_REGISTRY` that selects/validates enabled backend steps up-front |
| `behavior_config.py` | `NeuralBehaviorConfig`: simulator-facing activation semantics (reset, comparison, spike encode) |
| `certification.py` | Per-`(firing × sync × backend)` regression-floor freezing + `certify` gate on deployed accuracy and wall-clock budget |
| `activation_axes.py` | The authorable axis vocabulary itself (`spiking_family` / `spiking_variant` constants and their legal sets), the axis-legality validators (`require_known_spiking_family` / `require_known_spiking_axes`), the retired-key list, and the meaning-preserving reverse bridge `axes_from_legacy` — the layer beneath `activation_semantics.py`, which re-exports every name |
| `activation_semantics.py` | Authored activation-semantics axes SSOT: `(spiking_family × spiking_variant)` vocabulary, canonical six-point mode ids, the meaning-preserving legacy bridge (`axes_from_legacy`, `fold_spiking_axes`), legal/derived variant sets, retired-key list |
| `core_semantics.py` | Core-semantics taxonomy SSOT: the chip-domain axis (`spiking` vs value-domain `mvm`), queried by intent; owns the inert spiking-mode sentinel |
| `cost_extraction.py` | `CostRecord`/`CostScatter`: mines sim-run artifacts into a cell-keyed accuracy×cost scatter with Pareto front |
| `coverage_ci.py` | CI guards that fail loud on each way the coverage instrument could lie (unscreened collapse, merged tiers, aged flags) |
| `coverage_ledger.py` | Compatibility façade re-exporting the coverage instrument split across `hypervolume_*` and `coverage_*` modules |
| `coverage_reporting.py` | Coverage aggregation: GROUP BY hypervolume cell × validity tier into a `CoverageReport` |
| `coverage_rows.py` | Ledger-row interpretation: validity tiers, covered cells, timestamps, and flag-op mining |
| `cross_sim_parity.py` | Cross-simulator parity screen recording measured per-(cell, backend-pair) equivalence (AGREE/DISAGREE/INAPPLICABLE) |
| `deployment_contract.py` | `SpikingDeploymentContract`: cross-side deployment-semantics SSOT; the single reader of the deployment config keys |
| `deployment_faithfulness.py` | Pure-data registry of standing faithfulness gates and external-dependency (sanafe/lava/ffcv) boundary guards |
| `execution_bounds.py` | Wall-cap SSOT for external-simulator invocations: `simulation_step_timeout_s` resolution (env override > config > 900 s), process-group kill, retry-once-then-fail-loud, bounded process pool, in-process watchdog |
| `firing_strategy.py` | `FiringStrategy`/factory: SSOT for LIF firing-mode reset+threshold semantics and their backend gates |
| `hypervolume_axes.py` | Typed hypervolume axis model: deployment axes, screening statuses, and collapse rules |
| `hypervolume_axis_encoder.py` | Canonical config/plan/ledger-row → hypervolume-coordinate encoding |
| `hypervolume_cells.py` | Full-tuple `HypervolumeCell` (extends the certification cell) + claimed sub-product enumeration |
| `ledger_schema.py` | Normalized campaign-ledger science-row schema: axes, cell key, validity, timing, and cost provenance |
| `membrane_export.py` | [C2] deployed membrane-readout honesty gate: every enabled backend must export final membranes (declared on the registry), else fail toward counts; half-step charge SSOT |
| `mvm_core_policy.py` | `MvmCorePolicy`: the value-domain (MVM) policy — typed answers on the consumed seams (`values` observable, no backends), event seams stay loud |
| `value_run/` | Value-domain program execution: `ValueHybridCoreFlow` + the per-core affine kernel (`run_neural_segment_values`) — latency tiers as dependency order, span-plan gathers, fp64 certification mode, no cycle loop; `activation_bits` snaps entry cores' input-sourced columns onto the calibrated boundary grid (twin of the model-side `ValueGridQuantizer`); weight tensors are uploaded per forward and live per residency chain — `schedule_weights_resident` passes alias the chain head's uploaded tensors [wsm V3], and each chain is freed when the next non-resident neural stage begins or the forward returns |
| `neural_segment_executor.py` | Dispatches analytical neural-segment execution by spiking mode (TTFS analytical path for references) |
| `pareto.py` | Pareto decision layer over campaign rows: cascaded-vs-synchronized verdict + recipe proposal with banded cost |
| `parity_contract.py` | Parity/equivalence contract classification for deployment cells |
| `semantic_axis_screen.py` | Measured equivalence screen deciding whether a semantic knob (pruning/regime) collapses or stays enumerated |
| `spiking_mode_policy.py` | Behavior-carrying `SpikingModePolicy` per `(firing × sync)`; `policy_for_spiking_mode` is the mode-dispatch SSOT |
| `stage_timesteps.py` | [E1] The executed-window rule, ONE home for the two readers that must agree: the SANA-FE runner (which sizes the simulation) and candidate-time pricing (which prices the wall before any simulation exists). A stage runs `T + max_latency + 1` (the +1 is input delivery), a cycle-based TTFS window runs `(latency_groups + 1) x T`, a cascade spans the full ChipLatency; `program_latency_steps` sums over the program's execution stages — one per depth LEVEL when per-hop re-timing is armed (which `lif_exact_qat` pairs on), else one per segment. The candidate previously used `timesteps x neural_segment_count`, a second formula that measured 4 where the sealed record measured 15. |
| `spiking_semantics.py` | Spiking-mode taxonomy + per-backend capability matrix (`_BACKEND_CAPS`), queried through the policy |
| `synaptic_events.py` | [H1] The measured arrival census, ONE home for the estimand both planes must share: one synaptic event = one spike arriving at one OCCUPIED cell (used column) of a consumer row — matching the candidate's model (`onchip_macs x timesteps x activity`) so measured/(cells x steps) IS the effective activity. Joins the HCM's own axon-source spans with the trace's per-core LIF emissions and per-core input/always-on tallies; refuses (None) when a producer's trace was unparsed, because a partial count priced as a full one is the silent-zero shape. The SANA-FE finalizer computes it per stage, the snapshot aggregates with None-propagation, and `EnergyRecord.synaptic_events` seals the per-inference mean — making `energy_per_inference_mj` priceable on the RECORD plane for the first time. |
| `subsample.py` | Seeded test-subsample index SSOT shared by SCM/HCM/nevresim evaluation |
| `subtractive_lif.py` | Subtractive-reset LIF process + float model for Lava (top-level so Lava's model scan finds it) |
| `weight_reuse_cost_model.py` | Defensible per-phase weight-reuse DMA/sync cost model with a low/nominal/high uncertainty band |
| `hybrid_run/` | Shared hybrid stage loop (expands a re-timed fused stage's `retimed_level_stages` through each backend's per-stage handler with execution-unit indexing — `enumerate_execution_stages` is the ordinal SSOT reference-driven runners share), segment I/O + compute-op execution, and the inter-stage semantics contract for all hybrid backends; `stage_timing.StageTimer` is the opt-in host-op wall meter (`run_hybrid_stages(stage_timer=...)`, default `None` ⇒ byte-identical) — it wraps `on_compute` invocations only, accumulating measured monotonic walls per `(stage_index, name)` for the deployment record's `ComputeOpRecord.wall_s_total`; neural segments are the chip simulator's to measure |
| `lava_loihi/` | Host-scheduled Lava Loihi LIF backend: runner, wave-parallel per-segment execution (longest-path dependency waves through the bounded spawn pool), and timing. `carry.py` is its verbatim pass-carry seam — lava is host-scheduled, so `core_output_spikes` already holds every core's emissions host-side and `_output_raster` is a windowed gather over the same output spans (each core span at its source's `[latency, latency+T)`); replay rides the shared `apply_carried_input` on the encoded train, batched. Both the accuracy loop and `run_segments_from_reference` (the parity harness) thread the seam, and the runner takes the RUN's discipline explicitly (`pass_transfer=`) because a pipeline-less parity runner defaulting to collapse while the HCM reference ran verbatim is the split-brain the parity gate caught |
| `nevresim/` | Nevresim C++ simulator bridge: driver, compile, execute, segment binaries, compile cache, connectivity mode, profiling |
| `parity/` | Generic segment-record field-diff comparison utilities; float closeness is judged by `common.measurement.MetricTolerance` (the one definition of "matches") |
| `recording/` | Spike encoding modes plus spike-count recording/diffing shared by HCM and backend parity checks; `spike_modes.comb_spike_count`/`_np` is the comb-count tie SSOT (the chip's `llround` half-away-from-zero, NOT half-to-even — odd 1/(2T)-lattice rates are edges exact-QAT trains onto) |
| `sanafe/` | SANA-FE detailed-stats backend: arch/net synthesis, runner, neuron plugins, records, energy analysis; `noc_estimate.py` is the candidate-time wireload model — `estimate_noc(fragments, cores_per_tile, mesh_height, activity_factor, timesteps)` prices the wire census's traffic on the resolved floorplan with the record's own conventions (one message per firing source neuron per destination core, x-first XY hops, input/always-on somas intra-tile input-path, on-wires every cycle, cross-pass pairs excluded as carry), refusing an undeclared activity; `noc_geometry.py` is the mesh-convention SSOT — core→(tile, local) sequential fill, tile→(x, y) column-major placement, the x-first XY route walk, and the two mesh regimes (`replicated_mesh` for the declared floorplan, `legacy_mesh` for the packed-count fallback) — which arch/net synthesis, the runner's geometry records, the trace analysis's link loads, AND the candidate NoC estimator all delegate to, so the model and the measurement cannot disagree on geometry; the NoC floorplan is resolved from the DECLARED platform (`arch_synth/floorplan.py::resolve_floorplan` — explicit `cores_per_tile`/`tile_grid_*` keys win, else preset tile wiring, else ceil(sqrt(declared)); tile grids stay exact — phantom tiles SIGFPE), so equal declared platforms give identical, cross-run-comparable floorplans; preset `custom` loads the user arch YAML (accepted iff `sanafe_custom_arch_path` is set) and adopts ITS tile grouping (`runner/custom_floorplan.py`); `runner/carry.py` publishes a scheduled segment's OUTPUT raster to the later passes of its own segment (`_compute_seg_output_raster` is the per-cycle twin of the count gather: same spans, trace row instead of accumulated count, shifted by each source core's latency into producer-local time) and the stage input replays carried trains via `apply_carried_input` — both gated on the RUN's transfer discipline (`SanafeRunner(pass_transfer=...)`), because one collapsing peer backend collapses the whole run; `SanafeRunner(time_host_stages=True)` (W4.3, default off) times host ComputeOps with a fresh per-sample `StageTimer` and surfaces the walls as `SanafeRunRecord.compute_stage_walls` |
| `simulation_runner/` | `SimulationRunner` orchestrating end-to-end nevresim runs (flat single-segment and hybrid multi-segment); `carry.py` is nevresim's verbatim pass-carry seam (segment roles computed in `emit.prepare_all_segments`, which moved there with its compile pool: PRODUCING segments add the SPKTRN define to their record build, CONSUMING segments compile in SpikeTrain input mode; `assemble_carried_input_train` builds the consuming train — host slices via the SAME encoder twin the HCM flow deploys, carried slices verbatim — and `publish_segment_trains` gathers the extracted trains into the rasters a later pass replays); `run()` executes inside `measurement_plane()` so host ComputeOps feeding the chip decide exact lattice ties by snapped values (device/batch-invariant, identical to the certificate twin); opt-in `stage_timer` (W4.3) threads the `hybrid_run.StageTimer` through the hybrid stage loop, and the flat driver path surfaces the driver's measured total-output-spikes figure as `nevresim_total_spikes` (formerly printed-and-dropped) |
| `ttfs/` | TTFS execution: encoding kernels, analytical executor + hybrid contract runner, segment arrays, genuine cycle sim, recorder |

## Dependencies
- **`mapping`** — `HardCoreMapping`/`HybridHardCoreMapping` inputs, `ChipLatency` scheduling, core geometry and spike-source spans, IR `ComputeOp`, chip export.
- **`code_generation`** — `ChipModel`/`SpikeSource` C++ chip model and nevresim main-function generation.
- **`models`** — spiking wire semantics and TTFS/LIF numpy kernels shared with the torch spiking nodes.
- **`common`** — file utilities, C++20 compiler discovery, `loihi_quiet`/`loihi_wave_workers` env helpers.
- **`pipelining`** — `DeploymentPlan` (firing-strategy enforcement) and `nf_scm_parity` record comparison (lazy imports avoid the cycle).
- **`data_handling`** — `DataLoaderFactory` for simulation input providers.
- **`spiking`** — the inter-stage segment-boundary SSOT: `decode_segment_output` plus the rate/LIF wire-domain transcode (`boundary_normalization_scales`, `normalize_boundary_slices_numpy`) applied at every runner's segment-input assembly; `SpikingDeploymentContract` surfaces it (`boundary_config()`, `entry_quantizer()`, `seam_transcode()`).

## Dependents
- **`models`** — hybrid core-flow stage loop/semantics, spike recording, spiking-mode predicates and policies, TTFS encoding, firing strategies.
- **`pipelining`** — simulation/Loihi/SANA-FE verification steps, `DeploymentPlan`, `simulation_factory`, backend-registry step selection, certification + cost extraction.
- **`tuning`** — spiking-mode policy and semantics for conversion/calibration policies; `orchestration/run_instrumentation` imports `cost_extraction.FT_PASS_WALLS_FILENAME` (the walls-artifact name/shape SSOT shared by writer and reader).
- **`spiking`** — `spike_modes` encoding for spike trains and the TTFS segment policy.
- **`mapping`** — spiking semantics in pruning liveness; mode policy in bias compensation.
- **`config_schema`** — spiking semantics for deployment derivation and config validation.
- **`model_training`** — `compute_test_subsample_indices` for subsampled test evaluation.
- **`common`** — default nevresim connectivity mode in `file_utils` (lazy).
- **`code_generation`** — nevresim exec-policy resolution via `spiking_mode_policy`.

## Exported API
- Backend registry: `Backend`, `SimulationBackend`, `BackendRegistry`, `BACKEND_REGISTRY`.
- Certification: `CertificationCell`, `RegressionFloor`, `CertificationFloorBook`, `CertificationStatus`, `CertificationVerdict`, `certify`, `freeze_cell`, `load_floor_book`, `save_floor_book`.
- Cost extraction: `CostRecord`, `CostScatter`, `extract_cost_record`, `extract_cost_record_from_run`, `load_cost_record`, `save_cost_record`.
- Weight-reuse cost model: `DmaCostCoefficients`, `CoefficientBand`, `CostBand`, `PhaseCostBreakdown`, `DEFAULT_COEFFICIENT_BAND`, `phase_cost_model`, `phase_cost_band`, `vgg16_224_scheduled_cost`.
- Coverage ledger: `AXES`, `HypervolumeAxis`, `HypervolumeCell`, `ScreeningStatus`, `AttributionFidelity`, `FlagMetadata`, `CoverageStatus`, `CoverageReport`, `coverage_report`, `claimed_subproduct`, `honest_claimed_subproduct`, `interacting_axes`.
- Axis encoding: `AxisCoordinates`, `cell_coordinates_from_row`, `quantization_axis`, `pruning_axis`, `regime_axis`, `syncs_from_row`.
- Ledger schema: `LEDGER_SCHEMA_VERSION`, `LedgerSchemaError`, `normalize_ledger_record`, `normalize_planned_ledger_row`, `normalize_step_metrics`, `with_relative_timing`, `fastest_successful_baseline_wall_s`.
- Coverage CI guards: `CoverageGuardError`, `assert_axes_screening_sound`, `assert_no_merged_valid_tiers`, `assert_no_aged_unowned_flags`, `audit_coverage_instrument`.
- Cross-sim parity screen: `CrossSimState`, `CrossSimOutcome`, `CrossSimParityError`, `derive_applicability`, `screen_cell_pair`, `write_cross_sim_screen`, `assert_cross_sim_screen_sound`.
- Semantic-axis screen: `SemanticAxisState`, `SemanticPairOutcome`, `SemanticScreenError`, `SEMANTIC_AXES`, `screen_semantic_axis`, `write_semantic_screen`, `assert_semantic_screen_sound`, `screen_live_regime`, `screen_live_pruning`.
- Pareto layer: `CostProxyBand`, `ScheduleVerdict`, `CascadeVsSyncVerdict`, `RecipeProposal`, `pareto_front`, `schedule_cost_band`, `cascaded_vs_synchronized`, `propose_recipe`, `load_deep_cnn_rows`.
- `spike_modes` (re-exported from `recording/`): torch spike-encoding implementations.
