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
| `neural_segment_executor.py` | Dispatches analytical neural-segment execution by spiking mode (TTFS analytical path for references) |
| `pareto.py` | Pareto decision layer over campaign rows: cascaded-vs-synchronized verdict + recipe proposal with banded cost |
| `parity_contract.py` | Parity/equivalence contract classification for deployment cells |
| `semantic_axis_screen.py` | Measured equivalence screen deciding whether a semantic knob (pruning/regime) collapses or stays enumerated |
| `spiking_mode_policy.py` | Behavior-carrying `SpikingModePolicy` per `(firing × sync)`; `policy_for_spiking_mode` is the mode-dispatch SSOT |
| `spiking_semantics.py` | Spiking-mode taxonomy + per-backend capability matrix (`_BACKEND_CAPS`), queried through the policy |
| `subsample.py` | Seeded test-subsample index SSOT shared by SCM/HCM/nevresim evaluation |
| `subtractive_lif.py` | Subtractive-reset LIF process + float model for Lava (top-level so Lava's model scan finds it) |
| `weight_reuse_cost_model.py` | Defensible per-phase weight-reuse DMA/sync cost model with a low/nominal/high uncertainty band |
| `hybrid_run/` | Shared hybrid stage loop, segment I/O + compute-op execution, and the inter-stage semantics contract for all hybrid backends |
| `lava_loihi/` | Host-scheduled Lava Loihi LIF backend: runner, wave-parallel per-segment execution (longest-path dependency waves through the bounded spawn pool), and timing |
| `nevresim/` | Nevresim C++ simulator bridge: driver, compile, execute, segment binaries, compile cache, connectivity mode, profiling |
| `parity/` | Generic segment-record field-diff comparison utilities |
| `recording/` | Spike encoding modes plus spike-count recording/diffing shared by HCM and backend parity checks |
| `sanafe/` | SANA-FE detailed-stats backend: arch/net synthesis, runner, neuron plugins, records, energy analysis |
| `simulation_runner/` | `SimulationRunner` orchestrating end-to-end nevresim runs (flat single-segment and hybrid multi-segment) |
| `ttfs/` | TTFS execution: encoding kernels, analytical executor + hybrid contract runner, segment arrays, genuine cycle sim, recorder |

## Dependencies
- **`mapping`** — `HardCoreMapping`/`HybridHardCoreMapping` inputs, `ChipLatency` scheduling, core geometry and spike-source spans, IR `ComputeOp`, chip export.
- **`code_generation`** — `ChipModel`/`SpikeSource` C++ chip model and nevresim main-function generation.
- **`models`** — spiking wire semantics and TTFS/LIF numpy kernels shared with the torch spiking nodes.
- **`common`** — file utilities, C++20 compiler discovery, `loihi_quiet`/`loihi_wave_workers` env helpers.
- **`pipelining`** — `DeploymentPlan` (firing-strategy enforcement) and `nf_scm_parity` record comparison (lazy imports avoid the cycle).
- **`data_handling`** — `DataLoaderFactory` for simulation input providers.
- **`spiking`** — the inter-stage segment-boundary SSOT: `decode_segment_output` plus the rate/LIF wire-domain transcode (`boundary_normalization_scales`, `normalize_boundary_slices_numpy`) applied at every runner's segment-input assembly.

## Dependents
- **`models`** — hybrid core-flow stage loop/semantics, spike recording, spiking-mode predicates and policies, TTFS encoding, firing strategies.
- **`pipelining`** — simulation/Loihi/SANA-FE verification steps, `DeploymentPlan`, `simulation_factory`, backend-registry step selection, certification + cost extraction.
- **`tuning`** — spiking-mode policy and semantics for conversion/calibration policies.
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
