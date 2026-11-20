# pipelining/ — Pipeline engine: turns a deployment config into an ordered, cache-backed, contract-verified step run

This module owns the end-to-end deployment pipeline: parsing a deployment-config
JSON, resolving it into a declarative `DeploymentPlan`, assembling the ordered
step sequence via a contract-driven `StepPlan`, and executing it with the
`Pipeline` engine (persisted `PipelineCache`, requires/promises data contracts,
per-step metric-retention tolerance, pre/post/step-failed hooks — the failure
hooks observe a dying step with its error and the exception re-raises
untouched). `PipelineSession` in
`session.py` is the composition root the `run.py` entry point drives; the
concrete `PipelineStep` implementations live under `pipeline_steps/`.

## Key files
| File | Purpose |
|---|---|
| `session.py` | Composition root: `parse_deployment_config` → `ParsedDeploymentConfig`, and `PipelineSession` owning one configured `DeploymentPipeline` (presets, GUI attach, start/stop-step resolution, run lifecycle); it APPLIES the `PipelineSession/determinism` contract before any step runs (`core/determinism.apply_determinism`, re-exported here: seeds every RNG family from `DeploymentPlan.seed` and pins deterministic fp32 math, TF32 off). Surfaces deployment advisories (warnings only, `best_effort`-scoped): config advisories at `run()` start, the post-pretrain envelope gate via a post-step hook. [U-LIF2 r4] A run takes the point its working directory holds (`stored_point.run_directory_point`, read before its document is written) and its doors and pipeline run at it (`run_at_directory_point`); the session then persists the plan's point. |
| `determinism.py` | The run's randomness in ONE module: `apply_determinism` (the `PipelineSession/determinism` contract — seed every RNG family from `DeploymentPlan.seed`, pin deterministic fp32 math) beside its counterpart `isolated_rng_stream` (run a block without moving the stream the deployment draws from — RE-SEEDING included, not merely rewound), so the families one seeds and the other restores cannot drift apart. The architecture-search step runs its whole search inside it: a search CHOOSES a chip, it does not re-roll the weights the run then deploys. |
| `cache/` | `PipelineCache` — persisted step-output store keyed per step, with pluggable `LoadStoreStrategy` serializers (basic/JSON, torch model, pickle). The torch-model strategy enforces the prune-parity contract at the boundary: commit + verify on store, fail-loud verify on load; it also refuses (`UncertifiedCachedFlowError`, not a corrupt-entry error) a converted flow whose conversion certificate is missing, stale, uncertified or incomplete, so a resumed run never carries an unprobed conversion forward. `legacy_placement.py` resolves the encoding placement of flows cached BEFORE the placement stamp existed: such an artifact deserializes unstamped, which the placement guard refuses, so a pre-stamp run directory would be unresumable — `resolve_cached_flow_placements` APPLIES the configured placement to those flows (never re-stamps one that already has it) at `DeploymentPipeline` config resolution, the last moment before any step could add a host placement of its own. `legacy_soma_law.py` resolves, at the same moment, the soma point of a chip-aligned NF forward cached BEFORE the soma axes existed (it deserializes with no point and refuses at call time, never re-defaulting): `resolve_cached_forward_soma_laws` binds the run's resolved law when it is a default point and refuses by name (`SomaLawMissingError`) when the run declares another point. |
| `core/` | Engine and planning: `engine/` (`Pipeline` execution engine, resume/debug helpers; `pipeline_resume.py` starts a resume at the latest step from which every step up to the requested one finds each requirement in the cache or written by an earlier step of the run, with the entries earlier steps left unconsumed in the cache, so a start it returns resolves to itself and the session executes exactly the start it reports; no such start, like a step whose requirements are absent as it runs, refuses by name, `MissingRequirementsError`, naming the step, the entries, their writers and the remedy), `steps/` (`PipelineStep` base with class-level `REQUIRES`/`PROMISES` contracts plus trainer/tuner step bases; the tuner base persists the W3-S1 run-dir adaptation artifacts at commit time — the `ft_pass_walls.json` accumulator + one `retention_ledger.json` entry per tuner-hosting step, via `tuning.orchestration.run_instrumentation` — and [TS5] promises `adaptation_ledger`, sealing the controller's event ledger as `<Step>.adaptation_ledger.json` from `run` so the promise holds for every tuner-step shape; `pipeline_steps/verification/deployment_record_adaptation.py` reads BOTH run-dir adaptation surfaces back — the AC5 wall bundle and the TS5 ledger roll-up — and attaches them as the record's optional `adaptation` / `adaptation_ledger` fragments), `deployment_plan.py` (`DeploymentPlan` — the single config-resolution layer for every deployment axis), `step_plan.py` (`StepPlan` ordered registry filtered by each step's `applies_to(plan)`, contract-validated at assembly), `pipelines/` (`DeploymentPipeline`, `get_pipeline_step_specs` step ordering; `merge_pipeline_config` / `apply_provider_facts` / `apply_workload_profiles` are the exact config-merge + provider-fact + profile-fold seam the golden-resolution snapshot harness shares; the plan carries the folded result as `DeploymentPlan.workload`), `registry/` (`ModelRegistry`, trainer factory), plus `census/` ([U-CEN] `arithmetic.py`: the device and dtype every census of the deployed program runs in, [U-DAR] read from the record `mapping/ir/deployment_dtype.py`, announced as `[Census]` lines), `simulation_factory.py`, `nf_scm_parity.py` (NF↔SCM parity gates; every gate that reads the NF twin reads it inside the one `measurement_plane()`, the only plane where its armed lattice nodes decide exact ties as every deployed executor decides them in any plane: chip cores every cycle, host-run lattice nodes through the one host-op evaluator `chip_simulation/hybrid_run/host_compute.py`; the decision and streamed gates read both twins in the plane, and a gate between two deployed twins, the SANA-FE spike-parity gate, needs no plane), `gates/value_gates.py` (the value-domain R-edge model ≡ identity and C-edge identity ≡ packed gates, a program with dynamic operand banks judged by `certify_operand_program`; a program whose cores carry readout laws is judged at atol 0, `certification.value_certificate.value_edge_atol`; a lane program refuses a disarmed R-edge, `value_parity_samples` 0, by name, `LaneParityDisarmedError`), `gates/static_program_gate.py` ([W15] the static per-deployment check at Hard Core Mapping, beside the sampled certificates: the run's declaration, Theorem 1 and Proposition 2 apart, the check's wall against the program's compile wall, the report as `static_program_check.json` in the run directory and two certificate records into the `deployment_record_hcm` fragment; it never refuses a run, and a check that cannot run is recorded failed by its exception; the enabled backends' accumulation reaches come off the plan (`backend_accumulation`: nevresim's C++ `int` or `double`), and a mapping the step loaded from the cache (`hybrid_mapping_consumer.cached_hybrid_mapping`) is recorded `compile_source` loaded, its load wall never priced as a compile), `spike_count_gate.py` (deployed spike-count certificate gate, `spike_count_parity_samples`; on PASS returns `(cert, boundary_traffic)` — the packed program's per-node counts reduced eagerly at gate scope via `deployment_record.build.from_certificates`, raw tensors never escaping), `accuracy_budget.py`, `platform_constraints_resolver.py` (`build_platform_constraints_resolved`, ONE function of the flat config, which it first reads as its consumers read it, `config_schema/registry/consumer_reading.read_like_consumers`, exactly as the document door reads a document, so a spelling a consumer normalizes resolves to the value the consumer runs and an explicit per-layer S map whose every entry equals `simulation_steps` reads as uniform (the map itself is judged as written, a non-empty list of positive ints); its `declared_surface` is where a run judges the declarations it never validates: `judged_passthrough` judges every present key the registry gives a value schema (the Wave 1 keys, `cores`, and the keys whose legal set is their consumer's, `nevresim_connectivity_mode`, `s_allocation` and its per-depth map `s_allocation_explicit`) or the passthrough flag, by the schema the document door uses, and its domain tag where its presence is a declaration (a key without a schema default), refusing by name with `DeclarationValueError`; the passthrough keys (`target`, `target_overrides`, the two bundle labels, `readout_law`, `inference_batch_K`, `write_overlap`, `write_ports`, `inference_rate_hz`, `min_lifetime_hours`, `residency_plan`) ride verbatim and an absent one does not appear; and exactly for a value-domain run the surface names `core_semantics`, with `activation_bits` when declared, the one surface both pricing planes read — every spiking resolution is unchanged; a core type without its own `has_bias` takes the platform bias mode, `config_schema/registry/entries_write.platform_bias_mode`: the flat `has_bias` first, then the nested `platform_constraints.has_bias` a flat config may carry, else a bias lane; both are judged as booleans, and a flat and a nested value that disagree refuse by name, by the same rule the document door applies to the layout a run gives its two sections, `entries_write.conflicting_platform_bias_error`), `hybrid_mapping_consumer.py`, `search_mode.py`, `model_config_emit.py`. |
| `pipeline_steps/` | Concrete step implementations grouped by phase: `config/` (architecture search, model configuration/building, torch mapping, weight preloading — which asserts the accuracy the freshly loaded weights produce against the weight set's recorded `expected_accuracy` through `common.measurement`, disarmed by `unusable_baseline_reason` whenever the builder adapted the workload), `training/` (pretraining), `adaptation/` (activation/clamp/shift/pruning/LIF/TTFS-cycle/noise tuner steps, the pre-WQ `LIFAffineFoldStep` [C4], plus the registry-gated `ScaleMigrationStep` — exact cross-layer channel-scale migration between Pruning Adaptation and Activation Analysis, postcondition-checked to preserve the float function — and the exact-QAT-KD-gated `ReferenceTeacherSnapshotStep` — freezes the post-structural float model right after Scale Migration as the exact-QAT distillation teacher), `quantization/` (normalization fusion, activation/weight quantization, the mvm-only `BoundaryQuantizationStep` — platform `activation_bits` calibrates + installs value-grid quantizers at segment entries by the range rule `mapping.support.range_rule` (each boundary's scale the least squared error of its own grid; the clamp census per boundary in its line and its `boundary_quantization_report`, sealed as the record's optional `boundary_ranges`), the value-domain `ReadoutLawStep` ("Readout Law", right after Boundary Quantization, applies iff `readout_law` is declared) — reads the law from the resolved platform, refuses by name (`ReadoutLawDeclarationError`) without `activation_bits` or weight quantization and (`ReadoutLawCalibrationError`) when calibration never reaches a perceptron, installs the STE `ReadoutQuantizer` in every perceptron's scaler slot, ranges it by the same range rule (one source) or the declared `range_value`, and promises the per-perceptron clipping census as `readout_law_report` — verification), `mapping/` ([N2] `OperandLoweringStep` ("Operand Lowering", right after Torch Mapping, applies iff a core type declares `operand`): lowers every Contraction of the operand form before any calibration, refuses a spiking contract by name, promises `operand_lowering_report`; soft/hard core mapping, core quantization verification; Soft Core Mapping validates the dynamic bank bindings after the laws; Soft Core Mapping calls the declared-only hook `mapping.export.law_solver.install_readout_laws` right after IR pruning, which solves every core's law on its surviving columns and stamps it on the IR and on the model's readout quantizers, a no-op when `readout_law` is absent, and refuses by name (`ReadoutLawStaleRangeError`) a declared fixed `range_value` that a quantizer does not carry; the mapping steps also PRODUCE the W4 deployment-record fragments as JSON cache entries — Soft Core Mapping promises `deployment_record_scm` (weight-reuse plan, relay count, IR latency census) and Hard Core Mapping promises `deployment_record_hcm` (schedule/placement/utilization mirrors via `deployment_record.build.from_mapping` in `deployment_record_emission.py`, weight-programming totals, gate-reduced boundary traffic, the step's accuracy read)), `verification/` (nevresim, Loihi, SANA-FE simulation steps; the SANA-FE step reads the declared core capacity + `cores_per_tile`/`tile_grid_*` floorplan keys from `platform_constraints_resolved`, so the simulated NoC floorplan is fixed by the declared platform, not the packed model; the nevresim step's window-count certificate runs its HCM twin on the deployed host ([U-HDA], the record's device) and inside `measurement_plane()` — like the runner that fed the chip — so host-op tread ties are decided by lattice values, never by the device/batch GEMM dust of wherever the samples happened to live; the nevresim step also PRODUCES the W4.3 `deployment_record_nevresim` fragment — its probe read in `AccuracyReadRecord` shape, the driver's measured total-output-spikes figure, and the measured host-op walls from the opt-in `hybrid_run.StageTimer` it threads through `SimulationRunner`; `verification/` also hosts the W4.4 TERMINAL `DeploymentRecordStep` ("Deployment Record", group `record`, applies always, registered after the backend tail) — static REQUIRES `hard_core_mapping`/`deployment_record_scm`/`deployment_record_hcm`/`platform_constraints_resolved` with plan-extended INSTANCE requires (`sanafe_simulation_results` when SANA-FE is enabled, `deployment_record_nevresim` when nevresim applies — the `declared_contract` seam keeps the class contract the DAG lower bound), which re-types the cached fragments, folds StageTimer walls onto ComputeOps (name-keyed), joins SANA-FE energy/timing/NoC/floorplan via `deployment_record.build.from_simulators`, reads `ft_pass_walls.json` + `retention_ledger.json`, seals via `DeploymentRecordBuilder` against the plan matrix (`deployment_record_assembly.py` provides the `SealPlanView` adapter + identity/timing builders), writes `deployment_record.json` atomically, promises the record dict as `deployment_record`, and writes the legacy `cost_record.json` projection IFF the SANA-FE fragment is present (emission MOVED here from `SanafeSimulationStep`; fail-loud, no `best_effort`), and [C4/S] emits `fidelity.json` iff the run SEARCHED (`fidelity_emission.emit_run_fidelity` rebuilds the candidate view of the run's own deployed configuration — hardware-mode problem over the same platform resolver / builder registry / declared pruning the search step reads, the searched axes active minus the training proxy — and zips it axis-by-axis against the sealed measurement; a rebuild failure is loud, a twin defect being exactly what fidelity exists to surface); metric-neutral (`validate` carries the pipeline metric)). |

## The Range Rule of Every Calibrated Grid (U-BRANGE, U-BRANGE2 stage 1)

Boundary Quantization's entry grids and every grid Readout Law ranges (per perceptron, per consumer group, a bank's
payload, a lane's end) read one rule from one source, `mapping.support.range_rule`: the range is the least squared
error of the site's own grid over the calibration batches, the rounding and the clamp both counted, over the whole
interval up to the calibration maximum (3,073 geometric candidates, 256 an octave over 12 octaves; no quantile read;
ties to the least span; a minimum at the search's low end refused by name, `RangeSearchError`). A value is weighed by
its error, never by its rarity, so the rule clamps where the error minimum says so, at any bit width and any magnitude:
there is no absolute floor, and a site whose calibration values are all zero is refused by name with that measured
condition (`BoundaryQuantizationRefusal`, `ReadoutLawCalibrationError`, both toolchain refusals). W38 group 6's
ViT-B/16 lost its class token to the former 0.999 quantile, which clamped two values a LayerNorm writes at one
position of every image. The grids one reader reads together (a concatenation join, an operand product's banks, a
grouped producer read whole: the law solvers' own union, `mapping.support.reader_grids.read_together`) are one site,
their values pooled (`least_squares_pooled`), so the model and the chip read one range. The search
(`mapping.support.range_search`, per-call arithmetic `range_calls`, a signed window's bounds `range_bounds`) is exact:
its choice and census are the full search's bit for bit (`full=True` reads every candidate on every call); its error
values are the float64 prefix sums' (within about 1e-12 of the site's squared magnitude, `ERROR_RESOLUTION`); its
transient is bounded independent of candidates times levels, and an empty calibration call is no call. Boundary
Quantization prints, per boundary, the clamped fraction, the clamped values' share of the squared magnitude and the
largest value over the scale, and promises them as `boundary_quantization_report` (the search's least span
`low_span`), which the Deployment Record step seals as the record's optional `boundary_ranges`; a declared grid's row
is its deployed grid's census. `mapping.support.tensor_stats.safe_quantile` reads its quantile exactly above torch's
2^24 cap (order statistics by selection), never from a strided subsample. The always-on bias row's drive is a range
site of the same rule (`mapping.support.drive_rule`, stamped by `bias_row_drive_route` after Weight Quantization's
canonicalization): the code of its reader's grid whose row grid departs least from the bias, at the rows the deployment
gives it (one without bias-row splitting; under splitting the least that hold the bias on the weights' own grid, at
least a declared count, at most the largest declared core's axons less one, refused by name above).

## Declared value grids (Wave 2 WI-5a to WI-5d)

`pipeline_steps/quantization/declared_state.py` holds one hook per calibration step, each a no-op when its key is
absent: Boundary Quantization calls `declare_entry_state` after calibrating (the declared entry grids replace the
calibrated ones), Readout Law calls `declare_readout_state` (the declared readout ranges, then the identity output
law's inert output quantizers; counts in `readout_law_report`). Weight Quantization reads `projection_grids` and
hands `fractional_scale` and `float_bias` to the NAPQ tuner (a float bias skips the starved-bias
canonicalization); Quantization Verification and Core Quantization Verification accept the declared grids; Soft
Core Mapping passes `quantizer_state` to `install_boundary_grids` and `ParameterGrids` to `quantize_ir_graph`.
[U-WQB] Weight Quantization names the two-scale bias route (`register_bias = resolve_wq_two_scale_projection`, an
on-chip bias register: the bias on the weight lattice; else bias rows) and refuses by name a shift bake without its
recorded `input_shift`; Quantization Verification and Soft Core Mapping read the same route into `ParameterGrids`.

## The integer program after Weight Quantization (U-POST)

Weight Quantization projects the perceptrons the placement maps to cores (`perceptron.placed_on_cores`; a host-placed
perceptron, a subsumed encoder, keeps its float weights), and Quantization Verification checks those. The projection
records each LIF perceptron's membrane-lattice kind (`Perceptron.membrane_lattice_kind`; an encoding layer is on the
entry grid only when its input carries the 1/T entry grid), and every arming derives the lattice from that record and
the persisted `parameter_scale` (U-POST2: never cached, so a keep-best restore arms the restored scale). At its end
(`_commit_tuner_entries`) `quantization/host_normalization_recalibration.py` re-estimates every host normalization
over `wq_host_normalization_batches` training batches (default 60), kept only if the calibration read does not fall;
a kept re-estimation's read is the committed model's exit read (`TunerPipelineStep.hold_committed_read`: the
retention ledger and the D-hat high-water hold it). The step reads of an integer program (`PipelineStep._capped_test`,
Normalization Fusion's `validate`) run inside the measurement plane with the derived lattices armed
(`models.perceptron_mixer.membrane_lattice.integer_program_read`). Every model restore inside a step's tuning window
(keep-best, rollback, a fold's rollback) is `tuning/model_snapshot`'s: the state dictionary with each perceptron's
records (U-POST2).

## On-chip lanes and the output stage (Wave 2 WI-5e, W08)

With `onchip_arithmetic` declared on the integer path, Boundary Quantization installs grids only on the entries the
lanes leave (`mapping.lanes.model_lanes.lane_absorbed`), Readout Law installs the lane ends after its quantizers
(`install_planned_lanes`) and ranges each lane at its end (`measure_lane_ranges`, the clamp's output on a clamp
lane, the sum at a fused add, the pool's output on a pool lane; `lane_clipping_census` in its report), and
`declare_readout_state` reads a lane program's declared offsets and pool lane ranges (refused by name without
executed lanes). Soft Core Mapping's law hook fuses the lanes; the step then recomputes the IR latency and seals the
output stage (`mapping.lanes.output_stage.output_stage_of_program`) in its `deployment_record_scm` fragment, which
the Deployment Record step attaches as the record's optional `output_stage`. Without the declaration every step is
unchanged.

## Core types with their own weight bits (Wave 2 WI-5g)

Weight Quantization refuses typed widths outside value-domain cores by name and stamps each perceptron's width from
`layer_core_types` before its projection; Quantization Verification checks each perceptron on its own width's grid
(and, for a declared width, that its weight codes stay inside that register); Soft Core Mapping binds every IR core to
its layer's type before quantization and after the law hook and checks every core's codes; the pipeline's packer entry
(`core/simulation_factory.build_hybrid_mapping_for_pipeline`) hands the packer each type with its width written out.
Absent, every step is the base's.

## Pass replay refusals and the sealed value NoC census (Wave 2 closure)

From the closure unit's step 1 (IE-2, KA-G3-1). `pipeline_steps/mapping/hard_core_mapping_step.py` runs the built
program's replay refusals (`models/spiking/hybrid/carry.run_pass_transfer(config, hybrid_mapping=hybrid_mapping)`:
stochastic multi-pass; and, on a backend that does not read a stage input in core-local time on this checkout, a late stage-input reader or a delayed slice, which since W33 refuses on nevresim only where its tree lacks W33's read, `chip_simulation/stage_input_clock`) before any simulation; the judgement
leaves the built cores' latencies as built. `pipeline_steps/mapping/deployment_record_emission.py`: the schedule
fragment seals `value_noc_wire_hops` (`deployment_record/quantities/value_domain.record_value_noc_wire_hops`) on
value-domain runs.

## The IR at unbounded geometry and the host condition at Hard Core Mapping (Wave 2 final unit ux)

[E1 open item 1] Soft Core Mapping derives the IR through one method (`_derive_ir`: mapping, core-type binding, boundary
grids, negative-shift transfer, relays, weight quantization, IR latency, IR pruning, readout laws, bank validation,
core codes) and runs it twice: at the declared geometry (the deployed `ir_graph`, unchanged) and, on a copy of the model
taken as the first mapping finds it (`model_copy_for_reference`), with no core limits and no coalescing, its stdout,
records (the softcore-elimination record, the pre-pruning heatmap) and random draws kept out of the run. It promises
the result as `ir_graph_unbounded` (`{"ir": the IR or None, "note": "built" or the exception's name and text}`; a
second mapping that cannot run is recorded by name, never a refusal of the run). Hard Core Mapping requires it and
passes it to the static check (`gates/static_program_gate.run_static_program_check(unbounded_ir=...)`), so a pipeline
verdict establishes `ir-tiling` against the IR at unbounded geometry instead of assuming it; without it the condition
stays assumed with the note in its own note. [W49] Hard Core Mapping runs its sampled certificates (spike-count twin,
value twin, operand program) under `certification.host_condition.host_kernel_replay`; `gates/host_condition_gate.py`
prints Theorem 1's host condition (`[HostCondition] Theorem 1 host condition: ...`) and appends its `host_condition`
record after the static check's two in `deployment_record_hcm`. Neither refuses a run.

## The NF twin as the program evaluates its host kernels (Wave 2 final unit ups, U-PS)

`_derive_ir` ends by stamping the declared host-kernel evaluation (`host_kernel_evaluation`) on the program's host ops
(`mapping/support/host_kernel_evaluation.declare_host_kernel_evaluation`; absent, nothing is stamped). Wherever a parity
gate compares the NF twin with the deployed program, the twin is read as the program evaluates its host kernels
(`twin_as_program`): the four NF gates of `core/nf_scm_parity.py` (the per-neuron gate, the streamed exactness gate's
counts and rasters, the cascaded decision gate, the torch-versus-deployed gate, which reads the program off the flow),
the value R-edge (`core/gates/value_gates.run_model_value_parity_gate`) and the operand certificate's NF edge. Under
`per_sample` the twin then runs one sample at a time, its captures concatenated in order, so both sides of every gate
evaluate every host kernel on one sample alone and the gates stay exact at atol 0 (fixed, never budgeted); absent the
declaration every gate reads the twin on its batch as before. [U-TGAUGE2] The two decision gates rank both twins in one
unit, the value domain: the executor's count read decoded by its program's output gauge
(`chip_simulation/hybrid_run/output_decode.executor_output_values`), as the walk decodes its own output; the deployed
census reads the flow through `DeployedValueReadout`, and the membrane diagnostic counts the value domain's class flips.

## Dependencies
- `chip_simulation` — spiking-mode semantics and `SpikingDeploymentContract` read by `DeploymentPlan`; `BACKEND_REGISTRY` splices the simulation-step tail; simulation runners, cost extraction, and record comparison for the verification steps and parity gates.
- `deployment_record` — fragment converters (`build.from_mapping`, `build.from_certificates`) and schema types the mapping steps and the spike-count gate use to persist deployment-record fragments; the direction is one-way (`deployment_record` never imports `pipelining`).
- `mapping` — IR graph/mapping construction, packing, IR pruning, chip quantization, platform constraints, and capacity/majority verification used by the mapping steps and `simulation_factory`.
- `tuning` — tuner classes run by the adaptation/quantization steps; temporal allocation and tuning-budget orchestration read by `DeploymentPlan` and the tuner step base.
- `common` — `refusal.ToolchainRefusal` (base of `UncertifiedCachedFlowError`), `best_effort` degrade seam, diagnostics/profiling, env flags, file utils, `DefaultReporter`, the `measurement` baseline-agreement contract and the `pretrained` weight-set predicates.
- `model_training` — `BasicTrainer` construction (`registry/trainer_factory.py`), training recipes, weight-loading strategies.
- `data_handling` — data provider/loader factories for the session and steps; test-sample loading for simulation metrics.
- `models` — model layers and decorators used by steps, `SpikingHybridCoreFlow`, perceptron bias-reference refresh.
- `transformations` — normalization fusion, `PerceptronTransformer`, magnitude pruning, quantization bounds, `pruning.committed_masks` commit/verify at the cache store/load boundary, `pruning.seed_generators` (the `prune_criterion` seed seam in the soft-core structured-pruning hook), `equalize_channel_scales`/`DEFAULT_CLIP_RATIO` for the Scale Migration step and `DeploymentPlan`.
- `spiking` — cycle-accurate LIF train application and scale-aware boundary calibration; `establish_gauge_for_mixed_domain_seams` at the LIF Adaptation entry, so the wire-currency twin that step finalizes on has a domain at every host op (a graph without a heterogeneous fan-in is untouched).
- `search` — joint arch/HW search problem and result types for `ArchitectureSearchStep`, which injects the step's `make_platform_resolver(config)` as the problem's `PlatformResolver`: every searched chip is `platform_constraints_resolver.build_platform_constraints_resolved` over the declared platform plus the candidate's searched dimensions, so the winning candidate IS the deployed `platform_constraints_resolved` with nothing merged or re-stamped afterwards. [TS1] `arch_search.evaluation_budget` is the run's DECLARATION of what the search may spend — a count of distinct evaluations, resolved by `resolve_evaluation_budget` into the problem's `EvaluationBudget` and refused by name when it is not a positive count. `search_result_to_jsonable` seals the resulting `ResourceLedger` into the search artifact when an accountant metered the search, and omits the key entirely when none did — an undeclared budget leaves the run and its artifact byte-identical to what they were before the accountant existed. The candidate contract itself is assembled in one place, `pipeline_steps/config/architecture_search_problem.py` (`build_search_problem` — the run's declaration → `JointArchHwProblem` plus the active objective names; `no_candidate_failure` renders an empty search's own recorded refusals), so the step is left saying what a search step does and the fidelity twin's arming reads the same `firing_semantics_kwargs` call site. [TS2] Which BACKEND a run gets is one declarative table, `architecture_search_helpers.OPTIMIZER_BUILDERS` (name -> builder over a single `OptimizerRequest`), keyed by `search.optimizers.catalog`'s ids — the wizard offers that same catalogue and `OptimizerType` declares it, so a name cannot reach the pipeline without reaching the GUI; an unknown name is refused naming every choice. The model-side search space (`derive_arch_options`/`make_assembler`/`resolve_arch_options`) moved beside it into `pipeline_steps/config/architecture_search_space.py`, which knows nothing about optimizers.
- `config_schema` — config defaults and deployment derivation folded into pipeline configs.
- `torch_mapping` — torch-model conversion and conversion probing in `TorchMappingStep` (`convert_pipeline_model`: every probe stage the packaging requires, the identity value program for value-domain targets); `core/gates/value_gates.py` builds the R-edge identity program with `torch_mapping.value_parity.value_program.fp64_identity_program`, and the C-edge's identity reference with it too, exposing every node the packed stages write (`certification.value_certificate.exposed_nodes`: a scheduled lane program's pass carries, WI-5e).
- `gui` — JSON-safe serialization and `CompositeReporter` for GUI wiring.
- `visualization` — search-progress visualization during architecture search.
- `advisories` — deployment-advisory evaluation + surfacing (session config/post-pretrain seams, `TorchMappingStep` graph seam); `advisories` reads back only `pipelining.core.platform_constraints_resolver` (acyclic).

## Dependents
- `run.py` (entry point) — drives a full deployment run through `PipelineSession`.
- `chip_simulation` — reads `DeploymentPlan`, the platform-constraints resolver, the parity record comparator, and the pipeline step classes registered per backend.
- `tuning` — tuners and orchestration read `DeploymentPlan`, `resolve_bias_mode`, and activation utils.
- `models` — builders register themselves via `ModelRegistry`.
- `mapping` — wizard layout verification instantiates builders through `ModelRegistry`.
- `gui` — wizard schema/routes, run monitors, and collectors use `DeploymentPipeline`, step specs, and the model registry; the snapshot pruning gate consumes the canonical `PRUNING_ADAPTATION_STEP` step-name constant from `core/step_plan.py` (`core/pipelines/deployment_specs.py` re-exports it and builds the registry entry from it; the gate imports the `step_plan` original because the specs module's star-import of `pipeline_steps` cycles back through `gui`).
- `config_schema` — display-view build reads model config schemas and pipeline step specs.

## Exported API
- `Pipeline` — the step-sequencing execution engine (from `core/engine/`).
- `PipelineStep` — the abstract step base with requires/promises data contracts (from `core/steps/`).
