# tuning/ — Training-aware progressive adaptation of models toward hardware constraints

After the ANN is built and trained, the deployment pipeline uses this module to
apply hardware-motivated transformations (clamping, activation/weight
quantization, noise, ANN→SNN blend, LIF/TTFS conversion, pruning) gradually
while recovering accuracy at each step. The central abstractions are the
`SmoothAdaptationTuner` control loop and its services (`orchestration/`), the
rate-driven `AdaptationAxis` objects that know how to apply a transformation at
a fractional rate (`axes/`), and the concrete per-transformation tuners
(`tuners/`). Cross-cutting helpers at this level — target adjustment, LR
search, rate application, decision tracing — are shared by all tuner families.

## Key files
| File | Purpose |
|---|---|
| `adaptation_rate_tuner.py` | `AdaptationRateTuner`: `SmoothAdaptationTuner` driving one `AdaptationManager` rate field through a seeded `ManagerRateAxis`; base for the clamp/act-quant/noise family. |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster`: proportional target decay/growth with validation-set-sized decay and a `1 - degradation_tolerance` floor. |
| `forward_install.py` | `LazyExecutorForward` (picklable lazy cross-layer NF `model.forward` override) and `CascadeForwardInstall` (single-owner install/remove mixin). [U-POST2] `ChipAlignedNFForward` walks the gauge of the model's current thetas (`wire_gauge_seam.walk_wire_gauge`: re-derived, values only, whenever the thetas by value or a slot changed since its last derivation). |
| `lif_affine_fold.py` | [C4, `lif_deployment_exactness.md`] calibration-derived per-channel FULL affine (closed-form LS, layer-sequential) absorbing the LIF dead-zone bias: deployed chip-aligned rates fitted against the float envelope, folded into consumer effective-weight columns/bias (the negative-shift fold family) or the terminal perceptron's rows; consumers discovered via the shared `mapping.channel_axis_walk` mapper-DAG traversal (real converted graphs have no direct perceptron→perceptron edges); consumed by `LIFAffineFoldStep` before the WQ QAT. |
| `learning_rate_explorer.py` | `LRRangeFinder`: exponential LR sweep selecting the largest non-destructive LR, with optional coarse loss-slope pre-scoring; `find_lr_range_for_trainer` derives probe parameters from the `TuningBudget`. `clone_state_for_trainer` / `restore_state_for_trainer`: a trainer's snapshot (main and auxiliary models) through `model_snapshot`. |
| `model_snapshot.py` | [U-POST2] The one model snapshot every restore inside a tuning window uses (`snapshot_model_state` / `restore_model_state`, a `ModelState`): the state dictionary with each perceptron's records held outside it (its bake, install and lattice records, its non-persistent buffers), keyed by module path, so a restore returns what a reader reads and no state-dictionary key changes; what describes the parameters is derived instead (the membrane lattice at every arming; the wire gauge at every walk of the installed twin, `wire_gauge_seam.walk_wire_gauge`). |
| `perceptron_rate.py` | SSOT for applying a rate model-wide: `rebuild_activations`, `apply_manager_rate`, `set_blend_rate`, `set_surrogate_alpha`. |
| `shift_calculation.py` | `calculate_activation_shift`: activation-space shift for quantization alignment. |
| `teacher.py` | `snapshot_frozen_teacher` / `freeze_module`: eval-mode, gradient-frozen deepcopy of a model for KD recovery (deepcopy on CPU). |
| `trace.py` | `DecisionRecord` / `DecisionTrace`: JSON-round-trippable per-cycle decision trace; iterates as legacy `_cycle_log` dicts and backs the golden-trace tests. |
| `axes/` | `AdaptationAxis` contract plus concrete axes (manager-rate family, blend/LIF/TTFS, perceptron-transform/NAPQ, pruning, activation shift); each delegates math to `transformations` and rate application to `perceptron_rate`. Closure-apply axes dispatch `probe_replica` rate application to a model-targeted replica apply (never the live-bound closure), failing loud when none exists. |
| `orchestration/` | The smooth-adaptation machinery: `TunerBase`/`SmoothAdaptationTuner` (cycle/run mixins), `AdaptationManager` + factory, `TuningBudget`, `ConversionPolicy` SSOT, the per-family flag-thicket resolvers (`ttfs_adaptation_plan`, `lif_adaptation_plan`), temporal allocation, rate scheduler / recovery engine / acceptance sensor (which names its own verdict bases: instant / paired-correctness / marginal-probe) / checkpoint guard, the [TS5] `adaptation_ledger` — frozen events for the controller's REAL branches (proposal, verdict, refinement, escalation, recovery leg, endpoint stage) plus the accumulator whose totals are sums over those events and whose `completed_via` names the path the rate search exited through; emitted from the driver/scheduler/cycle/walk-recovery/endpoint seams, opt-in per host (`ledger_of` returns None and the run is byte-identical) and sealed by the tuner step as `<Step>.adaptation_ledger.json`, the monotone conversion-frontier subpackage (`frontier/`: the ONE concept behind the P4 prefix ramp, hop-staged AQ, and the hop-prefix ramp — `geometry` rate↔position/ladder SSOT, `reaffine` keep-best DFQ rung repair, the `hop_staging` and `prefix_ramp` strategies, the terminal P1'' stage `endpoint_recovery` — [C2] armed whenever engaged and ledger-funded, stopping by the [C1] convergence geometry (`endpoint_convergence_geometry` in `tuning_policy`: dip-cover min-steps + budget-scaled patience; budgets are ceilings, never mandatory burns) — and its default-off [C3] dead-run predicate + LR-backoff restart `divergence_guard`), optimization driver + fast ladder (driven by the default D-hat trust-region gate `mbh_gate` with its gate-grade measurements and verbose-flag `[MBH]` ledger in `mbh_ledger`; the LIF T-annealing realizable family `mbh_tanneal`; sync exact-kernel QAT-endpoint helpers in `adaptation_manager`; the LIF exact-QAT arm `lif_exact_qat` — predicate/markers/entry-snap installers + the LIF subsumed-ladder drop, un-subsuming the AQ ladder under the arm, lif_exact_qat_program.md §6; the §15.4 deployed-risk terminal stage `deployed_risk_finetune` — pure-torch origin-KD minimization THROUGH an injected deployed forward, keep-best ratchet seeded at entry, intra-stage checkpoint/resume), the P1'' conversion-endpoint stage (`frontier/endpoint_recovery` targeting the `dhat_highwater` pipeline SSOT, step-budgeted through the run-scoped `endpoint_steps` ledger — budgets are optimizer steps, never wall seconds: same config + same seed => same step trajectory on any platform, modulo GPU nondeterminism; `run_ledger` owns the run-scoped cache keys and their one lifecycle: fresh-run reset, per-draw snapshot/restore), the W3-S1 run-dir adaptation artifact writers (`run_instrumentation`: the step-qualified `ft_pass_walls.json` accumulator pinned to the `cost_extraction` reader shape + the per-tuner-step `retention_ledger.json`; read-modify-write with atomic replace, artifacts only — invoked by the tuner step base at commit time), the [MBH-DRAWS] best-of-N conversion-draw harness (`conversion_draws`: independent seed+k RNG streams, full-transform fp32 D-hat selection, run-ledger scope per draw; default 1 = bit-identical single draw), the [MBH-A6] install-resolution pre-flight gauges (`install_resolution/`: `capture` channel-resolved + cursor-isolated stats, `gauges` math, `report` emit seams; warn-only, corpus-conditioned thresholds), the [S1/R4] S-aware theta-quantile policy (`theta_quantile_policy`: grid levels -> calibration quantile, memo-anchored, deflate-only), blend-ramp mechanism (`blend_ramp`: `BlendActivation`, which declares the constant-OFF floor of `models.nn.off_floor` when every side its rate reads has it) + the polymorphic ramp strategies (`ramp_strategy`: value-domain proxy, genuine-ramp base, teacher↔genuine output blend), KD-blend tuner, and the uniform `RateTunerSeam`. [U-SC2] `negative_boundary_seam.py`: the negative-value boundary policy at the gauge seam (`lift_negative_boundaries`, `NegativeBoundarySeam`): on the policy's ONE scope (`mapping/support/negative_boundary.encoded_untrained_boundary`), each signed core entry is lifted in its op's own function before training (sigma and its cover kappa by the signed-seam capacity law, consumers baked), or subsumed forward when `negative_value_shift` is off; a lift carries the zero padding of the convolutions reading the op (`mapping/support/padded_output.ZeroPaddedOutput`), and its calibration walks bounded chunks (`CALIBRATION_CHUNK`, as Soft Core Mapping's minima now do); a boundary no lift carries gets Soft Core Mapping's own policy at the seam (its LIF calibration walk at the run's soma point, its shift and bake or its refusal by name; stamped `negative_shift_at_seam`, outside the later scope); every lift and seam shift is printed; `lif_exact_qat.deployed_negative_boundary_seam` binds it to Soft Core Mapping's calibration batch and the run's soma point, preserving the run's random state and the trainer's validation cursor, and `install_lif_entry_input_quantizers` (the mechanism in `lif_entry_encode`, U-LIF D1 and D2) installs the deployed entry encode on every segment entry a host value feeds, whatever the placement, at a gauge covering the calibration quantile of its entry values (the observed-traffic lift), and no input grid on a host-side encoder: under `offload` the carriers are installed before the seam decides its scope (an entry carrying the trained quantizer is the QAT's own clamp, as at Soft Core Mapping), under `subsume` after it, so a lifted signed boundary is encoded lifted. [U-LIF P1] each in-loop theta is floored at the per-cycle charge it was installed to carry at one spike per cycle (`lif_exact_qat.record_lif_charge_bounds`; the AQ decorator reads no theta below it, and the AQ exit, the D-hat clone and the LIF step's twin and endpoint pin it). |
| `tuners/` | Concrete tuners: clamp, activation adaptation/quantization/shift, normalization-aware weight quantization, noise, LIF, TTFS cycle, perceptron transform, and `pruning/`. |

## Declared gradient clip norm in the fast ladder (Wave 2 closure)

From the closure unit's step 1 (C4). The fast ladder's rung loop (`orchestration/fast_ladder.py`, `_fast_train_rung`)
reads a declared `grad_clip_norm` through `model_training/training_recipe.clip_norm_of`, the trainer's one reading:
an absent or non-positive norm clips nothing, so a norm of 0 no longer clips every rung gradient to zero.

## A rung writes what it trains (U-TGAUGE2)

Each rung step starts its backward from zero gradients and refuses by name where its optimizer holds none of the
tuner's model's parameters (`model_training/training_recipe.clear_step_gradients`): a transform trainer's step
optimizer over its trained copies, which the rung's own backward never reaches, used to step nothing while the model's
gradients accumulated.

## Dependencies
- `models` — activation types (`TTFSActivation`, `LeakyGradReLU`, `make_activation`), decorator layers (`RandomMaskAdjustmentStrategy`, `RateBuffer`, `NoisyDropout`, `SavedTensorDecorator`), and the blended-genuine / prefix-genuine spiking forwards the KD-blend and prefix-conversion ramps install.
- `model_training` — `BasicTrainer`, the perceptron-transform trainer, `build_recipe`/`build_optimizer` for recovery training, and the `metric_grade_eval` fp32 measurement seam (`mbh_ledger`).
- `transformations` — normalization-aware perceptron quantization, `PerceptronTransformer`, pruning mask application and activation-stat collection.
- `data_handling` — `DataLoaderFactory`/`DataProvider` for validation-set sizes that derive budgets and decay factors.
- `mapping` — pruning boundary policy for the pruning tuner; `channel_axis_walk.channel_aligned_consumer_targets` for the LIF affine fold's consumer discovery.
- `chip_simulation` — spiking-mode semantics read by the adaptation plans/policies; `cost_extraction.FT_PASS_WALLS_FILENAME` is the SSOT for the `run_instrumentation` walls artifact (writer and reader share one name + shape).
- `common` — `best_effort` error containment; `env` for the `MIMARSINAN_MBH_LEDGER` verbose-diagnostics flag; `workload_profile.ResolvedWorkloadProfile` for the profile-overridable clamps (budget caps, eval target, calibration extents, LR ceilings, recovery-depth law) — frozen generic defaults stay at their consumers.

## Dependents
- `pipelining` — pipeline steps construct the tuners, the adaptation-manager factory, and `tuning_budget_from_pipeline`; `deployment_plan` consumes `temporal_allocation`, `ConversionPolicy`, the optimization driver, and the calibration pipeline.
- `config_schema` — `temporal_allocation` for validation; `ConversionPolicy` for deployment derivation.
- `chip_simulation` — calibration-pipeline constants in the deployment contract.
- `model_training` — LR-explorer helpers in `basic_trainer_steps`.
- `models` — `LazyExecutorForward` for the blended genuine forward.
- `spiking` — `LIFBlendActivation` from the LIF adaptation tuner.
- `mapping` — `calculate_activation_shift` for bias compensation.
- `gui` — `S_ALLOCATION_MODES` in the wizard schema.

## Exported API
- `AdaptationManager` — per-perceptron decorator-rate host (re-exported from `orchestration/`).
- `LRRangeFinder` — the LR range search.
- `calculate_activation_shift` — quantization shift amount.

[U-RSHIFT] The negative-boundary seam (`orchestration/negative_boundary_seam.py`) and the opt-in signed-seam install
(`orchestration/signed_seam_install.py`) bake each lifted boundary's readers through
`mapping/support/received_shift.bake_received_shift`, every lift of one call at once; a lift is taken only where today's
admission holds (`_lift_admits`, `_bake_walk_feasible`) and its field refuses nothing (`_lifts_admitted`). The kept
`_bake` and `_bake_shift_into_host_bias` are callers of the one writer with the lift on every axon. `lif_affine_fold.py`
folds a producer's offset into its consumers as each receives it (`fold_affine_into_consumers`: the one writer at sign
+1, nothing recorded); `fold_affine_into_consumer` keeps its form for a direct call.
