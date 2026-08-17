# search/ — Multi-objective architecture + hardware search subsystem

Provides the search framework the pipeline's architecture-search step uses to
co-optimize model architecture (NAS) and neuromorphic hardware platform
choices. Central abstractions: `SearchProblem` (validate/evaluate a candidate
configuration), `SearchOptimizer` backends (NSGA-II, AgentEvolve LLM,
compilagent session) that produce a `SearchResult` of `Candidate`s over a
declared `ObjectiveSpec` set, and `SearchSpaceDescription` — the single source
of truth for the joint NAS + HW search space, rendered per backend.

## Key files
| File | Purpose |
|---|---|
| `problem.py` | `SearchProblem` protocol (validate, validate_detailed, evaluate, constraint_violation, meta), `ValidationResult` carrying failure details, and `CandidateInfeasibleError` — the typed candidate-dependent failure optimizers convert to penalties while everything else aborts |
| `constraints.py` | [C3] Typed feasibility constraints: a deployment constraint shapes the FEASIBLE REGION, so a violation belongs in the optimizer's own constraint channel — typed, with the measurement that explains it — rather than as a pipeline exception a run discovers after it has already picked a winner. `onchip_floor_violation` reports the on-chip parameter floor as a `ConstraintReport` whose `violation` is the SHORTFALL, a gradient the optimizer can descend (a near-miss ranks above a candidate that put almost nothing on chip); a floor of zero is no constraint at all rather than a zero-width one. The downstream pipeline gate stays as defence in depth. |
| `option_axes.py` | [C3] Deployment OPTIONS as decision variables — placement, schedule policy, weight width, pruning fraction — where promoting one is a DECLARATION, not code: `build_option_axes` derives each axis' section, legal values, bounds, label and documentation from the config-key registry, so the search never duplicates the configurability SSOT and a key the registry cannot describe as a range is refused at declaration time rather than becoming an axis a candidate never varies along. An axis carries exactly one shape — `choices` (index-coded, the arch-option pattern) or `bounds` (numeric, integral when the registry says INT) — because both would be two encodings of one axis and neither would be an axis at all; `decode_option_value` clips, never wraps. `candidate_option` is the ONE reader of a candidate's `deployment_options`, so a searched option and a merely-declared one are never told apart by hand at a call site. |
| `results.py` | The legacy PROJECTION of `deployment_record.objectives` — `ALL_OBJECTIVES` is that registry's search catalogue rendered as the frozen `ObjectiveSpec(name, goal)` tuple every optimizer indexes (byte-equality pinned by test); `resolve_active_specs` resolves the ACTIVE registry specs (what an evaluation must READ off a view) and `objectives_for_mode`/`resolve_active_objectives` project them, all FAILING LOUD on an unknown or mode-unavailable objective (the silent drop is gone); the per-search-mode DEFAULTS stay here (a search policy, not a record fact), together with the `Candidate`/`SearchResult` containers and the ONE minimax ranking (`rank_objective_rows`/`order_by_minimax_rank`, and `select_minimax_rank` on top of it) that candidate selection, the reported Pareto orderings, and the live panel's front all read |
| `search_space_description.py` | `SearchSpaceDescription` SSOT for the joint NAS + HW space (`CORE_DIM_GRANULARITY`), with renderers to AgentEvolve prompt schema/example/constraints and compilagent levers |
| `search_space_compilagent.py` | Renders a `SearchSpaceDescription` into compilagent `Lever` tuples and derives sampled integer candidates per HW dimension |
| `patch_borders.py` | `get_region_borders`: standalone patch-region border computation utility (no in-repo callers) |
| `evaluators/` | Fast NAS accuracy evaluators: one-epoch `FastAccuracyEvaluator` and `ExtrapolatingAccuracyEvaluator` with parametric learning-curve fitting |
| `optimizers/` | `SearchOptimizer` interface and backends: pymoo NSGA-II, AgentEvolve LLM evolution, compilagent session (with `MimarsinanLayoutBackend`), shared LLM trace utilities, and `search_events.py` — the live search-event channel's SSOT (the `emit_search_event` envelope plus the `generation_start`/`candidates_generated`/`generation_complete`/`search_complete` frame constructors). The classical and LLM backends BUILD their generation frames there, so the panel's vocabulary cannot fork; emission is telemetry and degrades through `best_effort`. The compilagent introspection surface is DERIVED from `deployment_record.introspection`'s registry: one read-only tool per payload the candidate view can answer, each response carrying its `payload`/`payload_version`, so registering a payload reaches the agent without a hand-written tool. These modules import the introspection types and NOTHING from `mapping` (AST-pinned) |
| `problems/` | Concrete problems: `EncodedProblem` (vector-encoded) protocol and `JointArchHwProblem` for joint architecture + hardware co-search — see "The problem surface" below |

## The problem surface

`JointArchHwProblem` answers two questions, each with exactly one implementation.

**Which chip is this candidate?** — `platform_resolver`, an injected
`Callable[[overlay], resolved_platform]`. The pipeline step curries the
DEPLOYMENT's own `build_platform_constraints_resolved` over the run's declared
platform (`make_platform_resolver`), so a candidate chip is that resolution with
only the searched declarations overlaid: core dimensions (snapped to
`CORE_DIM_GRANULARITY`), `target_tq`, and the platform's bias capability stamped
onto the new core types. Nothing is hand-carried, so a resolver key added
tomorrow (a scheduling policy, a NoC floorplan derivation) reaches the search for
free, and the searched chip cannot differ from its deployed twin. The resolution
is a FIXPOINT, so every entrance passes through it unconditionally:
`decode` (the encoded optimizers), `_resolved_configuration` at the
`evaluate`/`validate_detailed`/`constraint_violation` boundary (the LLM
optimizers, which declare platforms as JSON rather than vectors), and the base
itself (`fixed_platform_constraints` == the empty overlay). The resolver takes
no mode switch: it once omitted `allow_neuron_splitting` for the search only,
and since `ChipCapabilities` reads an absent permission as DENIED, candidates
were scored on a chip that could not split neurons while the run deploying them
could — a searched chip must resolve exactly as the same chip declared by hand.

**What is this candidate worth?** — one path, `_resolve_model` → `_resolve_layout`
→ one `CandidateStaticView` → `{spec.key: spec.value(view)}` over the ACTIVE
registry specs. The view's facts (model build → layout collection → packing
census) are computed once and only when an active axis needs them:
`_requires_fragment` asks the registry — via `candidate_probe_without` — whether
any active axis goes unavailable without a fragment, so accuracy is trained
exactly where the mode carries the axis and the mapping is skipped when no axis
reads a layout. [N0] The view carries the `physics` and `quantity_context`
fragments off the CANDIDATE's own resolved platform
(`joint/candidate_fragments.py`: `candidate_fragments` reads
`pcfg["platform_physics_resolved"]` + `candidate_context_from_platform`;
`compute_onchip_census` adds the host/on-chip param+MAC split through the
deployment's own estimator, gated on `_requires_fragment("quantity_context")`
and memoized on the hardware-only fixture) — the axis gate
(`resolve_active_specs(physics=...)`) and the view extraction must answer from
one source, or an admitted priced axis raises at extraction on every candidate.
[E1/E2] The same file derives what the candidate's PASS STRUCTURE implies:
`candidate_program_facts` returns the executed wall (`candidate_latency_steps`
applies `chip_simulation.stage_timesteps` — the runner's own rule — to the
candidate's per-execution-stage latencies) and the programming census
(`candidate_programming_census`: cores over every pass, plus cores/bytes over
the passes that actually install weights, under the DEPLOYED residency law
`mapping.support.schedule.pass_planner.resident_passes` that
`mark_bank_residency` also reads). Both are ABSENT together when there is no
pass structure, so the terms multiplying them refuse by name instead of
pricing zero; `cores_allocated` is that same per-pass count, the meaning the
record gives it. Payload bytes go through the record's own
`deployment_record.build.payload_sizes.params_bytes`, and stay absent when the
platform declares no weight width. [H2] The same facts also size the CARRY of
the planned program: `candidate_program_facts` runs the record's own
`carry_census_from_spans` over `carried_softcore_spans` (pass membership +
wire-census adjacency) under the run's transfer discipline (`pass_transfer`,
resolved by the deployment's `run_pass_transfer` and handed down by the search
step) — a sealed structure with nothing crossing claims KNOWN ZEROS (a
single-pass program is the best carry, never an unknown), while a missing
census, discipline or layout keeps the quantities absent.
Model construction lives in `joint/model_build.py` (`build_raw_model`,
`convert_to_mapper_repr`), which the hook delegates to. [P] `build_raw_model`
applies the RUN's DECLARED pruning to every candidate model
(`joint/candidate_pruning.py`): `prune_sparsity` runs the deployed
`prune_perceptron_chain` itself (weight-independent counts — candidate shapes
== deployed shapes by construction); the pruning tuner's
`pruning`/`pruning_fraction` applies the mask floor-count shrink (the same
`mask_prune_count` formula, the same IO exemptions via
`build_boundary_ir_graph`, propagation folded to its conservative `max` —
the cascade's weights-dependent harvest means the deployed program is never
LARGER, a stated upper bound); foreign `prune_criterion` values keep shapes
(same statement at zero elimination); pre-fusion candidate norms are sliced
to the kept channels. Pruning is NEVER a decision variable: `option_axes.
REFUSED_OPTION_AXES` refuses `pruning`/`pruning_fraction`/`prune_sparsity`
by name at declaration time (accuracy impact unmodeled at candidate time),
and the dead searched-pruning reader (`candidate_pruning_fraction`) is gone. [N3] When a NoC axis
is active (`_requires_fragment("noc_fragments")`), `_collect_softcores` walks
with `collect_wire_census=True` and `candidate_fragments.collect_candidate_noc`
plans+packs every pass under the candidate's OWN capability bits
(`layout_kwargs()`, the packing-census discipline) — the resulting
`LayoutNocFragments` ride `CandidateLayout.noc` and the view's `noc_fragments`
fragment, which the objectives layer prices through the SANA-FE wireload
estimator. A hardware-only search reuses the candidate-INDEPENDENT model
and its mapper representation, never its layout: tiling is a function of the
candidate's core geometry, so each candidate is packed on the chip it actually
declares. The validation cache is an optimization, not a dependency — an evicted
entry is re-resolved rather than scored off something stale. Scoring SEEDS the
world (`torch.manual_seed(accuracy_seed)` per candidate) so a candidate's score
is reproducible; that reseed is confined by the pipeline step, which runs the
whole search inside `pipelining.determinism.isolated_rng_stream` — a
search chooses a configuration and must not re-roll the weights the run goes on
to deploy. Every
candidate-scoped failure comes back as a typed `CandidateFailure` that the
boundary renders once — an invalid `ValidationResult` in the validate path, a
`CandidateInfeasibleError` (or a scored penalty, for packing infeasibility) in
the evaluate path — while problem-level breakage propagates untyped and aborts
the run.

**What does this candidate LOOK like?** — `candidate_layout(configuration)`
returns the `CandidateLayout` (chip, softcores, packing census, view) an
evaluation throws away, off the very same `_resolve_model`/`_resolve_layout`
path. It is the introspection seam the compilagent layout backend renders its
agent-visible payload from, so an agent cannot be shown a chip or a number the
search does not use. The backend does not RENDER those facts itself: it wraps
the returned census in `CandidateLayoutView.packed` (the packing is handed over,
never redone) and serves it through `deployment_record.introspection`'s
registry, so the payload shapes are declared and versioned and the optimizer
imports nothing from `mapping`.

### `EncodedProblem` is not a pymoo interface

`EncodedProblem` states the vector encoding in library-neutral terms —
`n_var` (dimension), `xl`/`xu` (box bounds as arrays), `decode(x) → config` —
and `SearchProblem` states the rest (`objectives` with per-axis `goal`,
`evaluate → {name: value}`, `constraint_violation ≤ 0` feasible). The pymoo
adapter lives entirely in `optimizers/nsga2_optimizer.py`. Porting to another
MOO library is therefore an adapter, not a problem rewrite:

| Contract | pymoo (`nsga2_optimizer`) | A non-pymoo host, e.g. Optuna / DEAP / scipy |
|---|---|---|
| n_var | `Problem(n_var=...)` | `len(bounds)`; DEAP individual length |
| xl / xu | `Problem(xl=, xu=)` | `scipy.optimize.differential_evolution(bounds=list(zip(xl, xu)))`; `trial.suggest_float(f"x{i}", xl[i], xu[i])` |
| decode(x) | called inside `_evaluate` | called at the top of the objective callable, on the raw vector the host proposes |
| objectives | `Problem(n_obj=len(objectives))`, minimized after sign-flipping every `goal == "max"` axis | Optuna `directions=[s.goal for s in objectives]`; DEAP `creator.create("Fitness", weights=...)` |
| evaluate(config) | `out["F"]` row, in `objectives` order | the tuple/list the host's objective callable returns, same order |
| constraint_violation | `Problem(n_ieq_constr=1)`, `out["G"]` | a penalty term, or the host's own constraint hook |
| CandidateInfeasibleError | caught per candidate → `invalid_penalty` row + warning | same: catch at the objective callable, return the host's worst-value row |

The integer/grid nature of the hardware variables lives in `decode`, not in the
encoding: every host may propose continuous vectors inside `[xl, xu]`.

## The live search channel

Every backend reports progress on ONE stream: `reporter("search_event", json)`,
built by `optimizers/search_events.py` and rendered by
`gui/static/js/search-live.js`, which dispatches on `event["type"]` and drops
anything it does not know — a frame invented outside those constructors is a
silently blank panel, which is exactly the state the classical optimizer was in
until it started emitting.

Frames both NSGA-II and AgentEvolve emit:

- `generation_start` — `gen`, `total_gens`, `phase`, `pop_size`, and the
  `{name, goal}` axes the panel ranks and colours candidates by.
- `candidates_generated` — `gen`, `count` (plus the LLM backends' `reasoning`).
  `count` is the batch this generation actually PRODUCED, never the configured
  budget: pin a dimension (`core_neurons_bounds: [256, 256]`) and duplicate
  elimination leaves fewer candidates than `pop_size`, which keeps its own field.
- `generation_complete` — `gen`, `valid_count`, `failed_count`, `pareto_size`,
  and the front's leading rows, INCUMBENT FIRST (`order_by_minimax_rank`): the
  preview is a truncation, never a ranking of its own. The two counts are that
  batch's own verdicts (feasible / infeasible-or-scored-penalty), so a search
  that is dying reads as dying.
- `search_complete` — the run totals and the size of the front handed over.

Generations are 1-BASED everywhere one `SearchResult` speaks about them: the
frames' `gen`, each candidate's `metadata["generation"]`, and the `history`
rows the search report plots — the classical backend's history used to
enumerate from 0 while its own candidate tags started at 1, so a single
generation answered to two ordinals in one artifact.

The LLM backends additionally stream per-candidate and per-call detail
(`candidate_result`, `batch_summary`, `llm_trace`, `compilagent_*`).

The classical optimizer emits at GENERATION granularity only: its candidates are
cheap and numerous, so a per-candidate stream would be a thousand-frame flood
that costs more than the silence it replaces. Emission is telemetry and rides
`best_effort` — a dead monitor never takes a search down.

## Dependencies
- `deployment_record` — the objectives registry (`OBJECTIVES`, `ObjectiveSpecV2`, `CandidateStaticView`, `candidate_probe_without`) that `results.py` projects and resolves through, the `chip_param_capacity`/`declared_core_capacity` formulas the joint layout hook computes candidate chip capacity with, and the introspection registry (`INTROSPECTION_REGISTRY`, `CandidateLayoutView`) the compilagent backend serves every agent-visible payload from — fed from the problem's own `candidate_layout` census via `CandidateLayoutView.packed`, so the agent surface is a projection of the scored layout rather than a second one. One-way: `deployment_record` never imports `search`.
- `mapping` — layout types (`LayoutSoftCoreSpec`, `LayoutHardCoreType`), `LayoutVerificationStats`, `ChipCapabilities`, `compute_mapping_stats`, and platform mapping params, used by the joint problem's layout hook/validation. Coalescing-config normalization is NOT among them: a candidate platform is normalized once, by the deployment resolver the problem is handed. `optimizers/compilagent` imports none of it (AST-pinned): it reaches the same facts through `deployment_record.introspection`.
- `data_handling` — `DataProviderFactory` / `DataLoaderFactory` powering the evaluators' train/validate loops.
- `torch_mapping` — `convert_torch_model` to lower candidate models into layout IR inside the joint problem's layout hook.
- `common` — `best_effort` wrappers for non-fatal reporting/trace paths in optimizers.
- `gui` — `to_json_safe` (guarded import) for JSON-safe payloads in compilagent backend tools.

## Dependents
- `pipelining` — the architecture search step and helpers construct `JointArchHwProblem`, `SearchSpaceDescription`, and the optimizer backends, and inject the `PlatformResolver` (the deployment's own platform resolution) the problem builds every candidate chip with.
- `gui` — wizard schema imports `ALL_OBJECTIVES` / `ACCURACY_OBJECTIVE_NAME` to present objective choices.

## Exported API
`__init__.py` re-exports the core contracts:
- `SearchProblem`, `ValidationResult` — problem interface and feasibility result.
- `CandidateInfeasibleError` — typed candidate-dependent infeasibility (optimizers penalize, everything else aborts).
- `ObjectiveSpec`, `Candidate`, `SearchResult` — objective and result containers.

Optimizers, evaluators, and concrete problems are imported from their
subpackages (`search.optimizers`, `search.evaluators`, `search.problems`).
