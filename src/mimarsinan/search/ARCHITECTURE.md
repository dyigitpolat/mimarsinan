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
| `results.py` | The legacy PROJECTION of `deployment_record.objectives` — `ALL_OBJECTIVES` is that registry's search catalogue rendered as the frozen `ObjectiveSpec(name, goal)` tuple every optimizer indexes (byte-equality pinned by test); `resolve_active_specs` resolves the ACTIVE registry specs (what an evaluation must READ off a view) and `objectives_for_mode`/`resolve_active_objectives` project them, all FAILING LOUD on an unknown or mode-unavailable objective (the silent drop is gone); the per-search-mode DEFAULTS stay here (a search policy, not a record fact), together with the `Candidate`/`SearchResult` containers and minimax-rank best-candidate selection |
| `search_space_description.py` | `SearchSpaceDescription` SSOT for the joint NAS + HW space (`CORE_DIM_GRANULARITY`), with renderers to AgentEvolve prompt schema/example/constraints and compilagent levers |
| `search_space_compilagent.py` | Renders a `SearchSpaceDescription` into compilagent `Lever` tuples and derives sampled integer candidates per HW dimension |
| `patch_borders.py` | `get_region_borders`: standalone patch-region border computation utility (no in-repo callers) |
| `evaluators/` | Fast NAS accuracy evaluators: one-epoch `FastAccuracyEvaluator` and `ExtrapolatingAccuracyEvaluator` with parametric learning-curve fitting |
| `optimizers/` | `SearchOptimizer` interface and backends: pymoo NSGA-II, AgentEvolve LLM evolution, compilagent session (with `MimarsinanLayoutBackend`), shared LLM trace utilities |
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
itself (`fixed_platform_constraints` == the empty overlay).

**What is this candidate worth?** — one path, `_resolve_model` → `_resolve_layout`
→ one `CandidateStaticView` → `{spec.key: spec.value(view)}` over the ACTIVE
registry specs. The view's facts (model build → layout collection → packing
census) are computed once and only when an active axis needs them:
`_requires_fragment` asks the registry — via `candidate_probe_without` — whether
any active axis goes unavailable without a fragment, so accuracy is trained
exactly where the mode carries the axis and the mapping is skipped when no axis
reads a layout. A hardware-only search reuses the candidate-INDEPENDENT model
and its mapper representation, never its layout: tiling is a function of the
candidate's core geometry, so each candidate is packed on the chip it actually
declares. The validation cache is an optimization, not a dependency — an evicted
entry is re-resolved rather than scored off something stale. Every
candidate-scoped failure comes back as a typed `CandidateFailure` that the
boundary renders once — an invalid `ValidationResult` in the validate path, a
`CandidateInfeasibleError` (or a scored penalty, for packing infeasibility) in
the evaluate path — while problem-level breakage propagates untyped and aborts
the run.

**What does this candidate LOOK like?** — `candidate_layout(configuration)`
returns the `CandidateLayout` (chip, softcores, packing census, view) an
evaluation throws away, off the very same `_resolve_model`/`_resolve_layout`
path. It is the introspection seam the compilagent layout backend renders its
per-softcore/per-layer/objective payload from, so an agent cannot be shown a
chip or a number the search does not use.

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

## Dependencies
- `deployment_record` — the objectives registry (`OBJECTIVES`, `ObjectiveSpecV2`, `CandidateStaticView`, `candidate_probe_without`) that `results.py` projects and resolves through, and the `chip_param_capacity`/`declared_core_capacity` formulas the joint layout hook computes candidate chip capacity with. One-way: `deployment_record` never imports `search`.
- `mapping` — layout types (`LayoutSoftCoreSpec`, `LayoutHardCoreType`), `LayoutVerificationStats`, `ChipCapabilities`, `compute_mapping_stats`, and platform mapping params, used by the joint problem's layout hook/validation and the compilagent layout backend. Coalescing-config normalization is NOT among them: a candidate platform is normalized once, by the deployment resolver the problem is handed.
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
