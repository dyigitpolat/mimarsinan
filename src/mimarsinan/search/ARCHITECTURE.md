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
| `results.py` | The legacy PROJECTION of `deployment_record.objectives` — `ALL_OBJECTIVES` is that registry's search catalogue rendered as the frozen `ObjectiveSpec(name, goal)` tuple every optimizer indexes (byte-equality pinned by test), `objectives_for_mode`/`resolve_active_objectives` delegate to the registry and now FAIL LOUD on an unknown or mode-unavailable objective (the silent drop is gone); the per-search-mode DEFAULTS stay here (a search policy, not a record fact), together with the `Candidate`/`SearchResult` containers and minimax-rank best-candidate selection |
| `search_space_description.py` | `SearchSpaceDescription` SSOT for the joint NAS + HW space (`CORE_DIM_GRANULARITY`), with renderers to AgentEvolve prompt schema/example/constraints and compilagent levers |
| `search_space_compilagent.py` | Renders a `SearchSpaceDescription` into compilagent `Lever` tuples and derives sampled integer candidates per HW dimension |
| `patch_borders.py` | `get_region_borders`: standalone patch-region border computation utility (no in-repo callers) |
| `evaluators/` | Fast NAS accuracy evaluators: one-epoch `FastAccuracyEvaluator` and `ExtrapolatingAccuracyEvaluator` with parametric learning-curve fitting |
| `optimizers/` | `SearchOptimizer` interface and backends: pymoo NSGA-II, AgentEvolve LLM evolution, compilagent session (with `MimarsinanLayoutBackend`), shared LLM trace utilities. The compilagent introspection surface is DERIVED from `deployment_record.introspection`'s registry: one read-only tool per payload the candidate view can answer, each response carrying its `payload`/`payload_version`, so registering a payload reaches the agent without a hand-written tool. These modules import the introspection types and NOTHING from `mapping` (AST-pinned) |
| `problems/` | Concrete problems: `EncodedProblem` (vector-encoded) protocol and `JointArchHwProblem` for joint architecture + hardware co-search |

## Dependencies
- `deployment_record` — the objectives registry (`OBJECTIVES`, `ObjectiveSpecV2`, `CandidateStaticView`) that `results.py` projects and resolves through, the `chip_param_capacity` formula the joint layout hook computes candidate chip capacity with, and the introspection registry (`INTROSPECTION_REGISTRY`, `CandidateLayoutView`) the compilagent backend serves every agent-visible payload from — including the layout answer itself, so the capability declaration is forwarded whole in ONE place. One-way: `deployment_record` never imports `search`.
- `mapping` — layout types (`LayoutSoftCoreSpec`, `LayoutHardCoreType`), `ChipCapabilities`, `compute_mapping_stats`, coalescing-config normalization, and platform mapping params, used by the joint problem's layout hook/validation. `optimizers/compilagent` does NOT: it reaches the same facts through `deployment_record.introspection`.
- `data_handling` — `DataProviderFactory` / `DataLoaderFactory` powering the evaluators' train/validate loops.
- `torch_mapping` — `convert_torch_model` to lower candidate models into layout IR inside the joint problem's layout hook.
- `common` — `best_effort` wrappers for non-fatal reporting/trace paths in optimizers.
- `gui` — `to_json_safe` (guarded import) for JSON-safe payloads in compilagent backend tools.

## Dependents
- `pipelining` — architecture search step and helpers construct `JointArchHwProblem`, `SearchSpaceDescription`, and the optimizer backends.
- `gui` — wizard schema imports `ALL_OBJECTIVES` / `ACCURACY_OBJECTIVE_NAME` to present objective choices.

## Exported API
`__init__.py` re-exports the core contracts:
- `SearchProblem`, `ValidationResult` — problem interface and feasibility result.
- `CandidateInfeasibleError` — typed candidate-dependent infeasibility (optimizers penalize, everything else aborts).
- `ObjectiveSpec`, `Candidate`, `SearchResult` — objective and result containers.

Optimizers, evaluators, and concrete problems are imported from their
subpackages (`search.optimizers`, `search.evaluators`, `search.problems`).
