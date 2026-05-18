# search/optimizers/compilagent/ — Compilagent Backend + Optimizer

Adapter that exposes mimarsinan's joint NAS + hardware search to
[compilagent](https://github.com/cursor/compilagent)'s `OptimizationSession`
loop and presents the result back as a third `SearchOptimizer` option
alongside `NSGA2Optimizer` and `AgentEvolveOptimizer`.

The package is intentionally thin: it owns the translation between
mimarsinan's existing data structures (`JointArchHwProblem`,
`LayoutSoftCoreSpec`, `LayoutVerificationStats`, `ObjectiveSpec`) and
compilagent's protocols (`Backend`, `Plan`, `Intervention`, `Lever`,
`SearchSpace`, `WorkloadSpec`). All packing / softcore-collection /
objective computation is reused from `JointArchHwProblem` — no parallel
implementations.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `CompilagentOptimizer`, `MimarsinanLayoutBackend`, `register_backend`, `BACKEND_ID` | Public surface; importing the package registers the backend. |
| `entrypoint.py` | `register_backend`, `BACKEND_ID = "mimarsinan_layout"` | Idempotent backend registration; advertised via `compilagent.backends` entry point in `pyproject.toml`. |
| `backend.py` | `MimarsinanLayoutBackend(BackendBase)` | Compilagent backend wrapping a live `JointArchHwProblem`: `analyze` exposes per-softcore, per-layer and `LayoutVerificationStats` payloads; `compile` delegates to `validate_detailed`; `time_workload` delegates to `evaluate`; `objectives_for_candidate` reports the full multi-objective tuple. |
| `workload.py` | `register_problem`, `unregister_problem`, `lookup_problem`, `build_workload_spec` | Per-process registry mapping `workload_id -> JointArchHwProblem`; the backend reads from it; the optimizer populates / drains it. |
| `plan_codec.py` | `encode_plan`, `decode_plan` | Lossless translation between compilagent `Plan` (ordered `Intervention`s on `target_kind in {"arch", "hw.core"}`) and `{model_config, platform_constraints}` configs. |
| `lever_factory.py` | `levers_from_description` | Render `SearchSpaceDescription` to `tuple[Lever, ...]` for `Backend.derive_search_space(...)`. Re-uses the renderer on the description object. |
| `tools.py` | `build_introspection_tools` | `inspect_softcores`, `inspect_layer_breakdown`, `inspect_layout_stats`, `list_objectives` — read-only `ToolDecl`s pulling from the backend's per-candidate cache. |
| `sink.py` | `MultiObjectiveSink` | `ObservationSink` decorator that captures `OBJECTIVES_RECORDED` + `CANDIDATE_PROPOSED` events into an in-memory list the optimizer drains at the end of the run. |
| `compilagent_optimizer.py` | `CompilagentOptimizer(SearchOptimizer)` | Orchestrates `OptimizationSession` + harness + sink. Translates the resulting candidate stream to a mimarsinan `SearchResult` using `agent_evolve_support` Pareto/minimax helpers (no reimplementation). Emits the same `search_event` JSON shape `AgentEvolveOptimizer` emits so the live GUI works unchanged. |

## Information flow per candidate

```
agent.propose_candidate(plan)
   -> Backend.validate_intervention(...)        # plan_codec sanity check
   -> Backend.compile(plan, artifact_dir)       # plan_codec.decode -> JointArchHwProblem.validate_detailed
   -> Backend.time_workload(plan, ...)          # JointArchHwProblem.evaluate; primary objective on median_ms
   -> Backend.objectives_for_candidate(...)     # full {accuracy, sync_barriers, ...} dict
session emits CANDIDATE_PROPOSED, BENCHMARK_COMPLETED, OBJECTIVES_RECORDED
   -> MultiObjectiveSink stores per-candidate record
optimizer drains sink -> CandidateResult list -> compute_pareto_front + select_best_candidate_minimax
   -> SearchResult (same shape AgentEvolveOptimizer returns)
```

## Dependencies

- **Internal**: `mimarsinan.search.problem`, `mimarsinan.search.problems.joint_arch_hw_problem`, `mimarsinan.search.results`, `mimarsinan.search.search_space_description`, `mimarsinan.search.optimizers.agent_evolve_support`, `mimarsinan.mapping.layout`, `mimarsinan.mapping.layout_verification_stats`.
- **External**: `compilagent>=0.2.0` (hard dependency; declared in `requirements.txt`). The optimizer additionally pulls in any `compilagent.harnesses.*` integration the user selects (default: `pydantic_ai`).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects the optimizer when `arch_search.optimizer == "compilagent"`.

## Exported API (`__init__.py`)

`CompilagentOptimizer`, `MimarsinanLayoutBackend`, `register_backend`,
`BACKEND_ID`. Importing the package registers the backend with
`compilagent.backend_registry`.

## Conventions inherited from compilagent's integration guide

- `Intervention.target.kind` is a free string; we use `"arch"` and `"hw.core"`.
- `CompileResult.artifacts` is the only path collection compilagent reads;
  rich per-candidate metadata lives in `CompileResult.metadata` and is
  republished by the introspection tools.
- Compile failures return `CompileResult(ok=False, diagnostics=...)` —
  never raise.
- Lever ranges come exclusively from `SearchSpaceDescription` (which
  derives them from `arch_options` + `core_*_bounds`); no hand-coded ranges.
