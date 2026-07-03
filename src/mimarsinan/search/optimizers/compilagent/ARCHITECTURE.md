# search/optimizers/compilagent/ — Compilagent Backend + Optimizer

Adapter that exposes mimarsinan's joint NAS + hardware search to
[compilagent](https://github.com/cursor/compilagent)'s `OptimizationSession`
loop and presents the result back as a third `SearchOptimizer` option
alongside `NSGA2Optimizer` and `AgentEvolveOptimizer`.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `__init__.py` | `CompilagentOptimizer`, `MimarsinanLayoutBackend`, `register_backend`, `BACKEND_ID` | Public surface; importing the package registers the backend. |
| `entrypoint.py` | `register_backend`, `BACKEND_ID = "mimarsinan_layout"` | Idempotent backend registration. |
| `backend/` | `MimarsinanLayoutBackend`, layout/eval/tools/validate helpers | Core backend orchestration and layout payload collection. |
| `sink/` | `MultiObjectiveSink`, `CandidateRecord`, `observe_event` | In-memory candidate stream + live reporter hooks via `search.optimizers.llm.trace.emit_search_event`. |
| `workload.py` | `register_problem`, `lookup_problem`, `build_workload_spec` | Per-process `workload_id -> JointArchHwProblem` registry. |
| `plan_codec.py` | `encode_plan`, `decode_plan` | Plan ↔ `{model_config, platform_constraints}` translation. |
| `lever_factory.py` | `levers_from_description` | `SearchSpaceDescription` → `tuple[Lever, ...]`. |
| `tools.py` | `build_introspection_tools` | Read-only backend introspection tools. |
| `guided_toolset.py` | `GuidedToolset` | Wraps session toolset with guidance injection. |
| `guidance_blocks.py` | `format_guidance_block`, … | `[GUIDANCE]` / `[BASELINE FOOTPRINT]` text builders. |
| `compilagent_optimizer.py` | `CompilagentOptimizer(SearchOptimizer)` | Session orchestration. |
| `optimizer_result.py` | `build_search_result`, `build_workload_instance` | Drain sink → `SearchResult` via `agent_evolve.codec`. |

## Dependencies

- **Internal**: `mimarsinan.search.problems.joint`, `mimarsinan.search.results`, `mimarsinan.search.search_space_description`, `mimarsinan.search.optimizers.agent_evolve.codec`, `mimarsinan.search.optimizers.llm.trace`, `mimarsinan.mapping.layout`, `mimarsinan.mapping.verification.layout_verification_stats`, `mimarsinan.common.best_effort` (pass-event delivery to the compilagent host is best-effort; compile/analyze fallbacks log a warning).
- **External**: `compilagent>=0.2.0`.

## Exported API (`__init__.py`)

`CompilagentOptimizer`, `MimarsinanLayoutBackend`, `register_backend`, `BACKEND_ID`.
