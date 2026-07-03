# search/optimizers/ -- Search Optimizer Implementations

Pluggable optimizer implementations for the search framework.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `base.py` | `SearchOptimizer` | Abstract base class for all optimizers |
| `nsga2_optimizer.py` | `NSGA2Optimizer` | Multi-objective genetic algorithm via pymoo's NSGA-II |
| `agent_evolve_optimizer.py` | `AgentEvolveOptimizer` | LLM-based optimizer using agentic evolution (optional, requires `pydantic-ai`); orchestration only, prompts in `agent_evolve_prompts`. **`optimize()`** runs the full search under a **single** `asyncio.run` so all `await self._llm_call(...)` share one event loop. **`_make_agent()`** returns a **new** `Agent` per LLM call (no caching) so async HTTP clients are not reused across teardown / provider SDK quirks. Candidate-generation methods return `Tuple[List[ConfigT], str]` (candidates + reasoning). `_coerce_llm_text` normalizes `reasoning` and other text fields when the provider returns dict/list so logging and trace previews never slice non-strings. On LLM failure, verbose mode logs the `__cause__` chain. `_report_search_event` emits structured JSON events (`generation_start`, `candidates_generated`, `candidate_result`, `batch_summary`, `generation_complete`, `search_complete`, `llm_trace`) via the reporter for live GUI tracking. Each `_llm_call` passes a `call_kind` (`initial_candidates`, `regenerate_candidates`, `offspring`, `regenerate_offspring`, `failure_insights`, `constraint_instruction`, `update_constraint`, `performance_insights`, `update_performance_insights`). `llm_trace` includes `gen`, `ordinal`, `call_kind`, `output_schema_keys`, `request` (sectioned prompt with caps), and `response` (structured summaries, not raw JSON dumps). Trace context uses `_trace_reporter` / `_trace_gen` / `_trace_seq` (reset per generation). |
| `agent_evolve_prompts.py` | `build_*_prompt`, `parse_candidates` | Prompt template builders and candidate parsing for Agentic Evolution LLM calls. All 4 candidate-generation prompts instruct the LLM to provide chain-of-thought reasoning before candidates. |
| `agent_evolve/` | `AgentEvolveOptimizer`, `schema.py`, `codec.py`, `host_contract.py` | LLM-based optimizer; `schema.py` holds LLM formatting + `CandidateResult`; `codec.py` holds Pareto/minimax selection and candidate conversions; `host_contract.py` holds `EvolveHostContract`, the annotation-only declaration of the host members the mixins (`batch_eval.py`, `prompting.py`) use through `self` (empty at runtime; kept in sync by `tests/unit/search/test_host_contracts.py`). |
| `llm/` | `trace.py`, `LLMTraceMixin`, `emit_search_event`, `parse_json_object` | Shared LLM trace payload builders and live-monitor event emission for AgentEvolve + Compilagent. `parse_json_object` extracts a JSON object from raw LLM text ({} on malformed JSON). |
| `compilagent/` | `CompilagentOptimizer`, `MimarsinanLayoutBackend`, … | Third optimizer option; backend split across `backend/`; optimizer result building in `optimizer_result.py`; sink events in `sink/`; guidance in `guidance_blocks.py`. Reuses `agent_evolve.codec` Pareto/minimax helpers. See `compilagent/ARCHITECTURE.md`. |

Manual checks for AgentEvolve live in `scripts/test_agent_evolve_optimizer.py` (support-only by default; `--full` for LLM).

**AgentEvolve live GUI string caps:** `generation_complete` truncates `constraint_instruction` and `performance_insights` to `_TRACE_MAX_GEN_COMPLETE_STR` (10k chars each). For `llm_trace`, constraint/performance-related `call_kind` values use `_TRACE_MAX_TEXT_PREVIEW` (12k) for `response.text_preview`, with `text_preview_truncated` and `text_preview_full_len` when trimmed.

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.problems.encoded_problem`, `search.search_space_description` (single source of truth for the AgentEvolve / compilagent search-space rendering), `mapping.layout` and `mapping.layout_verification_stats` (compilagent backend reuses softcores + stats), `common.best_effort` (reporter/telemetry emission is best-effort; failed candidate evaluations are penalized explicitly with a `logging` warning).
- **External**: `pymoo`, `numpy`, `pydantic-ai` (optional), `compilagent>=0.2.0` (hard dependency for the compilagent optimizer; see `requirements.txt`).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects and uses optimizers (`OptimizerType = Literal["nsga2", "agent_evolve", "compilagent"]`).

## Exported API (\_\_init\_\_.py)

`SearchOptimizer`, `NSGA2Optimizer`, and optionally `AgentEvolveOptimizer`,
`CompilagentOptimizer`, `MimarsinanLayoutBackend` (the latter two require
`compilagent` to be importable; importing the package registers the
backend with `compilagent.backend_registry`).
