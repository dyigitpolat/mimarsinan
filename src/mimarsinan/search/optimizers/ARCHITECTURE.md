# search/optimizers/ -- Search Optimizer Implementations

Pluggable optimizer implementations for the search framework.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `base.py` | `SearchOptimizer` | Abstract base class for all optimizers |
| `nsga2_optimizer.py` | `NSGA2Optimizer` | Multi-objective genetic algorithm via pymoo's NSGA-II |
| `agent_evolve_optimizer.py` | `AgentEvolveOptimizer` | LLM-based optimizer using agentic evolution (optional, requires `pydantic-ai`); orchestration only, prompts in `agent_evolve_prompts`. **`optimize()`** runs the full search under a **single** `asyncio.run` so all `await self._llm_call(...)` share one event loop. **`_make_agent()`** returns a **new** `Agent` per LLM call (no caching) so async HTTP clients are not reused across teardown / provider SDK quirks. Candidate-generation methods return `Tuple[List[ConfigT], str]` (candidates + reasoning). `_coerce_llm_text` normalizes `reasoning` and other text fields when the provider returns dict/list so logging and trace previews never slice non-strings. On LLM failure, verbose mode logs the `__cause__` chain. `_report_search_event` emits structured JSON events (`generation_start`, `candidates_generated`, `candidate_result`, `batch_summary`, `generation_complete`, `search_complete`, `llm_trace`) via the reporter for live GUI tracking. Each `_llm_call` passes a `call_kind` (`initial_candidates`, `regenerate_candidates`, `offspring`, `regenerate_offspring`, `failure_insights`, `constraint_instruction`, `update_constraint`, `performance_insights`, `update_performance_insights`). `llm_trace` includes `gen`, `ordinal`, `call_kind`, `output_schema_keys`, `request` (sectioned prompt with caps), and `response` (structured summaries, not raw JSON dumps). Trace context uses `_trace_reporter` / `_trace_gen` / `_trace_seq` (reset per generation). |
| `agent_evolve_prompts.py` | `build_*_prompt`, `parse_candidates` | Prompt template builders and candidate parsing for Agentic Evolution LLM calls. All 4 candidate-generation prompts instruct the LLM to provide chain-of-thought reasoning before candidates. |
| `agent_evolve_support.py` | `CandidateResult`, `compute_pareto_front`, `prettify_*`, etc. | Pareto/formatting and analysis helpers for Agentic Evolution. `CandidateResult.failure_phase` carries validation phase info for constraint learning. **Best** selection and **generation_complete** Pareto summaries use **minimax-rank** (`select_best_candidate_minimax`, `sort_pareto_results_minimax_first`), matching `select_minimax_rank` in `search/results.py`, the live GUI, and NSGA2. Lexicographic `select_best_candidate` remains for callers that need it. |
| `test_agent_evolve_optimizer.py` | (test) | Unit tests for AgentEvolveOptimizer |

**AgentEvolve live GUI string caps:** `generation_complete` truncates `constraint_instruction` and `performance_insights` to `_TRACE_MAX_GEN_COMPLETE_STR` (10k chars each). For `llm_trace`, constraint/performance-related `call_kind` values use `_TRACE_MAX_TEXT_PREVIEW` (12k) for `response.text_preview`, with `text_preview_truncated` and `text_preview_full_len` when trimmed.

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.problems.encoded_problem`.
- **External**: `pymoo`, `numpy`, `pydantic-ai` (optional).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects and uses optimizers.

## Exported API (\_\_init\_\_.py)

`SearchOptimizer`, `NSGA2Optimizer`, and optionally `AgentEvolveOptimizer`.
