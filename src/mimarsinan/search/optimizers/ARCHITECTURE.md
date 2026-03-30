# search/optimizers/ -- Search Optimizer Implementations

Pluggable optimizer implementations for the search framework.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `base.py` | `SearchOptimizer` | Abstract base class for all optimizers |
| `nsga2_optimizer.py` | `NSGA2Optimizer` | Multi-objective genetic algorithm via pymoo's NSGA-II |
| `agent_evolve_optimizer.py` | `AgentEvolveOptimizer` | LLM-based optimizer using agentic evolution (optional, requires `pydantic-ai`); orchestration only, prompts in `agent_evolve_prompts`. Candidate-generation methods return `Tuple[List[ConfigT], str]` (candidates + reasoning). `_report_search_event` emits structured JSON events (`generation_start`, `candidates_generated`, `candidate_result`, `batch_summary`, `generation_complete`, `search_complete`) via the reporter for live GUI tracking. |
| `agent_evolve_prompts.py` | `build_*_prompt`, `parse_candidates` | Prompt template builders and candidate parsing for Agentic Evolution LLM calls. All 4 candidate-generation prompts instruct the LLM to provide chain-of-thought reasoning before candidates. |
| `agent_evolve_support.py` | `CandidateResult`, `compute_pareto_front`, `prettify_*`, etc. | Pareto/formatting and analysis helpers for Agentic Evolution. `CandidateResult.failure_phase` carries validation phase info for constraint learning. |
| `test_agent_evolve_optimizer.py` | (test) | Unit tests for AgentEvolveOptimizer |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.problems.encoded_problem`.
- **External**: `pymoo`, `numpy`, `pydantic-ai` (optional).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects and uses optimizers.

## Exported API (\_\_init\_\_.py)

`SearchOptimizer`, `NSGA2Optimizer`, and optionally `AgentEvolveOptimizer`.
