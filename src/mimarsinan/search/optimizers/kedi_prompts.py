"""Prompt templates and candidate parsing for KediOptimizer LLM calls."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

ConfigT = Dict[str, Any]


def build_initial_candidates_prompt(n_candidates: int, search_space_desc: str) -> str:
    return f"""You are an optimization expert generating candidates for a multi-objective optimization problem.

{search_space_desc}

Generate exactly {n_candidates} different configuration candidates that:
1. Are diverse and explore different regions of the search space
2. Are likely to be valid (satisfy any constraints)
3. Trade off between different objectives

Return the configurations as a list of dictionaries."""


def build_regenerate_candidates_prompt(
    failed_str: str,
    search_space_desc: str,
    constraint_instruction: str,
    performance_insights: str,
    n_candidates: int,
) -> str:
    return f"""You are an optimization expert. Previous candidates failed validation. Learn from the failures and generate better candidates.

{search_space_desc}

FAILED CANDIDATES AND THEIR ISSUES:
{failed_str}

CONSTRAINT COMPLIANCE INSTRUCTIONS:
{constraint_instruction if constraint_instruction else "No specific constraints learned yet."}

PERFORMANCE INSIGHTS:
{performance_insights if performance_insights else "No performance insights available yet."}

Generate exactly {n_candidates} NEW configuration candidates that:
1. Address the issues from the failed candidates
2. Follow the constraint instructions
3. Are likely to be valid

Return the configurations as a list of dictionaries."""


def build_offspring_prompt(
    pareto_str: str,
    search_space_desc: str,
    constraint_instruction: str,
    performance_insights: str,
    n_candidates: int,
) -> str:
    return f"""You are an optimization expert. Generate offspring candidates based on high-quality Pareto-optimal configurations.

{search_space_desc}

PARETO-OPTIMAL CONFIGURATIONS (best performers so far):
{pareto_str}

CONSTRAINT COMPLIANCE INSTRUCTIONS:
{constraint_instruction if constraint_instruction else "Follow standard constraints."}

PERFORMANCE INSIGHTS:
{performance_insights if performance_insights else "Analyze the Pareto configurations for patterns."}

Generate exactly {n_candidates} NEW configuration candidates that:
1. Build upon the patterns in the Pareto configurations
2. Explore new trade-offs between objectives
3. Follow the constraint instructions
4. Try to improve on existing solutions

Return the configurations as a list of dictionaries."""


def build_regenerate_offspring_prompt(
    failed_str: str,
    pareto_str: str,
    search_space_desc: str,
    constraint_instruction: str,
    performance_insights: str,
    n_candidates: int,
) -> str:
    return f"""You are an optimization expert. Some offspring candidates failed. Learn from the failures while using the Pareto front as guidance.

{search_space_desc}

PARETO-OPTIMAL CONFIGURATIONS (reference for valid, high-quality solutions):
{pareto_str}

FAILED OFFSPRING AND THEIR ISSUES:
{failed_str}

CONSTRAINT COMPLIANCE INSTRUCTIONS:
{constraint_instruction if constraint_instruction else "Follow the patterns from Pareto configurations."}

PERFORMANCE INSIGHTS:
{performance_insights}

Generate exactly {n_candidates} NEW configuration candidates that:
1. Address the issues from the failed candidates
2. Stay close to the Pareto configurations (which are known to be valid)
3. Follow the constraint instructions
4. Try to improve on existing solutions

Return the configurations as a list of dictionaries."""


def build_failure_insights_prompt(
    failed_str: str,
    search_space_desc: str,
    n_failed: int,
) -> str:
    return f"""You are an optimization expert. Analyze why these candidates failed and provide specific insights.

{search_space_desc}

FAILED CANDIDATES:
{failed_str}

For each failed candidate, provide a specific insight about:
1. What constraint or requirement it violated
2. How to fix it in future candidates

Return a list of exactly {n_failed} insight strings, one for each failed candidate."""


def build_constraint_instruction_prompt(failed_str: str, search_space_desc: str) -> str:
    return f"""You are an optimization expert. Based on these failed candidates, create a consolidated set of constraint instructions.

{search_space_desc}

FAILED CANDIDATES AND INSIGHTS:
{failed_str}

Create a clear, actionable set of instructions that future candidates should follow to avoid these failures. Return a detailed paragraph describing how to satisfy constraints when proposing configurations."""


def build_update_constraint_prompt(
    previous_instruction: str,
    failed_str: str,
    search_space_desc: str,
) -> str:
    return f"""You are an optimization expert. Update the constraint instructions based on new failures.

PREVIOUS CONSTRAINT INSTRUCTIONS:
{previous_instruction}

NEW FAILED CANDIDATES:
{failed_str}

Update the constraint instructions to incorporate insights from these new failures. Return an updated, comprehensive set of constraint instructions."""


def build_performance_insights_prompt(stats_str: str, search_space_desc: str) -> str:
    return f"""You are an optimization expert. Analyze the performance patterns and provide insights for generating better candidates.

{search_space_desc}

PERFORMANCE STATISTICS:
{stats_str}

Analyze:
1. What patterns make configurations perform well?
2. What trade-offs exist between objectives?
3. What configuration choices lead to good overall performance?

Return a detailed analysis of what makes configurations perform well and how to generate better candidates."""


def build_update_performance_insights_prompt(
    previous_insights: str,
    pareto_str: str,
    n_valid: int,
    pareto_size: int,
    search_space_desc: str,
) -> str:
    return f"""You are an optimization expert. Update the performance insights based on new results.

PREVIOUS INSIGHTS:
{previous_insights}

CURRENT TOP PARETO CONFIGURATIONS:
{pareto_str}

TOTAL VALID CANDIDATES: {n_valid}
PARETO FRONT SIZE: {pareto_size}

Update the insights with any new patterns or observations. Return updated performance insights incorporating the new results."""


def parse_candidates(
    candidates: Any,
    expected_count: int,
    log_fn: Callable[[str], None],
) -> List[ConfigT]:
    """Parse and validate candidate configurations from LLM output."""
    if not isinstance(candidates, list):
        log_fn(f"Warning: LLM returned non-list candidates: {type(candidates)}")
        return []

    parsed: List[ConfigT] = []
    for c in candidates:
        if isinstance(c, dict):
            parsed.append(c)
        elif isinstance(c, str):
            try:
                import json
                parsed.append(json.loads(c))
            except Exception:
                log_fn(f"Warning: Could not parse candidate string: {c[:100]}")

    if len(parsed) != expected_count:
        log_fn(f"Warning: Expected {expected_count} candidates, got {len(parsed)}")

    return parsed
