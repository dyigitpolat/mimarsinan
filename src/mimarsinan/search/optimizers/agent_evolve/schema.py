"""LLM-facing schema and formatting helpers for AgentEvolve."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from mimarsinan.search.results import ObjectiveSpec


@dataclass
class CandidateResult:
    """Result of a candidate evaluation."""

    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None
    failure_phase: Optional[str] = None
    insight: str = ""


def prettify_configuration(config: Dict[str, Any], indent: int = 2) -> str:
    """Format a configuration dictionary for human readability."""
    return json.dumps(config, indent=indent, sort_keys=True)


def prettify_objectives(objectives: Sequence[ObjectiveSpec]) -> str:
    """Format objective specifications for LLM context."""
    lines = ["OBJECTIVES:", "=" * 60]
    for spec in objectives:
        goal_desc = "higher is better" if spec.goal == "max" else "lower is better"
        lines.append(f"  - {spec.name}: {goal_desc}")
    return "\n".join(lines)


def prettify_results(
    results: List["CandidateResult"],
    objectives: Sequence[ObjectiveSpec],
) -> str:
    """Format CandidateResult list for LLM prompts."""
    lines = []
    for i, result in enumerate(results, 1):
        lines.append(f"--- Candidate {i} ---")
        lines.append(f"Configuration: {prettify_configuration(result.configuration)}")

        if result.is_valid:
            obj_strs = []
            for spec in objectives:
                val = result.objectives.get(spec.name, 0.0)
                goal_arrow = "↑" if spec.goal == "max" else "↓"
                obj_strs.append(f"{spec.name}={val:.4f}{goal_arrow}")
            lines.append(f"Objectives: {', '.join(obj_strs)}")
        else:
            lines.append("Status: INVALID")
            if result.failure_phase:
                lines.append(f"Failure Phase: {result.failure_phase}")
            if result.error_message:
                lines.append(f"Error: {result.error_message}")

        if result.insight:
            lines.append(f"Insight: {result.insight}")
        lines.append("")

    return "\n".join(lines)


def format_performance_stats(
    stats: Dict[str, Any],
    objectives: Sequence[ObjectiveSpec],
    n_valid: int,
) -> str:
    """Format compute_performance_stats output into a comprehensive string for LLM prompts."""
    lines = []

    pareto_size = stats.get("pareto_size", 0)
    lines.append(f"TOTAL VALID CANDIDATES: {n_valid}")
    lines.append(f"PARETO FRONT SIZE: {pareto_size}")
    lines.append("")

    lines.append("BEST AND WORST PER OBJECTIVE:")
    for spec in objectives:
        best = stats.get(f"best_{spec.name}")
        worst = stats.get(f"worst_{spec.name}")
        if best:
            lines.append(f"  Best {spec.name}: {best.objectives.get(spec.name, 'N/A')}")
            lines.append(f"    Config: {prettify_configuration(best.configuration)}")
        if worst:
            lines.append(f"  Worst {spec.name}: {worst.objectives.get(spec.name, 'N/A')}")
            lines.append(f"    Config: {prettify_configuration(worst.configuration)}")
    lines.append("")

    top_pareto = stats.get("top_3_pareto", [])
    if top_pareto:
        lines.append("TOP PARETO CONFIGURATIONS (ranked by minimax-rank, best-balanced first):")
        lines.append(prettify_results(top_pareto, objectives))
    else:
        lines.append("TOP PARETO CONFIGURATIONS: None")
    lines.append("")

    bottom = stats.get("bottom_3_dominated", [])
    if bottom:
        lines.append("WORST NON-PARETO CONFIGURATIONS (most dominated):")
        lines.append(prettify_results(bottom, objectives))
    else:
        lines.append("WORST NON-PARETO CONFIGURATIONS: None")

    return "\n".join(lines)


def format_search_space_description(
    objectives: Sequence[ObjectiveSpec],
    config_schema: Optional[Dict[str, Any]] = None,
    example_config: Optional[Dict[str, Any]] = None,
    constraints: Optional[str] = None,
) -> str:
    """Generate a comprehensive search space description for the LLM."""
    lines = []

    lines.append("=" * 70)
    lines.append("MULTI-OBJECTIVE OPTIMIZATION PROBLEM")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVES:")
    for spec in objectives:
        goal_desc = "MAXIMIZE (higher is better)" if spec.goal == "max" else "MINIMIZE (lower is better)"
        lines.append(f"  • {spec.name}: {goal_desc}")
    lines.append("")

    if config_schema:
        lines.append("CONFIGURATION SCHEMA:")
        lines.append(prettify_configuration(config_schema))
        lines.append("")

    if example_config:
        lines.append("EXAMPLE CONFIGURATION:")
        lines.append(prettify_configuration(example_config))
        lines.append("")

    if constraints:
        lines.append("CONSTRAINTS:")
        lines.append(constraints)
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "CandidateResult",
    "format_performance_stats",
    "format_search_space_description",
    "prettify_configuration",
    "prettify_objectives",
    "prettify_results",
]
