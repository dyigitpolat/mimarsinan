"""
Python helper functions for the Kedi-based multi-objective optimizer.
Contains Pareto dominance, statistics, and formatting utilities.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.search.results import Candidate, ObjectiveSpec


@dataclass
class CandidateResult:
    """Result of a candidate evaluation."""
    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None
    insight: str = ""


def dominates(
    a: Dict[str, float],
    b: Dict[str, float],
    objectives: Sequence[ObjectiveSpec]
) -> bool:
    """
    Check if 'a' dominates 'b' (Pareto dominance).
    
    Args:
        a: First objective values
        b: Second objective values
        objectives: Objective specifications with goals (min/max)
        
    Returns:
        True if 'a' dominates 'b'
    """
    dominated_in_all = True
    better_in_one = False
    
    for spec in objectives:
        val_a = a.get(spec.name, 0.0)
        val_b = b.get(spec.name, 0.0)
        
        if spec.goal == "max":
            # Higher is better
            if val_a < val_b:
                dominated_in_all = False
            elif val_a > val_b:
                better_in_one = True
        else:  # min
            # Lower is better
            if val_a > val_b:
                dominated_in_all = False
            elif val_a < val_b:
                better_in_one = True
    
    return dominated_in_all and better_in_one


def compute_pareto_front(
    results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec]
) -> List[CandidateResult]:
    """
    Compute the Pareto front from a list of CandidateResults.
    
    Args:
        results: List of CandidateResult objects
        objectives: Objective specifications
        
    Returns:
        List of non-dominated CandidateResult objects
    """
    valid = [r for r in results if r.is_valid]
    if not valid:
        return []
    
    pareto: List[CandidateResult] = []
    for i, candidate in enumerate(valid):
        dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            if dominates(other.objectives, candidate.objectives, objectives):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return pareto


def compute_performance_stats(
    valid_results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec]
) -> Optional[Dict[str, Any]]:
    """
    Compute comprehensive performance statistics for all valid results.
    
    Includes:
    - Best/worst for each objective
    - Top 3 Pareto-optimal candidates
    - Bottom 3 most dominated candidates
    
    Args:
        valid_results: All valid CandidateResult objects
        objectives: Objective specifications
        
    Returns:
        Dictionary with performance statistics, or None if no valid results
    """
    if not valid_results:
        return None
    
    stats: Dict[str, Any] = {}
    
    # Best/worst for each objective
    for spec in objectives:
        key = spec.name
        if spec.goal == "max":
            best = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
            worst = min(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        else:
            best = min(valid_results, key=lambda r: r.objectives.get(key, float('inf')))
            worst = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        
        stats[f'best_{key}'] = best
        stats[f'worst_{key}'] = worst
    
    # Compute Pareto front
    pareto_front = compute_pareto_front(valid_results, objectives)
    
    # Compute domination counts for all candidates
    dominated_candidates: List[Tuple[CandidateResult, int]] = []
    for i, candidate in enumerate(valid_results):
        if candidate in pareto_front:
            continue
        domination_count = sum(
            1 for other in valid_results
            if dominates(other.objectives, candidate.objectives, objectives)
        )
        dominated_candidates.append((candidate, domination_count))
    
    # Sort Pareto by ranking-based metric (best overall balance)
    def compute_ranking_score(candidate: CandidateResult) -> float:
        total_rank = 0
        for spec in objectives:
            key = spec.name
            val = candidate.objectives.get(key, 0.0)
            if spec.goal == "max":
                rank = sum(1 for other in pareto_front if other.objectives.get(key, 0.0) > val) + 1
            else:
                rank = sum(1 for other in pareto_front if other.objectives.get(key, float('inf')) < val) + 1
            total_rank += rank
        return 1.0 / total_rank if total_rank > 0 else 0.0
    
    pareto_sorted = sorted(pareto_front, key=compute_ranking_score, reverse=True)
    stats['top_3_pareto'] = pareto_sorted[:3]
    
    # Sort dominated by domination count (most dominated = worst)
    dominated_sorted = sorted(dominated_candidates, key=lambda x: -x[1])
    stats['bottom_3_dominated'] = [c for c, _ in dominated_sorted[:3]]
    
    stats['pareto_front'] = pareto_front
    stats['pareto_size'] = len(pareto_front)
    
    return stats


def prettify_configuration(config: Dict[str, Any], indent: int = 2) -> str:
    """
    Format a configuration dictionary for human readability.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
        
    Returns:
        Formatted string representation
    """
    return json.dumps(config, indent=indent, sort_keys=True)


def prettify_objectives(objectives: Sequence[ObjectiveSpec]) -> str:
    """
    Format objective specifications for LLM context.
    
    Args:
        objectives: Sequence of ObjectiveSpec
        
    Returns:
        Human-readable string
    """
    lines = ["OBJECTIVES:", "=" * 60]
    for spec in objectives:
        goal_desc = "higher is better" if spec.goal == "max" else "lower is better"
        lines.append(f"  - {spec.name}: {goal_desc}")
    return "\n".join(lines)


def prettify_results(
    results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec]
) -> str:
    """
    Format CandidateResult list for LLM prompts.
    
    Args:
        results: List of CandidateResult objects
        objectives: Objective specifications
        
    Returns:
        Formatted string representation
    """
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
            lines.append(f"Status: INVALID")
            if result.error_message:
                lines.append(f"Error: {result.error_message}")
        
        if result.insight:
            lines.append(f"Insight: {result.insight}")
        lines.append("")
    
    return "\n".join(lines)


def sample_failed_for_constraint(
    latest_failed: List[CandidateResult],
    all_previous_failed: List[CandidateResult],
    max_examples: int
) -> List[CandidateResult]:
    """
    Sample failed examples for constraint instruction generation.
    
    Strategy:
    - Always include all latest failures
    - If latest failures exceed cap, truncate to max_examples
    - Otherwise, add random previous failures to fill up to max_examples
    
    Args:
        latest_failed: Failed candidates from the current batch
        all_previous_failed: All accumulated failed candidates
        max_examples: Maximum number of examples to return
        
    Returns:
        Sampled list of failed candidates
    """
    sampled = list(latest_failed)
    
    if len(sampled) >= max_examples:
        return sampled[:max_examples]
    
    remaining_slots = max_examples - len(sampled)
    
    # Get previous failures (excluding the latest batch)
    latest_ids = {id(r) for r in latest_failed}
    previous = [r for r in all_previous_failed if id(r) not in latest_ids]
    
    if previous and remaining_slots > 0:
        sample_size = min(remaining_slots, len(previous))
        sampled.extend(random.sample(previous, sample_size))
    
    return sampled


def candidate_to_result(candidate: Candidate[Dict[str, Any]]) -> CandidateResult:
    """Convert a Candidate to CandidateResult."""
    return CandidateResult(
        configuration=candidate.configuration,
        objectives=candidate.objectives,
        is_valid=True,
        error_message=None,
        insight=""
    )


def result_to_candidate(result: CandidateResult, metadata: Optional[Dict[str, Any]] = None) -> Candidate[Dict[str, Any]]:
    """Convert a CandidateResult to Candidate."""
    return Candidate(
        configuration=result.configuration,
        objectives=result.objectives,
        metadata=metadata or {"is_pareto": False}
    )


def select_best_candidate(
    pareto: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    priority_order: Optional[List[str]] = None
) -> Optional[CandidateResult]:
    """
    Select the best candidate from the Pareto front.
    
    Uses lexicographic ordering based on priority_order, or defaults to
    maximizing the first 'max' objective then minimizing 'min' objectives.
    
    Args:
        pareto: Pareto front candidates
        objectives: Objective specifications
        priority_order: Optional list of objective names in priority order
        
    Returns:
        Best candidate, or None if pareto is empty
    """
    if not pareto:
        return None
    
    if priority_order is None:
        # Default: max objectives first, then min objectives
        max_objs = [s for s in objectives if s.goal == "max"]
        min_objs = [s for s in objectives if s.goal == "min"]
        priority_order = [s.name for s in max_objs] + [s.name for s in min_objs]
    
    obj_map = {s.name: s for s in objectives}
    
    def sort_key(candidate: CandidateResult) -> Tuple:
        key = []
        for name in priority_order:
            spec = obj_map.get(name)
            val = candidate.objectives.get(name, 0.0)
            if spec and spec.goal == "max":
                key.append(-val)  # Negate for descending sort
            else:
                key.append(val)
        return tuple(key)
    
    return min(pareto, key=sort_key)


def format_search_space_description(
    objectives: Sequence[ObjectiveSpec],
    config_schema: Optional[Dict[str, Any]] = None,
    example_config: Optional[Dict[str, Any]] = None,
    constraints: Optional[str] = None
) -> str:
    """
    Generate a comprehensive search space description for the LLM.
    
    Args:
        objectives: Objective specifications
        config_schema: Optional schema describing configuration structure
        example_config: Optional example configuration
        constraints: Optional constraint description
        
    Returns:
        Formatted search space description
    """
    lines = []
    
    # Objectives
    lines.append("=" * 70)
    lines.append("MULTI-OBJECTIVE OPTIMIZATION PROBLEM")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVES:")
    for spec in objectives:
        goal_desc = "MAXIMIZE (higher is better)" if spec.goal == "max" else "MINIMIZE (lower is better)"
        lines.append(f"  • {spec.name}: {goal_desc}")
    lines.append("")
    
    # Configuration schema
    if config_schema:
        lines.append("CONFIGURATION SCHEMA:")
        lines.append(prettify_configuration(config_schema))
        lines.append("")
    
    # Example configuration
    if example_config:
        lines.append("EXAMPLE CONFIGURATION:")
        lines.append(prettify_configuration(example_config))
        lines.append("")
    
    # Constraints
    if constraints:
        lines.append("CONSTRAINTS:")
        lines.append(constraints)
        lines.append("")
    
    return "\n".join(lines)




