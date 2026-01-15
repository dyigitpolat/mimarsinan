"""
Python helper functions for mapping_optimizer_2.kedi
Contains all Python-only logic for evolutionary optimization.
"""

from typing import List, Optional, Callable, Any
import random
from mapping_core import (
    MappingResult, 
    Metrics,
    MappingRecommendation,
    Architecture,
    Workload,
    Mapping,
    EvaluationOutcome,
    recommendation_to_mapping,
    evaluate_mapping,
    extract_recommendation_from_mapping,
)


def sample_failed_for_constraint(
    latest_failed: List[MappingResult],
    all_previous_failed: List[MappingResult],
    max_examples: int
) -> List[MappingResult]:
    """
    Sample failed examples for constraint instruction generation.
    
    Strategy:
    - Always include all latest failures
    - If latest failures exceed cap, truncate to max_examples
    - Otherwise, add random previous failures to fill up to max_examples
    - Ensure latest failures are NOT included in the previous failures pool
    
    Args:
        latest_failed: Failed mappings from the current batch
        all_previous_failed: All accumulated failed mappings (including latest)
        max_examples: Maximum number of examples to return
        
    Returns:
        Sampled list of failed mappings (latest + random previous)
    """
    print(f"[TRACE] sample_failed_for_constraint called with {len(latest_failed)} latest failures and {len(all_previous_failed)} total failures")
    
    # Always include all latest failures
    sampled = list(latest_failed)
    
    # If we're already at or over the limit, truncate to max_examples
    if len(sampled) >= max_examples:
        return sampled[:max_examples]
    
    # Calculate how many previous failures we can add
    remaining_slots = max_examples - len(sampled)
    
    # Get previous failures (excluding the latest batch)
    # Use object identity to ensure we exclude the exact same objects
    latest_ids = {id(mr) for mr in latest_failed}
    previous = [mr for mr in all_previous_failed if id(mr) not in latest_ids]
    
    print(f"[TRACE] sample_failed_for_constraint: {len(previous)} previous failures available, {remaining_slots} slots remaining")
    
    # Randomly sample from previous failures
    if previous and remaining_slots > 0:
        sample_size = min(remaining_slots, len(previous))
        sampled.extend(random.sample(previous, sample_size))
    
    print(f"[TRACE] sample_failed_for_constraint: returning {len(sampled)} sampled failures")
    return sampled


def prettify_results(results: List[MappingResult]) -> str:
    """
    Format MappingResult list for LLM prompts.
    
    Args:
        results: List of MappingResult objects
        
    Returns:
        Formatted string representation
    """
    print("[TRACE] prettify_results called")
    results_string = "\n\n"
    for result in results:
        results_string += f"Config: {result.recommendation.to_string()}\n"
        if isinstance(result.result, Metrics):
            results_string += f"Outcome: {result.result.to_string()}\n"
        else:
            results_string += f"Outcome: {result.result.to_string()}\n"
        results_string += f"Insights: {result.insight}\n\n"
    return results_string


def is_valid(mr: MappingResult) -> bool:
    """Check if a MappingResult has valid metrics."""
    return isinstance(mr.result, Metrics)


def report_valid_results(
    valid_batch: List[MappingResult],
    base_metrics: Optional[Metrics],
    prefix: str = "  "
) -> None:
    """
    Report metrics and deltas for valid mappings.
    
    Args:
        valid_batch: List of valid MappingResult objects
        base_metrics: Baseline metrics for comparison (optional)
        prefix: String prefix for each line
    """
    if not valid_batch:
        return
    for i, mr in enumerate(valid_batch):
        m: Metrics = mr.result  # type: ignore[assignment]
        if base_metrics is not None:
            dE = (m.energy - base_metrics.energy) / base_metrics.energy * 100.0
            dC = (m.cycles - base_metrics.cycles) / base_metrics.cycles * 100.0
            dA = (m.area - base_metrics.area) / base_metrics.area * 100.0
            delta_str = f"ΔE={dE:+.1f}% ΔC={dC:+.1f}% ΔA={dA:+.1f}%"
        else:
            delta_str = "vs base: N/A"
        print(f"{prefix}[{i+1}] {m.to_string()} ({delta_str})")


def dominates(a: Metrics, b: Metrics) -> bool:
    """
    Check if metrics 'a' dominates metrics 'b' (Pareto dominance).
    
    Args:
        a: First metrics
        b: Second metrics
        
    Returns:
        True if 'a' dominates 'b'
    """
    return (
        a.energy <= b.energy and
        a.cycles <= b.cycles and
        a.area <= b.area and
        (a.energy < b.energy or a.cycles < b.cycles or a.area < b.area)
    )


def compute_pareto_front(results: List[MappingResult]) -> List[MappingResult]:
    """
    Compute the Pareto front from a list of MappingResults.
    
    Args:
        results: List of MappingResult objects
        
    Returns:
        List of non-dominated MappingResult objects
    """
    valid = [mr for mr in results if is_valid(mr)]
    pareto: List[MappingResult] = []
    for i, mr in enumerate(valid):
        dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            if dominates(other.result, mr.result):  # type: ignore[arg-type]
                dominated = True
                break
        if not dominated:
            pareto.append(mr)
    return pareto


def compute_performance_stats(all_valid_results: List[MappingResult]) -> Optional[dict]:
    """
    Compute comprehensive performance statistics for all valid results.
    
    Includes:
    - Best/worst for each objective (energy, cycles, area)
    - Top 3 Pareto-optimal mappings (best on Pareto front)
    - Bottom 3 most dominated mappings (worst overall)
    
    Args:
        all_valid_results: All valid MappingResult objects across generations
        
    Returns:
        Dictionary with performance statistics, or None if no valid results
    """
    if not all_valid_results:
        return None
    
    # Extract metrics
    valid_with_metrics = [(mr, mr.result) for mr in all_valid_results if isinstance(mr.result, Metrics)]
    
    if not valid_with_metrics:
        return None
    
    # Best for each objective (minimize)
    best_energy = min(valid_with_metrics, key=lambda x: x[1].energy)
    best_cycles = min(valid_with_metrics, key=lambda x: x[1].cycles)
    best_area = min(valid_with_metrics, key=lambda x: x[1].area)
    
    # Worst for each objective
    worst_energy = max(valid_with_metrics, key=lambda x: x[1].energy)
    worst_cycles = max(valid_with_metrics, key=lambda x: x[1].cycles)
    worst_area = max(valid_with_metrics, key=lambda x: x[1].area)
    
    # Compute Pareto front and dominated candidates
    pareto_front = []
    dominated_candidates = []
    
    for i, (mr_i, m_i) in enumerate(valid_with_metrics):
        dominated = False
        domination_count = 0  # Count how many candidates dominate this one
        
        for j, (mr_j, m_j) in enumerate(valid_with_metrics):
            if i == j:
                continue
            if dominates(m_j, m_i):
                dominated = True
                domination_count += 1
        
        if not dominated:
            pareto_front.append((mr_i, m_i))
        else:
            dominated_candidates.append((mr_i, m_i, domination_count))
    
    # Sort Pareto by ranking-based metric (best overall balance)
    # Rank each Pareto candidate by energy, cycles, and area separately
    # Then use inverse sum of ranks as the sorting key
    def compute_ranking_score(mr: MappingResult, m: Metrics) -> float:
        # Rank by energy (lower is better)
        energy_rank = sum(1 for _, other_m in pareto_front if other_m.energy < m.energy) + 1
        # Rank by cycles (lower is better)
        cycles_rank = sum(1 for _, other_m in pareto_front if other_m.cycles < m.cycles) + 1
        # Rank by area (lower is better)
        area_rank = sum(1 for _, other_m in pareto_front if other_m.area < m.area) + 1
        
        # Lower sum of ranks is better (best is rank 1+1+1=3)
        # Use inverse so we can sort ascending
        return 1.0 / (energy_rank + cycles_rank + area_rank)
    
    pareto_sorted = sorted(pareto_front, key=lambda x: compute_ranking_score(x[0], x[1]), reverse=True)
    top_3_pareto = pareto_sorted[:3]
    
    # Sort dominated candidates by domination count (most dominated = worst)
    # Break ties by normalized sum of objectives
    def normalized_sum(mr: MappingResult, m: Metrics) -> float:
        all_metrics = [x[1] for x in valid_with_metrics]
        min_e = min(x.energy for x in all_metrics)
        max_e = max(x.energy for x in all_metrics)
        min_c = min(x.cycles for x in all_metrics)
        max_c = max(x.cycles for x in all_metrics)
        min_a = min(x.area for x in all_metrics)
        max_a = max(x.area for x in all_metrics)
        
        norm_e = (m.energy - min_e) / (max_e - min_e + 1e-9)
        norm_c = (m.cycles - min_c) / (max_c - min_c + 1e-9)
        norm_a = (m.area - min_a) / (max_a - min_a + 1e-9)
        
        return norm_e + norm_c + norm_a
    
    # Sort by domination count (descending), then by normalized sum (descending)
    dominated_sorted = sorted(
        dominated_candidates,
        key=lambda x: (-x[2], -normalized_sum(x[0], x[1]))
    )
    
    # Get bottom 3 most dominated (worst performers)
    bottom_3_dominated = [(mr, m) for mr, m, _ in dominated_sorted[:3]]
    
    best_energy_mr, best_energy_m = best_energy
    best_cycles_mr, best_cycles_m = best_cycles
    best_area_mr, best_area_m = best_area
    worst_energy_mr, worst_energy_m = worst_energy
    worst_cycles_mr, worst_cycles_m = worst_cycles
    worst_area_mr, worst_area_m = worst_area
    
    return {
        'best_energy_mr': best_energy_mr,
        'best_energy_m': best_energy_m,
        'best_cycles_mr': best_cycles_mr,
        'best_cycles_m': best_cycles_m,
        'best_area_mr': best_area_mr,
        'best_area_m': best_area_m,
        'worst_energy_mr': worst_energy_mr,
        'worst_energy_m': worst_energy_m,
        'worst_cycles_mr': worst_cycles_mr,
        'worst_cycles_m': worst_cycles_m,
        'worst_area_mr': worst_area_mr,
        'worst_area_m': worst_area_m,
        'top_3_pareto': top_3_pareto,
        'bottom_3_dominated': bottom_3_dominated
    }


def evolve_mappings_impl(
    architecture: Architecture,
    workload: Workload,
    base_mapping: Mapping,
    base_result: EvaluationOutcome,
    population_size: int,
    num_generations: int,
    candidates_per_batch: int,
    max_failed_examples_for_constraint: int,
    # Kedi procedure callbacks
    generate_initial_mappings_fn: Callable[[Architecture, Workload, Mapping, int], List[MappingRecommendation]],
    regenerate_mappings_fn: Callable[[List[MappingResult], Architecture, Workload, Mapping, int, str, str], List[MappingRecommendation]],
    generate_failure_insights_batch_fn: Callable[[List[MappingResult], Architecture, Workload], List[str]],
    generate_constraint_instruction_fn: Callable[[List[MappingResult], Architecture, Workload], str],
    update_constraint_instruction_fn: Callable[[str, List[MappingResult], Architecture, Workload], str],
    generate_performance_insights_fn: Callable[[List[MappingResult], Architecture, Workload], str],
    update_performance_insights_fn: Callable[[str, List[MappingResult], Architecture, Workload], str],
    generate_offspring_mappings_fn: Callable[[Architecture, Workload, Mapping, List[MappingResult], str, str, int], List[MappingRecommendation]],
    regenerate_offspring_mappings_fn: Callable[[List[MappingResult], Architecture, Workload, Mapping, List[MappingResult], int, str, str], List[MappingRecommendation]],
) -> List[MappingResult]:
    """
    Core evolutionary loop implementing steps 1-9.
    
    This function orchestrates the evolutionary optimization by:
    1. Managing the evolutionary loop structure
    2. Evaluating mappings
    3. Calling Kedi procedures (via callbacks) for LLM interactions
    4. Computing Pareto fronts and statistics
    
    Args:
        architecture: Target hardware architecture
        workload: Workload to optimize for
        base_mapping: Baseline mapping
        base_result: Evaluation result for baseline
        population_size: Number of candidates per generation
        num_generations: Number of generations to evolve
        candidates_per_batch: Number of candidates to generate per batch
        max_failed_examples_for_constraint: Max failed examples for constraint instruction
        *_fn: Kedi procedure callbacks for LLM interactions
        
    Returns:
        Final Pareto front across all generations
    """
    print(f"[TRACE] evolve_mappings_impl called with population_size={population_size}, num_generations={num_generations}, candidates_per_batch={candidates_per_batch}")
    
    # Track seen recommendations to avoid duplicates
    seen_recommendations: set[str] = set()
    
    # Add base mapping to seen set
    try:
        base_rec = extract_recommendation_from_mapping(base_mapping)
        seen_recommendations.add(base_rec.to_string())
    except Exception as e:
        print(f"[WARN] Could not extract recommendation from base mapping: {e}")

    def filter_new_recommendations(recs: List[MappingRecommendation]) -> List[MappingRecommendation]:
        """Filter out recommendations that have already been seen."""
        new_recs = []
        for r in recs:
            s = r.to_string()
            if s not in seen_recommendations:
                seen_recommendations.add(s)
                new_recs.append(r)
        
        n_rejected = len(recs) - len(new_recs)
        if n_rejected > 0:
            print(f"  [INFO] Rejected {n_rejected} duplicate recommendations.")
        return new_recs

    all_valid_results: List[MappingResult] = []
    constraint_instruction = ""
    performance_insights = ""
    
    # Track valid count when constraint instruction was last updated
    constraint_valid_count = 0
    
    base_metrics = base_result if isinstance(base_result, Metrics) else None
    
    # ===== Generation 1: initial sampling (Steps 1–5) =====
    print(f"=== Generation 1 / {num_generations} (initial sampling) ===")
    population_results: List[MappingResult] = []
    all_failed_results: List[MappingResult] = []
    
    max_regen_rounds = 10
    regen_round = 0
    last_round_failed: List[MappingResult] = []
    
    while (
        len([mr for mr in population_results if is_valid(mr)]) < population_size
        and regen_round < max_regen_rounds
    ):
        if regen_round == 0:
            # Step 1: ask LLM to generate candidates_per_batch candidates
            recommendations = generate_initial_mappings_fn(architecture, workload, base_mapping, candidates_per_batch)
            recommendations = filter_new_recommendations(recommendations)
        else:
            # Step 2: regenerate based on last round's failures, using the known-valid base mapping as a reference.
            recommendations = regenerate_mappings_fn(
                last_round_failed,
                architecture,
                workload,
                base_mapping,
                candidates_per_batch,
                constraint_instruction,
                performance_insights,
            )
            recommendations = filter_new_recommendations(recommendations)
        
        # Evaluate batch
        eval_results: List[MappingResult] = []
        for rec in recommendations:
            mapping = recommendation_to_mapping(rec, workload, architecture)
            result = evaluate_mapping(workload, architecture, mapping)
            eval_results.append(MappingResult(recommendation=rec, result=result, insight=""))
        
        valid_batch = [mr for mr in eval_results if is_valid(mr)]
        failed_batch = [mr for mr in eval_results if not is_valid(mr)]
        print(f"\n  <<< Batch evaluation: {len(valid_batch)} valid, {len(failed_batch)} failed >>>")
        
        # Report valid results with deltas
        if valid_batch:
            report_valid_results(valid_batch, base_metrics, "    ")
        
        print("\n")
        
        if failed_batch:
            batch_insights = generate_failure_insights_batch_fn(failed_batch, architecture, workload)
            for mr, insight in zip(failed_batch, batch_insights):
                mr.insight = insight
            
            # Consolidate insights into constraint instruction
            all_failed_results.extend(failed_batch)
        
        for mr in valid_batch:
            population_results.append(mr)
            all_valid_results.append(mr)
        
        # Keep only this round's failures for next regeneration
        last_round_failed = failed_batch
        
        regen_round += 1
        cumulative_valid_count = len([mr for mr in population_results if is_valid(mr)])
        batch_valid_count = len(valid_batch)
        print(f"  Collected {cumulative_valid_count}/{population_size} valid mappings so far (total failed so far: {len(all_failed_results)})")
        
        # Update constraint instruction if: (1) it doesn't exist yet, OR (2) this batch has more valid candidates than the previous batch that triggered an update
        if all_failed_results:  # Only if we have failures to learn from
            if not constraint_instruction:
                # Always create if it doesn't exist
                sampled_failures = sample_failed_for_constraint(last_round_failed, all_failed_results, max_failed_examples_for_constraint)
                constraint_instruction = generate_constraint_instruction_fn(sampled_failures, architecture, workload)
                constraint_valid_count = batch_valid_count
                print(f"  [INFO] Initial constraint instruction created (batch valid count: {constraint_valid_count})")
            elif batch_valid_count > constraint_valid_count:
                # Update if this batch has more valid candidates than the batch that last triggered an update
                sampled_failures = sample_failed_for_constraint(last_round_failed, all_failed_results, max_failed_examples_for_constraint)
                new_constraint_instruction = update_constraint_instruction_fn(
                    constraint_instruction,
                    sampled_failures,
                    architecture,
                    workload,
                )
                if new_constraint_instruction != constraint_instruction:
                    print(f"  [INFO] Constraint instruction updated (batch valid: {constraint_valid_count} -> {batch_valid_count})")
                    constraint_instruction = new_constraint_instruction
                    constraint_valid_count = batch_valid_count
        if cumulative_valid_count >= population_size:
            break
    
    # Keep exactly population_size valid mappings if possible
    population_results = [mr for mr in population_results if is_valid(mr)][:population_size]
    
    # Print final constraint-compliance instruction for generation 1
    if constraint_instruction:
        print("Constraint-compliance instruction:")
        print(constraint_instruction)
        print()
    
    # Step 4: Pareto front for this generation
    pareto = compute_pareto_front(population_results)
    print(f"Generation 1: {len(population_results)} valid, Pareto size={len(pareto)}")
    
    # Step 5: initial performance insights (using all valid results across all generations so far)
    performance_insights = generate_performance_insights_fn(
        all_valid_results,
        architecture,
        workload,
    )
    print("Performance insights:")
    print(performance_insights)
    print()
    
    prev_pareto = pareto
    
    # ===== Generations 2..num_generations: repeat 6–9 =====
    for gen in range(2, num_generations + 1):
        print(f"=== Generation {gen} / {num_generations} ===")
        population_results = []
        gen_all_failed: List[MappingResult] = []
        
        # Pick a random example from top 3 Pareto mappings
        top_3_count = min(3, len(prev_pareto))
        example_pareto_mr = random.choice(prev_pareto[:top_3_count])
        example_pareto_mapping = recommendation_to_mapping(
            example_pareto_mr.recommendation,
            workload,
            architecture
        )
        
        # Step 6: generate new candidates using Pareto + insights
        recommendations = generate_offspring_mappings_fn(
            architecture,
            workload,
            example_pareto_mapping,
            prev_pareto,
            performance_insights,
            constraint_instruction,
            population_size,
        )
        recommendations = filter_new_recommendations(recommendations)
        
        # Evaluate initial offspring
        offspring_results: List[MappingResult] = []
        for rec in recommendations:
            mapping = recommendation_to_mapping(rec, workload, architecture)
            result = evaluate_mapping(workload, architecture, mapping)
            offspring_results.append(MappingResult(recommendation=rec, result=result, insight=""))
        
        valid_batch = [mr for mr in offspring_results if is_valid(mr)]
        failed_batch = [mr for mr in offspring_results if not is_valid(mr)]
        print(f"\n  <<< Offspring evaluation: {len(valid_batch)} valid, {len(failed_batch)} failed >>>")
        
        # Report valid results with deltas
        if valid_batch:
            report_valid_results(valid_batch, base_metrics, "    ")
        
        print("\n")
        
        if failed_batch:
            batch_insights = generate_failure_insights_batch_fn(failed_batch, architecture, workload)
            for mr, insight in zip(failed_batch, batch_insights):
                mr.insight = insight
            
            # Consolidate insights into constraint instruction
            gen_all_failed.extend(failed_batch)
        
        for mr in valid_batch:
            population_results.append(mr)
            all_valid_results.append(mr)
        
        # Keep only this batch's failures for next regeneration
        last_round_failed = failed_batch
        
        # Track valid count for this batch (not cumulative)
        batch_valid_count = len(valid_batch)
        
        # Step 7: if fewer than N valid, regenerate using last round's failures
        regen_round = 0
        tried_shuffle = False  # Track if we've tried shuffling insights first
        
        while (
            len([mr for mr in population_results if is_valid(mr)]) < population_size
            and regen_round < 10
        ):
            if not last_round_failed:
                break
            
            # Pick a random example from top 3 Pareto mappings for regeneration
            example_pareto_mr = random.choice(prev_pareto[:top_3_count])
            example_pareto_mapping = recommendation_to_mapping(
                example_pareto_mr.recommendation,
                workload,
                architecture
            )
            
            # OPTIMIZATION 1: First try shuffling existing insights before generating new ones
            if regen_round == 0 and not tried_shuffle and len(last_round_failed) > 1:
                print("  [FAST] Trying shuffled insights first...")
                tried_shuffle = True
                # Shuffle the insights among failed mappings
                shuffled_failed = list(last_round_failed)
                insights = [mr.insight for mr in shuffled_failed]
                random.shuffle(insights)
                for mr, insight in zip(shuffled_failed, insights):
                    mr.insight = insight
                last_round_failed = shuffled_failed
            
            # Always request candidates_per_batch candidates
            extra_recs = regenerate_offspring_mappings_fn(
                last_round_failed,
                architecture,
                workload,
                example_pareto_mapping,
                prev_pareto,
                candidates_per_batch,
                constraint_instruction,
                performance_insights,
            )
            extra_recs = filter_new_recommendations(extra_recs)
            
            regen_results: List[MappingResult] = []
            for rec in extra_recs:
                mapping = recommendation_to_mapping(rec, workload, architecture)
                result = evaluate_mapping(workload, architecture, mapping)
                regen_results.append(MappingResult(recommendation=rec, result=result, insight=""))
            
            valid_batch = [mr for mr in regen_results if is_valid(mr)]
            failed_batch = [mr for mr in regen_results if not is_valid(mr)]
            print(f"\n  <<< Regen round {regen_round + 1} evaluation: {len(valid_batch)} valid, {len(failed_batch)} failed >>>")
            
            # Report valid results with deltas
            if valid_batch:
                report_valid_results(valid_batch, base_metrics, "    ")
            
            print("\n")
            
            # Only generate new insights if we didn't just shuffle (or after shuffle failed)
            if failed_batch and (regen_round > 0 or tried_shuffle):
                batch_insights = generate_failure_insights_batch_fn(failed_batch, architecture, workload)
                for mr, insight in zip(failed_batch, batch_insights):
                    mr.insight = insight
                
                # Update constraint instruction with new failures
                gen_all_failed.extend(failed_batch)
            
            for mr in valid_batch:
                population_results.append(mr)
                all_valid_results.append(mr)
            
            # Keep only this round's failures for next regeneration
            last_round_failed = failed_batch
            
            regen_round += 1
            cumulative_valid_count = len([mr for mr in population_results if is_valid(mr)])
            regen_batch_valid_count = len(valid_batch)
            batch_valid_count += regen_batch_valid_count  # Track total valid for this generation
            print(f"  Regen round {regen_round}: cumulative valid={cumulative_valid_count}, total failed this gen={len(gen_all_failed)}")
            
            # Update constraint instruction if this batch has more valid candidates than the previous batch that triggered an update
            if gen_all_failed and regen_batch_valid_count > constraint_valid_count:
                sampled_failures = sample_failed_for_constraint(last_round_failed, gen_all_failed, max_failed_examples_for_constraint)
                new_constraint_instruction = update_constraint_instruction_fn(
                    constraint_instruction,
                    sampled_failures,
                    architecture,
                    workload,
                )
                if new_constraint_instruction != constraint_instruction:
                    print(f"  [INFO] Constraint instruction updated (batch valid: {constraint_valid_count} -> {regen_batch_valid_count})")
                    constraint_instruction = new_constraint_instruction
                    constraint_valid_count = regen_batch_valid_count
        
        # Keep at most population_size valid mappings
        population_results = [mr for mr in population_results if is_valid(mr)][:population_size]
        
        # Step 8: update performance insights for this generation (using all valid results)
        pareto = compute_pareto_front(population_results)
        performance_insights = update_performance_insights_fn(
            performance_insights,
            all_valid_results,
            architecture,
            workload,
        )
        
        print(f"Generation {gen}: {len(population_results)} valid, Pareto size={len(pareto)}")
        prev_pareto = pareto
    
    # ===== Final reporting (Step 10) =====
    final_pareto = compute_pareto_front(all_valid_results) if all_valid_results else []
    print("\n=== Final Pareto Set (across all generations) ===")
    if base_metrics is not None:
        print(f"Base mapping metrics: {base_metrics.to_string()}")
    for i, mr in enumerate(final_pareto):
        m: Metrics = mr.result  # type: ignore[assignment]
        if base_metrics is not None:
            dE = (m.energy - base_metrics.energy) / base_metrics.energy * 100.0
            dC = (m.cycles - base_metrics.cycles) / base_metrics.cycles * 100.0
            dA = (m.area - base_metrics.area) / base_metrics.area * 100.0
            delta_str = f"ΔE={dE:+.1f}% ΔC={dC:+.1f}% ΔA={dA:+.1f}%"
        else:
            delta_str = "vs base: N/A"
        print(f"[{i+1}] {m.to_string()} ({delta_str}) :: {mr.recommendation.to_string()}")
    
    return final_pareto


def export_pareto_to_json(pareto_results: List[MappingResult], output_path: str) -> None:
    """Export Pareto front results to JSON format."""
    import json
    
    data = []
    for mr in pareto_results:
        m: Metrics = mr.result  # type: ignore[assignment]
        rec = mr.recommendation
        entry = {
            "mapping": {
                "pe_weight_regs_temporal_factors": rec.pe_weight_regs_temporal_factors,
                "pe_weight_regs_temporal_permutation": rec.pe_weight_regs_temporal_permutation,
                "pe_accu_buffer_spatial_factors": rec.pe_accu_buffer_spatial_factors,
                "pe_accu_buffer_spatial_permutation": rec.pe_accu_buffer_spatial_permutation,
                "pe_accu_buffer_spatial_split": rec.pe_accu_buffer_spatial_split,
                "pe_accu_buffer_temporal_factors": rec.pe_accu_buffer_temporal_factors,
                "pe_accu_buffer_temporal_permutation": rec.pe_accu_buffer_temporal_permutation,
                "pe_weight_buffer_temporal_factors": rec.pe_weight_buffer_temporal_factors,
                "pe_weight_buffer_temporal_permutation": rec.pe_weight_buffer_temporal_permutation,
                "pe_input_buffer_spatial_factors": rec.pe_input_buffer_spatial_factors,
                "pe_input_buffer_spatial_permutation": rec.pe_input_buffer_spatial_permutation,
                "pe_input_buffer_spatial_split": rec.pe_input_buffer_spatial_split,
                "pe_input_buffer_temporal_factors": rec.pe_input_buffer_temporal_factors,
                "pe_input_buffer_temporal_permutation": rec.pe_input_buffer_temporal_permutation,
                "global_buffer_spatial_factors": rec.global_buffer_spatial_factors,
                "global_buffer_spatial_permutation": rec.global_buffer_spatial_permutation,
                "global_buffer_spatial_split": rec.global_buffer_spatial_split,
                "global_buffer_temporal_factors": rec.global_buffer_temporal_factors,
                "global_buffer_temporal_permutation": rec.global_buffer_temporal_permutation,
                "dram_temporal_factors": rec.dram_temporal_factors,
                "dram_temporal_permutation": rec.dram_temporal_permutation
            },
            "metrics": {
                "energy": m.energy,
                "cycles": m.cycles,
                "area": m.area
            }
        }
        data.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[INFO] Exported {len(data)} Pareto optimal mappings to {output_path}")

