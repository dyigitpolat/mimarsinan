"""
Kedi-based multi-objective optimizer using LLM for candidate generation.

This optimizer uses agentic reasoning to explore the search space,
learning from failures and performance patterns to guide the search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.kedi_optimizer_support import (
    CandidateResult,
    compute_pareto_front,
    compute_performance_stats,
    dominates,
    format_search_space_description,
    prettify_configuration,
    prettify_objectives,
    prettify_results,
    result_to_candidate,
    sample_failed_for_constraint,
    select_best_candidate,
)
from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult


# Type alias for configuration
ConfigT = Dict[str, Any]


@dataclass
class KediOptimizer(SearchOptimizer[ConfigT]):
    """
    LLM-based multi-objective optimizer using Kedi DSL.
    
    This optimizer uses an agentic approach where the LLM:
    1. Generates initial candidate configurations
    2. Learns from validation/evaluation failures via insights
    3. Consolidates constraint knowledge over generations
    4. Uses performance analysis to guide search direction
    5. Generates offspring from Pareto front patterns
    """
    
    # Search hyperparameters
    pop_size: int = 8
    generations: int = 5
    candidates_per_batch: int = 5
    max_regen_rounds: int = 10
    max_failed_examples: int = 5
    
    # LLM configuration
    model: str = "openai:gpt-4o"
    adapter_type: str = "pydantic"  # "pydantic" or "dspy"
    llm_retries: int = 3  # Retries for LLM output validation
    
    # Problem description (optional, for better LLM context)
    config_schema: Optional[Dict[str, Any]] = None
    example_config: Optional[ConfigT] = None
    constraints_description: Optional[str] = None
    
    # Verbosity
    verbose: bool = True
    
    # Penalty for invalid candidates
    invalid_penalty: float = 1e18
    
    # Internal state (not part of dataclass init)
    _adapter: Any = field(default=None, init=False, repr=False)
    _runtime: Any = field(default=None, init=False, repr=False)
    _adapter_initialized: bool = field(default=False, init=False, repr=False)
    
    def _setup_adapter(self) -> None:
        """Setup the Kedi adapter for LLM interactions (lazy initialization)."""
        if self._adapter_initialized:
            return
        
        try:
            if self.adapter_type == "dspy":
                from kedi.agent_adapter.adapters import DSPyAdapter
                self._adapter = DSPyAdapter(model=self.model)
            else:
                from kedi.agent_adapter.adapters import PydanticAdapter
                self._adapter = PydanticAdapter(
                    model=self.model,
                    retries=self.llm_retries,
                )
            self._adapter_initialized = True
        except ImportError as e:
            raise ImportError(
                f"Kedi package is required for KediOptimizer. "
                f"Install it from the kedi directory. Error: {e}"
            )
    
    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)
    
    def _llm_call(
        self,
        template: str,
        output_schema: Dict[str, type],
    ) -> Any:
        """
        Make an LLM call with the given template and output schema.
        
        Args:
            template: The prompt template
            output_schema: Dictionary mapping output names to their types
            
        Returns:
            The LLM response with filled outputs
        """
        # Lazy initialization of adapter
        self._setup_adapter()
        
        return self._adapter.produce_sync(
            template=template,
            output_schema=output_schema,
        )
    
    def optimize(self, problem: SearchProblem[ConfigT]) -> SearchResult[ConfigT]:
        """
        Run the LLM-based multi-objective optimization.
        
        Args:
            problem: The search problem to optimize
            
        Returns:
            SearchResult containing Pareto front and all evaluated candidates
        """
        objectives = list(problem.objectives)
        if not objectives:
            raise ValueError("SearchProblem.objectives must not be empty")
        
        # Build search space description for LLM
        search_space_desc = format_search_space_description(
            objectives=objectives,
            config_schema=self.config_schema,
            example_config=self.example_config,
            constraints=self.constraints_description,
        )
        
        # Track all results
        all_valid_results: List[CandidateResult] = []
        all_failed_results: List[CandidateResult] = []
        all_candidates: List[Candidate[ConfigT]] = []
        history: List[Dict[str, Any]] = []
        
        # Accumulated instructions from learning
        constraint_instruction = ""
        performance_insights = ""
        
        # ===== Generation 1: Initial Sampling =====
        self._log(f"=== Generation 1 / {self.generations} (initial sampling) ===")
        
        gen1_valid, gen1_failed, constraint_instruction = self._run_initial_generation(
            problem=problem,
            objectives=objectives,
            search_space_desc=search_space_desc,
            all_failed_results=all_failed_results,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
        )
        
        all_valid_results.extend(gen1_valid)
        all_failed_results.extend(gen1_failed)
        
        # Convert to Candidate objects
        for r in gen1_valid:
            all_candidates.append(result_to_candidate(r, {"generation": 1, "is_pareto": False}))
        for r in gen1_failed:
            all_candidates.append(result_to_candidate(r, {"generation": 1, "is_pareto": False, "valid": False}))
        
        # Compute initial Pareto front
        pareto = compute_pareto_front(all_valid_results, objectives)
        self._log(f"Generation 1: {len(gen1_valid)} valid, Pareto size={len(pareto)}")
        
        # Generate initial performance insights
        if all_valid_results:
            performance_insights = self._generate_performance_insights(
                valid_results=all_valid_results,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )
        
        history.append({
            "gen": 1,
            "valid_count": len(gen1_valid),
            "failed_count": len(gen1_failed),
            "pareto_size": len(pareto),
        })
        
        prev_pareto = pareto
        
        # ===== Generations 2..N: Evolution =====
        for gen in range(2, self.generations + 1):
            self._log(f"\n=== Generation {gen} / {self.generations} ===")
            
            gen_valid, gen_failed, constraint_instruction = self._run_evolution_generation(
                problem=problem,
                objectives=objectives,
                search_space_desc=search_space_desc,
                prev_pareto=prev_pareto,
                all_failed_results=all_failed_results,
                constraint_instruction=constraint_instruction,
                performance_insights=performance_insights,
            )
            
            all_valid_results.extend(gen_valid)
            all_failed_results.extend(gen_failed)
            
            # Convert to Candidate objects
            for r in gen_valid:
                all_candidates.append(result_to_candidate(r, {"generation": gen, "is_pareto": False}))
            for r in gen_failed:
                all_candidates.append(result_to_candidate(r, {"generation": gen, "is_pareto": False, "valid": False}))
            
            # Update Pareto front
            pareto = compute_pareto_front(all_valid_results, objectives)
            
            # Update performance insights
            if all_valid_results:
                performance_insights = self._update_performance_insights(
                    previous_insights=performance_insights,
                    valid_results=all_valid_results,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            
            self._log(f"Generation {gen}: {len(gen_valid)} valid, Pareto size={len(pareto)}")
            
            history.append({
                "gen": gen,
                "valid_count": len(gen_valid),
                "failed_count": len(gen_failed),
                "pareto_size": len(pareto),
            })
            
            prev_pareto = pareto
        
        # ===== Build Final Results =====
        final_pareto = compute_pareto_front(all_valid_results, objectives)
        
        # Mark Pareto candidates
        pareto_configs = {prettify_configuration(r.configuration) for r in final_pareto}
        pareto_candidates: List[Candidate[ConfigT]] = []
        for r in final_pareto:
            pareto_candidates.append(result_to_candidate(r, {"is_pareto": True}))
        
        # Update all_candidates with Pareto status
        for c in all_candidates:
            c_key = prettify_configuration(c.configuration)
            if c_key in pareto_configs:
                c.metadata["is_pareto"] = True
        
        # Select best candidate
        best_result = select_best_candidate(final_pareto, objectives)
        if best_result:
            best = result_to_candidate(best_result, {"is_pareto": True})
        else:
            # Fallback to empty candidate if no valid results
            best = Candidate(configuration={}, objectives={}, metadata={})
        
        self._log(f"\n=== Final Results ===")
        self._log(f"Total valid: {len(all_valid_results)}, Pareto size: {len(final_pareto)}")
        if best_result:
            self._log(f"Best: {best_result.objectives}")
        
        return SearchResult(
            objectives=objectives,
            best=best,
            pareto_front=pareto_candidates,
            all_candidates=all_candidates,
            history=history,
        )
    
    def _run_initial_generation(
        self,
        problem: SearchProblem[ConfigT],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """
        Run the initial generation with regeneration loops.
        
        Returns:
            (valid_results, failed_results, updated_constraint_instruction)
        """
        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []
        
        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if regen_round == 0:
                # Generate initial candidates
                candidates = self._generate_initial_candidates(
                    n_candidates=self.candidates_per_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            else:
                # Regenerate using failure insights
                candidates = self._regenerate_candidates(
                    failed_results=last_round_failed,
                    n_candidates=self.candidates_per_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                    constraint_instruction=constraint_instruction,
                    performance_insights=performance_insights,
                )
            
            # Evaluate batch
            valid_batch, failed_batch = self._evaluate_batch(problem, candidates, objectives)
            
            self._log(f"  Batch {regen_round + 1}: {len(valid_batch)} valid, {len(failed_batch)} failed")
            
            # Generate failure insights
            if failed_batch:
                insights = self._generate_failure_insights(
                    failed_results=failed_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
                for r, insight in zip(failed_batch, insights):
                    r.insight = insight
                gen_failed.extend(failed_batch)
            
            population_valid.extend(valid_batch)
            last_round_failed = failed_batch
            
            # Update constraint instruction if we have failures
            if gen_failed and (not constraint_instruction or len(valid_batch) > 0):
                sampled_failures = sample_failed_for_constraint(
                    last_round_failed, gen_failed, self.max_failed_examples
                )
                constraint_instruction = self._generate_constraint_instruction(
                    failed_results=sampled_failures,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            
            regen_round += 1
            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")
            
            if len(population_valid) >= self.pop_size:
                break
        
        # Trim to population size
        population_valid = population_valid[:self.pop_size]
        
        return population_valid, gen_failed, constraint_instruction
    
    def _run_evolution_generation(
        self,
        problem: SearchProblem[ConfigT],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        prev_pareto: List[CandidateResult],
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """
        Run an evolution generation using Pareto front.
        
        Returns:
            (valid_results, failed_results, updated_constraint_instruction)
        """
        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []
        
        if not prev_pareto:
            # No Pareto front to evolve from, fall back to initial generation
            return self._run_initial_generation(
                problem, objectives, search_space_desc,
                all_failed_results, constraint_instruction, performance_insights,
            )
        
        # Generate offspring from Pareto front
        candidates = self._generate_offspring(
            pareto_results=prev_pareto,
            n_candidates=self.pop_size,
            objectives=objectives,
            search_space_desc=search_space_desc,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
        )
        
        # Evaluate batch
        valid_batch, failed_batch = self._evaluate_batch(problem, candidates, objectives)
        
        self._log(f"  Offspring: {len(valid_batch)} valid, {len(failed_batch)} failed")
        
        # Generate failure insights
        if failed_batch:
            insights = self._generate_failure_insights(
                failed_results=failed_batch,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )
            for r, insight in zip(failed_batch, insights):
                r.insight = insight
            gen_failed.extend(failed_batch)
        
        population_valid.extend(valid_batch)
        last_round_failed = failed_batch
        
        # Regeneration loop if needed
        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if not last_round_failed:
                break
            
            # Regenerate using failure insights
            candidates = self._regenerate_offspring(
                failed_results=last_round_failed,
                pareto_results=prev_pareto,
                n_candidates=self.candidates_per_batch,
                objectives=objectives,
                search_space_desc=search_space_desc,
                constraint_instruction=constraint_instruction,
                performance_insights=performance_insights,
            )
            
            valid_batch, failed_batch = self._evaluate_batch(problem, candidates, objectives)
            
            self._log(f"  Regen {regen_round + 1}: {len(valid_batch)} valid, {len(failed_batch)} failed")
            
            if failed_batch:
                insights = self._generate_failure_insights(
                    failed_results=failed_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
                for r, insight in zip(failed_batch, insights):
                    r.insight = insight
                gen_failed.extend(failed_batch)
                
                # Update constraint instruction
                sampled_failures = sample_failed_for_constraint(
                    failed_batch, gen_failed, self.max_failed_examples
                )
                constraint_instruction = self._update_constraint_instruction(
                    previous_instruction=constraint_instruction,
                    failed_results=sampled_failures,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            
            population_valid.extend(valid_batch)
            last_round_failed = failed_batch
            regen_round += 1
            
            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")
        
        population_valid = population_valid[:self.pop_size]
        
        return population_valid, gen_failed, constraint_instruction
    
    def _evaluate_batch(
        self,
        problem: SearchProblem[ConfigT],
        candidates: List[ConfigT],
        objectives: Sequence[ObjectiveSpec],
    ) -> Tuple[List[CandidateResult], List[CandidateResult]]:
        """
        Evaluate a batch of candidates.
        
        Returns:
            (valid_results, failed_results)
        """
        valid_results: List[CandidateResult] = []
        failed_results: List[CandidateResult] = []
        
        for idx, config in enumerate(candidates):
            try:
                # Debug: log the candidate being evaluated
                if self.verbose:
                    self._log(f"    Candidate {idx+1}: {prettify_configuration(config)[:200]}...")
                
                if not problem.validate(config):
                    error_msg = "Validation failed"
                    if self.verbose:
                        # Try to get more details about why validation failed
                        self._log(f"    -> FAILED: {error_msg}")
                    failed_results.append(CandidateResult(
                        configuration=config,
                        objectives={s.name: (0.0 if s.goal == "max" else self.invalid_penalty) for s in objectives},
                        is_valid=False,
                        error_message=error_msg,
                    ))
                    continue
                
                obj = problem.evaluate(config)
                if self.verbose:
                    self._log(f"    -> VALID: {obj}")
                valid_results.append(CandidateResult(
                    configuration=config,
                    objectives=obj,
                    is_valid=True,
                ))
            except Exception as e:
                if self.verbose:
                    self._log(f"    -> EXCEPTION: {e}")
                failed_results.append(CandidateResult(
                    configuration=config,
                    objectives={s.name: (0.0 if s.goal == "max" else self.invalid_penalty) for s in objectives},
                    is_valid=False,
                    error_message=str(e),
                ))
        
        return valid_results, failed_results
    
    # ===== LLM Interaction Methods =====
    
    def _generate_initial_candidates(
        self,
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> List[ConfigT]:
        """Generate initial candidate configurations using LLM."""
        template = f"""You are an optimization expert generating candidates for a multi-objective optimization problem.

{search_space_desc}

Generate exactly {n_candidates} different configuration candidates that:
1. Are diverse and explore different regions of the search space
2. Are likely to be valid (satisfy any constraints)
3. Trade off between different objectives

Return the configurations as a list of dictionaries."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "candidates": List[Dict[str, Any]],
            },
        )
        
        candidates = getattr(result, "candidates", [])
        if self.verbose:
            self._log(f"  LLM generated {len(candidates)} candidates")
            for i, c in enumerate(candidates[:2]):  # Show first 2
                self._log(f"    Sample {i+1}: {str(c)[:300]}...")
        return self._parse_candidates(candidates, n_candidates)
    
    def _regenerate_candidates(
        self,
        failed_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[ConfigT]:
        """Regenerate candidates using failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        
        template = f"""You are an optimization expert. Previous candidates failed validation. Learn from the failures and generate better candidates.

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
        
        result = self._llm_call(
            template=template,
            output_schema={
                "candidates": List[Dict[str, Any]],
            },
        )
        
        candidates = getattr(result, "candidates", [])
        return self._parse_candidates(candidates, n_candidates)
    
    def _generate_offspring(
        self,
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[ConfigT]:
        """Generate offspring from Pareto front configurations."""
        pareto_str = prettify_results(pareto_results[:5], objectives)  # Limit to top 5
        
        template = f"""You are an optimization expert. Generate offspring candidates based on high-quality Pareto-optimal configurations.

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
        
        result = self._llm_call(
            template=template,
            output_schema={
                "candidates": List[Dict[str, Any]],
            },
        )
        
        candidates = getattr(result, "candidates", [])
        return self._parse_candidates(candidates, n_candidates)
    
    def _regenerate_offspring(
        self,
        failed_results: List[CandidateResult],
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[ConfigT]:
        """Regenerate offspring using both Pareto front and failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        pareto_str = prettify_results(pareto_results[:3], objectives)  # Top 3
        
        template = f"""You are an optimization expert. Some offspring candidates failed. Learn from the failures while using the Pareto front as guidance.

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
        
        result = self._llm_call(
            template=template,
            output_schema={
                "candidates": List[Dict[str, Any]],
            },
        )
        
        candidates = getattr(result, "candidates", [])
        return self._parse_candidates(candidates, n_candidates)
    
    def _generate_failure_insights(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> List[str]:
        """Generate insights for why candidates failed."""
        failed_str = prettify_results(failed_results, objectives)
        
        template = f"""You are an optimization expert. Analyze why these candidates failed and provide specific insights.

{search_space_desc}

FAILED CANDIDATES:
{failed_str}

For each failed candidate, provide a specific insight about:
1. What constraint or requirement it violated
2. How to fix it in future candidates

Return a list of exactly {len(failed_results)} insight strings, one for each failed candidate."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "insights": List[str],
            },
        )
        
        insights = getattr(result, "insights", [])
        # Ensure we have the right number of insights
        while len(insights) < len(failed_results):
            insights.append("Unknown failure reason")
        return insights[:len(failed_results)]
    
    def _generate_constraint_instruction(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate consolidated constraint instructions from failures."""
        failed_str = prettify_results(failed_results, objectives)
        
        template = f"""You are an optimization expert. Based on these failed candidates, create a consolidated set of constraint instructions.

{search_space_desc}

FAILED CANDIDATES AND INSIGHTS:
{failed_str}

Create a clear, actionable set of instructions that future candidates should follow to avoid these failures. Return a detailed paragraph describing how to satisfy constraints when proposing configurations."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "constraint_instruction": str,
            },
        )
        
        return getattr(result, "constraint_instruction", "")
    
    def _update_constraint_instruction(
        self,
        previous_instruction: str,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Update constraint instructions with new failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        
        template = f"""You are an optimization expert. Update the constraint instructions based on new failures.

PREVIOUS CONSTRAINT INSTRUCTIONS:
{previous_instruction}

NEW FAILED CANDIDATES:
{failed_str}

Update the constraint instructions to incorporate insights from these new failures. Return an updated, comprehensive set of constraint instructions."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "updated_instruction": str,
            },
        )
        
        return getattr(result, "updated_instruction", previous_instruction)
    
    def _generate_performance_insights(
        self,
        valid_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate performance insights from valid results."""
        stats = compute_performance_stats(valid_results, objectives)
        if not stats:
            return ""
        
        # Format stats for the prompt
        stats_lines = []
        for spec in objectives:
            best = stats.get(f'best_{spec.name}')
            worst = stats.get(f'worst_{spec.name}')
            if best:
                stats_lines.append(f"Best {spec.name}: {best.objectives.get(spec.name, 'N/A')}")
                stats_lines.append(f"  Config: {prettify_configuration(best.configuration)}")
            if worst:
                stats_lines.append(f"Worst {spec.name}: {worst.objectives.get(spec.name, 'N/A')}")
        
        top_pareto = stats.get('top_3_pareto', [])
        if top_pareto:
            stats_lines.append("\nTop Pareto configurations:")
            stats_lines.append(prettify_results(top_pareto, objectives))
        
        stats_str = "\n".join(stats_lines)
        
        template = f"""You are an optimization expert. Analyze the performance patterns and provide insights for generating better candidates.

{search_space_desc}

PERFORMANCE STATISTICS:
{stats_str}

Analyze:
1. What patterns make configurations perform well?
2. What trade-offs exist between objectives?
3. What configuration choices lead to good overall performance?

Return a detailed analysis of what makes configurations perform well and how to generate better candidates."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "performance_insights": str,
            },
        )
        
        return getattr(result, "performance_insights", "")
    
    def _update_performance_insights(
        self,
        previous_insights: str,
        valid_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Update performance insights with new results."""
        stats = compute_performance_stats(valid_results, objectives)
        if not stats:
            return previous_insights
        
        top_pareto = stats.get('top_3_pareto', [])
        pareto_str = prettify_results(top_pareto, objectives) if top_pareto else "None"
        
        template = f"""You are an optimization expert. Update the performance insights based on new results.

PREVIOUS INSIGHTS:
{previous_insights}

CURRENT TOP PARETO CONFIGURATIONS:
{pareto_str}

TOTAL VALID CANDIDATES: {len(valid_results)}
PARETO FRONT SIZE: {stats.get('pareto_size', 0)}

Update the insights with any new patterns or observations. Return updated performance insights incorporating the new results."""
        
        result = self._llm_call(
            template=template,
            output_schema={
                "updated_insights": str,
            },
        )
        
        return getattr(result, "updated_insights", previous_insights)
    
    def _parse_candidates(
        self,
        candidates: Any,
        expected_count: int,
    ) -> List[ConfigT]:
        """Parse and validate candidate configurations from LLM output."""
        if not isinstance(candidates, list):
            self._log(f"Warning: LLM returned non-list candidates: {type(candidates)}")
            return []
        
        parsed: List[ConfigT] = []
        for c in candidates:
            if isinstance(c, dict):
                parsed.append(c)
            elif isinstance(c, str):
                # Try to parse as JSON
                try:
                    import json
                    parsed.append(json.loads(c))
                except:
                    self._log(f"Warning: Could not parse candidate string: {c[:100]}")
        
        if len(parsed) != expected_count:
            self._log(f"Warning: Expected {expected_count} candidates, got {len(parsed)}")
        
        return parsed

