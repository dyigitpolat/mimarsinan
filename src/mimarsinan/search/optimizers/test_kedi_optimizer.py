"""
Test script for KediOptimizer.

This script tests the Kedi optimizer with a simple mock problem
to verify the implementation works correctly before using it
with the full JointArchHwProblem.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Sequence

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import ObjectiveSpec


@dataclass
class MockMultiObjectiveProblem:
    """
    A simple mock problem for testing the Kedi optimizer.
    
    Search space:
    - x: float in [0, 10]
    - y: float in [0, 10]
    
    Objectives:
    - f1: minimize (x - 5)^2 + (y - 5)^2  (minimize distance from (5,5))
    - f2: maximize x + y  (maximize sum)
    
    These objectives conflict: f1 wants (5,5), f2 wants (10,10).
    """
    
    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return [
            ObjectiveSpec("distance_from_center", "min"),
            ObjectiveSpec("sum_of_coords", "max"),
        ]
    
    def validate(self, configuration: Dict[str, Any]) -> bool:
        """Check if configuration is valid."""
        try:
            x = float(configuration.get("x", 0))
            y = float(configuration.get("y", 0))
            return 0 <= x <= 10 and 0 <= y <= 10
        except (TypeError, ValueError):
            return False
    
    def evaluate(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the objectives for a configuration."""
        x = float(configuration["x"])
        y = float(configuration["y"])
        
        # f1: distance from (5, 5) - minimize
        f1 = (x - 5) ** 2 + (y - 5) ** 2
        
        # f2: sum of coordinates - maximize
        f2 = x + y
        
        return {
            "distance_from_center": f1,
            "sum_of_coords": f2,
        }
    
    def meta(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        return {}


def test_kedi_optimizer_with_mock():
    """Test KediOptimizer with the mock problem."""
    from mimarsinan.search.optimizers.kedi_optimizer import KediOptimizer
    
    problem = MockMultiObjectiveProblem()
    
    # Create optimizer with small parameters for quick testing
    optimizer = KediOptimizer(
        pop_size=4,
        generations=2,
        candidates_per_batch=3,
        max_regen_rounds=3,
        max_failed_examples=3,
        model="openai:gpt-4o-mini",  # Use cheaper model for testing
        adapter_type="pydantic",
        config_schema={
            "x": "float between 0 and 10",
            "y": "float between 0 and 10",
        },
        example_config={"x": 5.0, "y": 5.0},
        constraints_description="x and y must both be between 0 and 10 (inclusive).",
        verbose=True,
    )
    
    print("=" * 70)
    print("Testing KediOptimizer with MockMultiObjectiveProblem")
    print("=" * 70)
    
    result = optimizer.optimize(problem)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nObjectives: {[o.name for o in result.objectives]}")
    print(f"Best configuration: {result.best.configuration}")
    print(f"Best objectives: {result.best.objectives}")
    print(f"Pareto front size: {len(result.pareto_front)}")
    print(f"Total candidates evaluated: {len(result.all_candidates)}")
    
    print("\nPareto front:")
    for i, c in enumerate(result.pareto_front):
        print(f"  {i+1}. config={c.configuration}, obj={c.objectives}")
    
    print("\nHistory:")
    for h in result.history:
        print(f"  Gen {h['gen']}: valid={h['valid_count']}, failed={h['failed_count']}, pareto={h['pareto_size']}")
    
    return result


def test_support_functions():
    """Test the support functions independently."""
    from mimarsinan.search.optimizers.kedi_optimizer_support import (
        CandidateResult,
        compute_pareto_front,
        compute_performance_stats,
        dominates,
        prettify_configuration,
        prettify_objectives,
        prettify_results,
        select_best_candidate,
    )
    
    print("=" * 70)
    print("Testing support functions")
    print("=" * 70)
    
    objectives = [
        ObjectiveSpec("f1", "min"),
        ObjectiveSpec("f2", "max"),
    ]
    
    # Test dominates
    # f1 is minimize, f2 is maximize
    a = {"f1": 1.0, "f2": 10.0}  # Best on both
    b = {"f1": 2.0, "f2": 5.0}   # Worse on both
    c = {"f1": 1.0, "f2": 5.0}   # Equal on f1, worse on f2
    d = {"f1": 0.5, "f2": 8.0}   # Better on f1, worse on f2 (trade-off)
    
    assert dominates(a, b, objectives), "a should dominate b (better on both)"
    assert not dominates(b, a, objectives), "b should not dominate a"
    assert dominates(a, c, objectives), "a should dominate c (equal f1, better f2)"
    assert not dominates(c, a, objectives), "c should not dominate a"
    assert not dominates(a, d, objectives), "a should not dominate d (trade-off: d better on f1)"
    assert not dominates(d, a, objectives), "d should not dominate a (trade-off: a better on f2)"
    print("✓ dominates() works correctly")
    
    # Test Pareto front computation
    results = [
        CandidateResult({"x": 1}, {"f1": 1.0, "f2": 10.0}, True),  # Pareto
        CandidateResult({"x": 2}, {"f1": 2.0, "f2": 5.0}, True),   # Dominated by first
        CandidateResult({"x": 3}, {"f1": 0.5, "f2": 8.0}, True),   # Pareto
        CandidateResult({"x": 4}, {"f1": 3.0, "f2": 12.0}, True),  # Pareto
    ]
    
    pareto = compute_pareto_front(results, objectives)
    assert len(pareto) == 3, f"Expected 3 Pareto points, got {len(pareto)}"
    print(f"✓ compute_pareto_front() works correctly (found {len(pareto)} Pareto points)")
    
    # Test prettify functions
    print("\nPrettified objectives:")
    print(prettify_objectives(objectives))
    
    print("\nPrettified configuration:")
    print(prettify_configuration({"x": 1.0, "y": 2.0}))
    
    print("\nPrettified results (first 2):")
    print(prettify_results(results[:2], objectives))
    
    # Test performance stats
    stats = compute_performance_stats(results, objectives)
    assert stats is not None, "Stats should not be None"
    assert "best_f1" in stats, "Stats should have best_f1"
    assert "pareto_front" in stats, "Stats should have pareto_front"
    print(f"✓ compute_performance_stats() works correctly (pareto_size={stats['pareto_size']})")
    
    # Test best selection
    best = select_best_candidate(pareto, objectives)
    assert best is not None, "Best should not be None"
    print(f"✓ select_best_candidate() works correctly (best={best.configuration})")
    
    print("\n✓ All support function tests passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KediOptimizer")
    parser.add_argument("--full", action="store_true", help="Run full optimizer test (requires API key)")
    parser.add_argument("--support-only", action="store_true", help="Only test support functions")
    args = parser.parse_args()
    
    # Always test support functions
    test_support_functions()
    
    if args.support_only:
        print("\nSkipping full optimizer test (--support-only)")
    elif args.full:
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nWarning: OPENAI_API_KEY not set. Full test may fail.")
        test_kedi_optimizer_with_mock()
    else:
        print("\nTo run full optimizer test, use --full flag")
        print("(Requires OPENAI_API_KEY environment variable)")

