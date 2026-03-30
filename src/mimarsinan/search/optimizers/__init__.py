"""
Search optimizers for multi-objective optimization.

Available optimizers:
- NSGA2Optimizer: Genetic algorithm-based optimizer using pymoo's NSGA-II
- AgentEvolveOptimizer: LLM-based optimizer using pydantic-ai for agentic evolution
"""

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer

# AgentEvolveOptimizer is optional (requires pydantic-ai)
try:
    from mimarsinan.search.optimizers.agent_evolve_optimizer import AgentEvolveOptimizer
    __all__ = ["SearchOptimizer", "NSGA2Optimizer", "AgentEvolveOptimizer"]
except ImportError:
    __all__ = ["SearchOptimizer", "NSGA2Optimizer"]




