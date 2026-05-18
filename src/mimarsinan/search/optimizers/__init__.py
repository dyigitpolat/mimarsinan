"""
Search optimizers for multi-objective optimization.

Available optimizers:
- NSGA2Optimizer: Genetic algorithm-based optimizer using pymoo's NSGA-II
- AgentEvolveOptimizer: LLM-based optimizer using pydantic-ai for agentic evolution
- CompilagentOptimizer: LLM-driven session via the compilagent integration
  (registers ``MimarsinanLayoutBackend`` so layout-mapping internals are
  exposed to the agent and the leaderboard becomes truly multi-objective).
"""

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer

__all__ = ["SearchOptimizer", "NSGA2Optimizer"]

try:
    from mimarsinan.search.optimizers.agent_evolve_optimizer import AgentEvolveOptimizer
    __all__.append("AgentEvolveOptimizer")
except ImportError:
    pass

# Compilagent is a hard dependency (see requirements.txt) but we still
# guard the import: a partial install (e.g. compilagent without the
# `pydantic_ai` extra) should not break NSGA2 / AgentEvolve callers.
try:
    from mimarsinan.search.optimizers.compilagent import (
        CompilagentOptimizer,
        MimarsinanLayoutBackend,
    )
    __all__.extend(["CompilagentOptimizer", "MimarsinanLayoutBackend"])
except ImportError:
    pass




