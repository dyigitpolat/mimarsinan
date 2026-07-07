"""Search optimizers for multi-objective optimization: NSGA2Optimizer, AgentEvolveOptimizer, CompilagentOptimizer."""

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer

__all__ = ["SearchOptimizer", "NSGA2Optimizer"]

try:
    from mimarsinan.search.optimizers.agent_evolve import AgentEvolveOptimizer
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




