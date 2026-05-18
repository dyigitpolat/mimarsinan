"""Compilagent integration for mimarsinan's architecture search.

Wraps `JointArchHwProblem` as a compilagent ``Backend`` so an LLM-driven
``OptimizationSession`` can search the same NAS + hardware space the
NSGA2 / AgentEvolve optimizers explore, while exposing every
layout-mapping signal (per-softcore counts, latency tiers, full
``LayoutVerificationStats``, multi-objective values) through compilagent's
analysis / artifact / introspection-tool / leaderboard surfaces.

Public entry points:

* ``CompilagentOptimizer`` — the ``SearchOptimizer`` implementation
  exposed alongside ``NSGA2Optimizer`` / ``AgentEvolveOptimizer``.
* ``MimarsinanLayoutBackend`` — the compilagent ``Backend`` the session
  drives; ``entrypoint.register_backend()`` advertises it under the id
  ``"mimarsinan_layout"``.

Importing the package triggers backend registration so callers that load
the optimizer directly do not have to remember the registry side effect.
"""

from __future__ import annotations

from .backend import MimarsinanLayoutBackend
from .compilagent_optimizer import CompilagentOptimizer
from .entrypoint import BACKEND_ID, register_backend

# Side effect: register the backend on first import. Idempotent — safe to
# import the package multiple times in one process.
register_backend()

__all__ = [
    "BACKEND_ID",
    "CompilagentOptimizer",
    "MimarsinanLayoutBackend",
    "register_backend",
]
