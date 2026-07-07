"""Compilagent integration: wraps `JointArchHwProblem` as a compilagent ``Backend`` for LLM-driven architecture search."""

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
