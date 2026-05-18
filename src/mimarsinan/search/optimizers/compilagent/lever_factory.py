"""Render a ``SearchSpaceDescription`` to a tuple of compilagent ``Lever``s.

Thin delegating module: the real arithmetic — choosing candidates that
respect ``CORE_DIM_GRANULARITY``, deriving evidence strings — lives on
``SearchSpaceDescription.to_compilagent_levers`` so AgentEvolve and the
compilagent backend pull from one source. This module exists so that
``Backend.derive_search_space`` can call into a small named function
instead of reaching across packages, which makes the call site easier to
read and easier to mock in tests.
"""

from __future__ import annotations

from typing import Any, Tuple

from mimarsinan.search.search_space_description import SearchSpaceDescription


def levers_from_description(
    description: SearchSpaceDescription,
    *,
    workload_id: str,
    backend_id: str,
) -> Tuple[Any, ...]:
    """Return a tuple of ``compilagent.Lever``s describing the search space."""

    return description.to_compilagent_levers(
        workload_id=workload_id, backend_id=backend_id,
    )


__all__ = ["levers_from_description"]
