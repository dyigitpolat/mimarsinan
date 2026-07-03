"""Thin delegator rendering a ``SearchSpaceDescription`` to a tuple of compilagent ``Lever``s."""

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
