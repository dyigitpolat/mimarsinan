"""Resolve the encoding placement of cached flows written before the stamp existed.

``encoding_layer_placement`` is resolved once, at flow birth, and stamped on the
``ModelRepresentation`` so no consumer can read an unplaced marking as a
deployment fact. Artifacts written before that stamp existed deserialize
unstamped — and unstamped is exactly what the guard refuses, so every
pre-change run directory became unresumable (``start_step``/``stop_step`` are
first-class).

The fix is to APPLY the placement, not to wave the flow through: a legacy
native flow genuinely never had one applied. This runs at cache load, before
any step of this run executes, which is the only point where applying it is
safe — the late validity gate runs AFTER the negative-boundary subsume-forward
policy, and re-marking there would erase host placements the deployment
depends on.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, cast

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.packaging_contract import PackagingContract
from mimarsinan.torch_mapping.encoding_layers import (
    resolve_unstamped_encoding_placement,
)


def _mapper_repr_of(entry: Any) -> Optional[ModelRepresentation]:
    """The entry's ``ModelRepresentation``, or ``None`` if it is not a flow."""
    get_mapper_repr = getattr(entry, "get_mapper_repr", None)
    if not callable(get_mapper_repr):
        return None
    return cast(Callable[[], ModelRepresentation], get_mapper_repr)()


def resolve_cached_flow_placements(
    cache, *, placement: str, packaging: PackagingContract,
) -> List[str]:
    """Apply ``placement`` to every cached flow that carries no placement stamp.

    Returns the cache keys it resolved (empty for a fresh run, and for any
    directory whose artifacts were already written with a stamp — so this is a
    no-op on everything but a legacy resume).
    """
    resolved: List[str] = []
    for key in list(cache.keys()):
        mapper_repr = _mapper_repr_of(cache.get(key))
        if mapper_repr is None:
            continue
        if resolve_unstamped_encoding_placement(
            mapper_repr, placement=placement, packaging=packaging,
        ):
            resolved.append(key)
    return resolved
