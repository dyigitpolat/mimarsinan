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


#: How a deserialized artifact from an older structural layout fails to yield
#: its mapper graph: a field the class no longer/not yet has, an abstract base
#: that never implemented the accessor, a signature that moved. Anything else is
#: a code bug and propagates untouched. (Sibling of
#: ``load_store_strategies.ENTRY_LOAD_FAILURES``, one level up the same path.)
STALE_ARTIFACT_FAILURES = (AttributeError, NotImplementedError, TypeError, KeyError)


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
        try:
            mapper_repr = _mapper_repr_of(cache.get(key))
        except STALE_ARTIFACT_FAILURES as exc:
            # Loud and actionable: this runs at pipeline construction, so a bare
            # traceback here would name no artifact.
            raise RuntimeError(
                f"cache entry {key!r} looks like a mapper flow but its mapper "
                f"graph could not be read ({exc!r}), so its encoding placement "
                "cannot be resolved. The artifact predates a structural change; "
                "quarantine it so the producing step re-runs "
                f"(PipelineCache.quarantine_entry(<run dir>, {key!r}))."
            ) from exc
        if mapper_repr is None:
            continue
        if resolve_unstamped_encoding_placement(
            mapper_repr, placement=placement, packaging=packaging,
        ):
            resolved.append(key)
    return resolved
