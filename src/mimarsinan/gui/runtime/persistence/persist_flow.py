"""ONE write path for a completed step's resources, shared by both render policies.

Every run persists each resource's SOURCE (cheap array/JSON I/O; the deferred
reproduction and the on-demand full-resolution variant both feed off it) AND
its UI-resolution artifact (the pure-numpy render is bounded by
``DEFAULT_TARGET_LONG_SIDE``, so a browser attach is a plain file read instead
of a per-request render storm). The policies differ only in who is watching:
EAGER additionally warms the live in-memory store the monitor serves from.
"""

from __future__ import annotations

import logging

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.resources import (
    ResourceDescriptor,
    ResourceRenderPolicy,
    ResourceStore,
    encode_resource_payload,
)
from mimarsinan.gui.runtime.persistence.resource_sources import save_resource_source
from mimarsinan.gui.runtime.persistence.store import save_resource_to_disk

logger = logging.getLogger("mimarsinan.gui")


def persist_step_resources(
    *,
    policy: ResourceRenderPolicy,
    store: "ResourceStore | None",
    working_dir: str | None,
    step_name: str,
    descriptors: list[ResourceDescriptor],
) -> None:
    """Persist sources and UI-resolution artifacts for one completed step."""
    if working_dir:
        for desc in descriptors:
            with best_effort(f"resource source for {desc.kind}/{desc.rid}", logger=logger):
                save_resource_source(
                    working_dir, step_name, desc.kind, desc.rid, desc.source,
                )

    warm_store = store if policy is ResourceRenderPolicy.EAGER else None
    if not working_dir and warm_store is None:
        return
    for desc in descriptors:
        payload = None
        if warm_store is not None:
            payload = warm_store.prewarm(step_name, desc.kind, desc.rid)
        if payload is None:
            produced = False
            with best_effort(f"resource producer for {desc.kind}/{desc.rid}", logger=logger):
                payload = desc.producer()
                produced = True
            if not produced:
                continue
        if not working_dir:
            continue
        encoded = encode_resource_payload(payload, desc.media_type)
        if encoded is None:
            continue
        save_resource_to_disk(
            working_dir, step_name, desc.kind, desc.rid,
            encoded, media_type=desc.media_type,
        )


__all__ = ["persist_step_resources"]
