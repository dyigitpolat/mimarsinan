"""GUI resources SSOT: what a resource is made of, when it is rendered, and where it is cached."""

from mimarsinan.gui.resources.descriptor import ResourceDescriptor
from mimarsinan.gui.resources.policy import (
    ResourceRenderPolicy,
    resolve_resource_render_policy,
)
from mimarsinan.gui.resources.sources import (
    HeatmapSource,
    JsonSource,
    ResourceSource,
    as_host_array,
    encode_resource_payload,
    resource_source_from_state,
)
from mimarsinan.gui.resources.store import ResourceStore

__all__ = [
    "HeatmapSource",
    "JsonSource",
    "ResourceDescriptor",
    "ResourceRenderPolicy",
    "ResourceSource",
    "ResourceStore",
    "as_host_array",
    "encode_resource_payload",
    "resolve_resource_render_policy",
    "resource_source_from_state",
]
