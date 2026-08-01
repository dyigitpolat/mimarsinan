"""The one policy deciding WHEN a run's GUI resources are rendered."""

from __future__ import annotations

from enum import Enum

from mimarsinan.common.env import gui_resource_render_override


class ResourceRenderPolicy(str, Enum):
    """Who pays for rendering a step's heavy resources, and when.

    ``EAGER`` -- a monitor is attached, so the run renders each step's resources
    as it goes and the browser fetches finished bytes. ``DEFERRED`` -- nobody is
    watching, so the run persists only the resources' SOURCE data (cheap array
    and JSON I/O) and whoever attaches later renders from it. A headless run must
    never pay for pixels no one asked for: that backlog is what made the process
    sit for minutes after its last step, holding its scheduler allocation.
    """

    EAGER = "eager"
    DEFERRED = "deferred"


def resolve_resource_render_policy(declared: ResourceRenderPolicy) -> ResourceRenderPolicy:
    """``declared`` unless an operator override names the other policy; raises on a bad name.

    The run mode declares the policy (``--ui`` renders, ``--headless`` defers);
    the environment override exists so a headless run can be told to render
    in-line anyway, and is the only thing allowed to contradict the declaration.
    """
    override = gui_resource_render_override()
    if override is None:
        return declared
    return ResourceRenderPolicy(override)


__all__ = ["ResourceRenderPolicy", "resolve_resource_render_policy"]
